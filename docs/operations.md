# 运维与故障恢复

## 日常状态检查

```bash
akg experiment status -c configs/experiment_910c_kimi_k3.yaml
akg verify-run --experiment-root runs/exp_910c_kimi_k3_v1
npu-smi info
```

systemd：

```bash
systemctl status ascend-kernel-worker.service
systemctl status ascend-kernel-controller.service
journalctl -u ascend-kernel-worker.service --since today --no-pager
journalctl -u ascend-kernel-controller.service --since today --no-pager
```

日志输出只能用于诊断，不能代替 SQLite 状态和已提交 artifact。

## 安全恢复顺序

发生中断后按以下顺序操作：

1. 停止重复启动脚本，确认只有一个 controller 和每卡一个 Worker。
2. 运行 `npu-smi info` 和 `doctor`，确认基础环境仍健康。
3. 运行 `experiment status`，定位最后一个 committed 状态。
4. 运行 `verify-run`，检查 artifact 哈希和引用。
5. 保留现场日志，不编辑数据库或 JSON。
6. 修复基础原因后启动 Worker。
7. 执行 `experiment resume`。

恢复命令：

```bash
akg experiment resume -c configs/experiment_910c_kimi_k3.yaml
```

`resume` 应重复读取已有 Prompt、模型响应和阶段结果，只执行尚未 committed 的步骤。若进程是在模型 HTTP 请求途中被杀死，而响应尚未原子提交，恢复时可能再次发起同一模型请求；这也是为什么模型调用不能直接产生外部副作用。

## 分阶段故障

### Model/AIPing

检查 controller 日志和该轮：

```text
prompt.json
raw_response.json
model_response.json
model_failure.json
```

- 连接、限流、5xx、CLI 异常退出：有界退避，同一轮重试。
- 输出截断：只有完整、可解析、符合 Schema 的响应才提交。
- 格式修复耗尽：记录本轮模型失败，下一轮仍可继续。
- 401/403：停止 controller，修复 `/etc/ascend-kernel-lab/controller.env`，不要把 token 打到日志。
- 404/model not found：核对 AIPing 映射和 `ANTHROPIC_MODEL`，不修改任务 Prompt 规避。

### Source/compile

- Source 失败：查看 AST 违规列表；该轮不会启动 NPU 编译。
- Compile 失败：查看阶段 stderr、`stage_result.json`、IR 和错误 stage/source location。
- 编译超时：确认不是另一个进程持卡，再检查 shape/config 数量；不要直接移除 timeout。
- 编译缓存只属于该候选尝试；不要清理全局 Triton cache 来影响其他实验。

### Correctness

正确性失败不会 benchmark。重点检查：dtype、shape、mask、归约精度、非对齐尾部、输入是否修改、NaN/Inf、地址偏移和 canary。隐藏失败只报告脱敏结论；不得把 shape 和具体错误元素反馈给下一轮或手工选择候选。

### Benchmark

CV 高时：

- 查同卡其他进程；
- 查温度、功耗、频率和降频；
- 查 CPU contention、NUMA 和 host enqueue 抖动；
- 确认 baseline/candidate 交替测量；
- 让系统按配置重测一次。

仍不稳定则保留原始批次并标记 unstable，不手工挑最快样本。

### Profiler

`msprof` 失败不删除已经完成的正确性和延迟。比较环境探测时的 `msprof op --help` 与实际命令，检查 CANN 版本导致的 CSV 列名或输出目录变化。解析器不认识的指标保持 `null`，把脱敏 fixture 加入回归测试后再扩展 parser。

### NPU runtime/失联

1. 阶段监督器终止整个候选进程组。
2. 等待设备健康检查。
3. 确认设备锁是否由活进程持有。
4. 若 `npu-smi` 仍异常，停止 Worker，把设备标记不可用并通知平台运维。
5. 驱动/设备恢复后先执行 smoke/acceptance 的 K01 子集，再恢复队列。

不要让自动脚本重载驱动、重启主机或删除设备节点。

## SQLite 与 artifact

- SQLite、`-wal`、`-shm` 和 artifact 必须作为一致整体备份。
- Controller/Worker 是不同 UID、相同部署组：共享状态根和控制面目录必须为
  setgid `02770`，SQLite/WAL/SHM 与 committed artifact 必须为 `0660`，other 权限为
  零；两个服务的有效 umask 必须为 `0007`。
- candidate `worker_jobs/.../attempt_*` 及其 work/cache/IR 子树必须保持 `0700`。
  不要通过递归放宽权限来解决队列报错；旧部署只按部署指南的停机、剪枝流程修复。
- 在线运行时不要直接复制数据库作为可恢复备份；先停止所有写者或使用 SQLite 官方一致性备份方法。
- `.tmp` 文件表示未提交写入，可保留用于取证；控制器不应把它当正式 artifact。
- 正式 artifact 哈希失败时停止恢复，复制现场到新目录后调查；不要覆盖文件来“修好”哈希。
- 数据库应位于本地文件系统。NFS 上的锁和 rename 语义不可作为默认部署。

权限告警时先同时停止两个服务，再检查路径的每一级目录、属组和模式：

```bash
namei -l /opt/ascend-kernel-lab/runs/metadata.db
stat -c '%A %a %U %G %n' /opt/ascend-kernel-lab/runs/metadata.db*
systemctl show ascend-kernel-controller.service ascend-kernel-worker.service \
  -p User -p Group -p UMask
```

不要只删除 `-wal`/`-shm` 来“修复”权限；它们是数据库一致性状态的一部分。若 owner
来自意外账号，先保留现场并按部署指南由 root 修复 group/mode，再进行完整性验证。

## 磁盘容量

IR 和 `msprof` raw 往往远大于 JSON。监控：

```bash
du -sh runs
df -h runs
```

本项目不自动删除实验。归档前先结束实验并运行 `verify-run`，记录目录级校验值，再移动整个实验目录。SQLite 引用的相对 artifact 路径必须保持一致。

## 停止服务

优先停止 controller，让当前 stage 完成，再停止 Worker：

```bash
sudo systemctl stop ascend-kernel-controller.service
sudo systemctl stop ascend-kernel-worker.service
```

紧急停止可能使当前 stage 未提交；恢复机制会重做该 stage。不要使用递归删除或强制清数据库作为停止手段。

## 配置或代码变化

以下变化要求重新 probe/baseline：

- 驱动、固件、CANN、torch、torch_npu、triton；
- SoC 或设备；
- task spec/reference/validator；
- benchmark 参数；
- harness Git commit 中会影响执行或计时的代码。

不要复用 identity hash 不匹配的 baseline。新的实验 ID 应对应一份冻结配置快照。
