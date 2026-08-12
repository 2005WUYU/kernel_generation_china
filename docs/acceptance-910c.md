# 910C 验收清单

验收分为自动检查、人工证据检查和破坏性故障演练。目标是证明流水线可靠、可恢复、可追溯；它不要求某次 Kimi 输出一定获得指定 speedup。

## 前置条件

- 使用全新 Git checkout，并记录 `git rev-parse HEAD`。
- `git status --short` 为空。
- 驱动、固件、CANN、torch、torch_npu、Triton-Ascend 已由平台团队确认兼容。
- 当前卡没有其他 benchmark、训练、推理或 profiler 任务。
- venv 由 `scripts/install-910c.sh` 创建。
- 自动汇总命令需要 controller 环境，以便 `doctor` 确认 AIPing 引用和隐藏 seed 已配置；它不会调用模型。
- 输出报告位于本地磁盘，且目标文件尚不存在。

## 自动证据汇总

以下环境文件为 root-only，命令应在受控 root 维护 shell 中执行。汇总器不调用模型或运行候选；它只检查配置、Git、显式 gate evidence 和既有 run。

```bash
export AKG_CANN_ENV_FILE=/usr/local/Ascend/ascend-toolkit/set_env.sh
export AKG_VENV_DIR=/opt/ascend-kernel-lab-venv
export AKG_CONFIG_PATH=/opt/ascend-kernel-lab/configs/experiment_910c_kimi_k3.yaml
export AKG_ACCEPTANCE_OUTPUT=/opt/ascend-kernel-lab/runs/acceptance_report_001.json
export AKG_ACCEPTANCE_EVIDENCE_ROOT=/opt/ascend-kernel-lab/runs/acceptance_evidence
# 若实验 run/resume 使用过 --experiment-id，再设置同一个值：
# export AKG_EXPERIMENT_ID=exp_910c_kimi_k3_release_001
set -a
. /etc/ascend-kernel-lab/controller.env
. /etc/ascend-kernel-lab/hidden.env
set +a
./scripts/acceptance-910c.sh
```

等价底层入口：

```bash
akg acceptance \
  -c "$AKG_CONFIG_PATH" \
  --evidence-root "$AKG_ACCEPTANCE_EVIDENCE_ROOT" \
  -o "$AKG_ACCEPTANCE_OUTPUT"
```

脚本拒绝覆盖已有报告。`akg acceptance` 聚合 G0–G8 显式证据、`doctor` 和已完成实验的 `verify-run` 结果；证据缺失时对应 gate 为 `pending`。它不会自动运行 probe、baseline、100 次循环、进程 kill、设备故障、主机重启或升级演练。以下项目是完整发布门禁，其中只有一部分能由汇总命令自动证明。

## G0–G8 证据契约

G1（干净 clone）和 G2（doctor）由汇总器直接计算。其他 gate 必须在 `--evidence-root` 下提供以下精确文件名：

| Gate | 文件 | 内容范围 |
| --- | --- | --- |
| G0 | `g0_release.json` | 发布 commit、配置冻结、CPU CI 与变更审批 |
| G3 | `g3_worker_smoke.json` | Worker smoke、100 次 K01、编译/runtime 故障恢复 |
| G4 | `g4_crash_concurrency.json` | crash、lease fencing、幂等、单卡并发与恢复 |
| G5 | `g5_functional_security.json` | 十任务功能、公开/隐藏隔离、反绕过与 stage 环境 |
| G6 | `g6_measurement.json` | 计时稳定性、baseline identity、profiler 覆盖 |
| G7 | `g7_real_model.json` | 真实 AIPing/Kimi 五轮、历史最佳和最终隐藏 gate |
| G8 | `g8_reboot_upgrade.json` | drain、重启、升级、数据库迁移与回滚演练 |

每个文件必须是普通非 symlink JSON，不超过 16 MiB，格式如下：

```json
{
  "schema_version": "ascend_acceptance_gate_v1",
  "gate": "G3",
  "experiment_id": "exp_910c_kimi_k3_v1",
  "harness_git_commit": "exact-clean-git-commit",
  "status": "pass",
  "checks": [
    {
      "name": "k01_100_consecutive_runs",
      "status": "pass",
      "evidence": "path-or-non-secret-summary"
    }
  ]
}
```

`gate` 和文件名必须匹配，`experiment_id` 必须与 `--experiment-id` 或配置 ID 一致，`harness_git_commit` 必须等于当前干净 checkout 的 `git rev-parse HEAD`。`checks` 非空且每项 `status` 都必须为 `pass`。证据中不得包含 token、authorization、password、secret 值或真实隐藏输入。

G7 即使自身文件为 pass，也必须同时通过 `verify-run`，否则汇总器会判 fail。不要手写虚假 pass：每项检查应由相应命令输出、日志哈希或经过批准的演练记录产生。

## 必须通过的验收项

### 环境

- `npu-smi info` 可用，选定 `npu:0` 存在。
- Python 导入的 torch、torch_npu、triton 路径来自预期系统栈/venv 可见路径。
- `torch.npu.is_available()` 为真。
- CANN、BiSheng 和 `msprof op` 能力有真实探测结果。
- `env_manifest.json`、`capability_matrix.json`、`profiler_capabilities.json` 可解析并带 schema/version 信息。

### Triton 功能 smoke

- Vector Add 正确处理整齐和非整齐长度。
- fp16、bf16、fp32 的支持结果分别记录。
- mask、二维 grid、`tl.sum/max/exp/dot` 各自真实编译并运行，失败项保留阶段和错误。
- 多 kernel case 可被 profiler 识别。

不是所有特性都必须在所有版本可用，但配置启用的十任务所需特性必须可用；否则验收失败，而不是跳过后声称成功。

### 计时与 profiler

- 自动选择的 NPU 计时方法 verified。
- 空操作开销、样本数和 CV 为有限数。
- K01 候选预热后可稳定测量，CV 超阈值会重测。
- profiler 在独立进程运行。
- 原始 `msprof` 产物和统一 summary 同时存在。
- summary 中不可用指标为 `null`，不是零或推测值。
- 至少识别一个实际候选 Triton kernel，并计算 coverage。

### 隔离与恢复

- K01 固定正确候选可连续执行 100 次。
- 注入 Python 语法错误只使 source 阶段失败。
- 注入编译错误不使 Worker 退出，下一 job 可执行。
- 注入 runtime error 后完成设备健康检查，下一 job 可执行。
- correctness 失败不会运行 benchmark/profile。
- profiler 失败仍保留 correctness 和 benchmark。
- stage timeout 终止整个子进程组，没有遗留进程或永久设备锁。
- 重复提交相同 job 幂等；过期 lease 可重新入队，旧 Worker 无法再提交结果。
- 中断 controller 后执行 `experiment resume` 不重复已提交模型调用或评测阶段。

### 安全

- 候选无法导入文件、网络、subprocess、ctypes 和动态导入接口。
- 高层 PyTorch 计算、CPU/Numpy fallback 和输入修改会被拒绝。
- Worker stage 环境不包含 token、key、proxy 或 hidden seed。
- Claude CLI 使用无工具、无会话、隔离 HOME/cwd 的一次性模式。
- 模型 Prompt 和候选 cwd 均不可读取部署端 hidden seed 或真实隐藏 case。
- 输出真实生成且同步，候选 kernel 设备时间覆盖率满足任务策略。

### 数据与导出

- 所有 committed artifact 哈希正确。
- SQLite 状态、事件和 `events.jsonl` 投影一致。
- 历史最佳未被第五轮退化代码覆盖。
- 最终只评测公开排名最高候选；隐藏失败不会试探其他候选。
- `verify-run` 通过。
- SFT、RL 和 report 导出成功。
- 导出中没有凭据、hidden seed、隐藏 shape 或私有错误位置。

## 模型端 E2E

基础硬件检查完成后，在 controller 专用环境运行 K01 五轮。生产默认是独立 Worker + durable queue：

生产由 systemd controller 读取 root-only 环境文件，并通过 durable queue 与独立 Worker 协作：

```bash
sudo systemctl start ascend-kernel-controller.service
journalctl -u ascend-kernel-controller.service -f
```

若验收只覆盖 K01，应使用单独配置文件或在受控维护运行中给 `scripts/run-controller.sh` 传入 `--task k01_vector_add`；不要把真实 token 复制到可读 shell 脚本。

`--with-local-worker` 仅可用于受控调试；它把 Worker 线程放进持有凭据的 controller 进程，不满足生产隔离要求。`--direct` 仅用于后端诊断，不计入 durable queue 恢复验收。

人工确认：

- 每轮都有 Prompt、原始响应、Schema 后响应、完整 candidate、evaluation 和 feedback。
- AIPing 请求失败在同一轮重试，不产生额外优化轮。
- JSON 格式修复次数不超过配置。
- `no_change` 仍返回完整源码；字节一致时复用评测结果。
- 第五轮完成后保存最佳轮次并进行一次隐藏最终评测。
- raw response 和日志中没有 token。

## 证据包

验收完成后保留：

```text
git commit 与 dirty 状态
配置的脱敏副本
三份环境/能力 manifest
K01 baseline snapshot
acceptance summary
events.jsonl
关键失败注入结果
profile summary 与原始文件目录哈希
verify-run 结果
SFT/RL/report 的哈希与行数
```

证据包可以归档到受控存储，但不得直接提交公开 Git。传输前按 [security.md](security.md) 做脱敏。

## 验收失败准则

以下任一情况不得进入正式十任务实验：

- 环境探测失败但被手工填值绕过；
- B0 eager baseline 不可用或版本身份不匹配；
- correctness/benchmark 未强制同步；
- profiler 与普通计时共进程；
- Worker 能看到模型 token，或候选 stage 能看到 hidden seed；
- 候选能读仓库真实隐藏输入；
- crash 后只能删除数据库重新开始；
- artifact 哈希不一致；
- 同卡存在并发评测；
- 任何脚本自动替换 torch、torch_npu 或 triton。
