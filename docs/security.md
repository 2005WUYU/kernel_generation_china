# 安全模型与凭据边界

候选源码和模型输出均是不可信输入。本系统提供针对 Kernel benchmark 的纵深防御，但不是能运行任意敌意 Python 的通用强隔离沙箱。

## 信任边界

| 组件 | 可持有模型凭据 | 可读取隐藏 seed | 可访问 NPU | 可访问网络 |
| --- | --- | --- | --- | --- |
| Controller | 是 | 是，仅最终评测编排 | 默认否 | 仅 AIPing/必要依赖 |
| Claude CLI 子进程 | 临时获得调用 token | 否 | 否 | AIPing |
| Worker 长期进程 | 否 | 是，仅用于派生隐藏 case | 是 | 否 |
| 候选 stage 子进程 | 否 | 否 | 是，单卡独占 | 否 |
| 导出器 | 否 | 否 | 否 | 否 |

若当前部署让 controller 直接执行 NPU stage，而非通过持久队列交给 Worker，则信任边界被合并。正式部署应使用独立服务账号/进程并记录这一偏差，不能仍声称 Worker 与密钥完全隔离。

生产 CLI 默认使用 durable SQLite queue 和独立 `akg worker run`。`--with-local-worker` 保留 durable job 协议但在 controller 进程内启动 Worker 线程；`--direct` 直接使用设备 backend。两者都只用于受控调试，不能作为生产凭据隔离或跨进程恢复的验收证据。

## 模型调用

Claude CLI gateway 使用：

- 一次性 `--print`；
- JSON output 与 JSON Schema；
- 空 tools；
- 禁用 session persistence；
- 临时 HOME、TMPDIR、XDG 和 Claude config；
- 隔离 cwd；
- stdin 输入 Prompt，不在 argv 暴露 Prompt；
- stderr 尾部有凭据脱敏；
- timeout 后终止整个进程组。

token 只从配置引用的环境变量解析，配置清单只保存变量名。不要增加允许 Claude CLI 读文件、运行 shell 或保留会话的 extra args。

## Worker 和候选

Worker service 不加载 controller env，启动脚本还会拒绝常见模型变量。可信 Worker 从独立 root-only `hidden.env` 读取 seed，根据队列中的私有 case-set 描述在内存中派生真实隐藏输入；队列不携带隐藏 shape。stage runner 只保留 Ascend、CANN、NPU、Triton 和最小运行路径变量，并主动排除包含 API key、auth、credential、password、private key、proxy、secret 或 token 的名字，因此候选子进程不继承 seed。

候选静态策略至少检查：

- import 根白名单；
- 固定 `custom_op`；
- 至少一个 `@triton.jit` 和 kernel launch；
- 高层 torch 计算调用；
- 文件、网络、子进程、ctypes、动态导入、eval/exec；
- CPU/Numpy fallback；
- 源码大小。

静态检查无法证明安全，因此运行时仍要求：

- 每阶段新进程、私有 cwd/HOME/tmp/cache；
- argv 启动和 `shell=False`；
- CPU、内存、文件、进程、打开文件、输出和 wall clock 限制；
- 终止进程组并回收；
- symlink/path traversal 检查；
- 独占设备锁；
- 输出同步、正确性与输入不变检查；
- profiler 证明候选 Triton kernel 实际承担主要设备时间。

## Hidden 数据

仓库只保存 `hidden_template.json`，不保存真实 hidden seed 或具体生成结果。真实 seed：

- 只存在于可信 controller 与可信 Worker 的环境；
- 存于独立 root-owned、权限 `0600` 的 `hidden.env`，不与模型凭据同文件；
- 不进入 Prompt、模型响应、队列实际 hidden cases、候选 cwd/stage 环境或导出；
- 同一实验冻结不变；
- 日志只能报告通过/失败和经过脱敏的聚合信息。

最终先按公开数据选择一个候选，再运行一次隐藏评测。隐藏失败不得依次测试历史候选，否则隐藏集会成为选择集。

## systemd 防护

Controller 与 Worker 使用不同 UID，只通过共享组访问 SQLite/artifact。三个 root-owned 环境文件由 systemd manager 读取：模型文件仅挂到 controller，Worker 文件不含模型信息，hidden 文件挂到两个可信父进程。Worker 模板还包含共享安全 umask、只读系统、私有 tmp、禁止提权、进程视图保护、内核保护和 IP 网络阻断。

不要启用 `MemoryDenyWriteExecute`，因为 Triton JIT/运行时可能需要可执行映射；不要启用 `PrivateDevices`，因为它会隐藏 NPU 设备。若发行版 systemd 不支持 `ProtectProc`/`ProcSubset`，应由安全团队逐项评审兼容偏差。

controller 需要 AIPing 网络，因此不使用全局 IP deny。生产可通过主机防火墙或 egress proxy 只允许 AIPing 目标。

## 凭据轮换

1. 停止 controller，不停止正在完成的 Worker stage。
2. 更新 `/etc/ascend-kernel-lab/controller.env`，保持 `0600`。
3. `systemctl daemon-reload` 通常不需要，除非 unit 变化。
4. 启动/恢复 controller。
5. 检查新日志没有 token。

hidden seed 不属于普通 token 轮换：一个实验中途更换会破坏可比性，应新建实验 ID。

## 日志与事件响应

若怀疑泄密：

1. 停止 controller，撤销 AIPing token。
2. 不删除现场；限制 artifact 和 journal 的访问权限。
3. 搜索泄漏范围时只输出文件名和匹配类型，不把 secret 再打印到终端。
4. 检查 Prompt、raw response、stderr、systemd journal、导出文件和 CI 日志。
5. 更换 token；如 hidden seed 泄漏，作废相关实验和训练导出。
6. 修复脱敏或环境边界并增加测试。

禁止把真实日志直接粘贴到公开 issue 或模型 Prompt。
