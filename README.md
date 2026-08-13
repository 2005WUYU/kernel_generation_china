# Ascend Kernel Lab

Ascend Kernel Lab 是面向华为 Ascend 910C 与 Triton-Ascend 的完整 Kernel 生成、评测和
五轮优化流水线。模型只返回完整候选源码；Prompt、状态推进、编译、正确性、计时、
`msprof`、历史最佳选择、隐藏评测和训练数据导出全部由本地程序控制。冷启动 SFT 采集配置
使用 DeepSeek V4 Pro；轨迹不设硬加速目标，性能只与 PyTorch eager NPU 比较。

项目适合下面的交付方式：

```text
本机开发与 CPU 测试
        │
        ├── git commit / push
        │
        ▼
910C 主机干净 git clone
        │
        ├── 复用板端已有 torch / torch_npu / Triton-Ascend / CANN
        ├── AIPing + Claude CLI 调用 DeepSeek V4 Pro
        └── 八张 NPU 并行完成真实编译、校验、计时与 quick profiling
```

> 本机没有 910C 时，CPU 测试只能证明控制面、状态机、存储、解析器和假后端流程正确，不能替代板端验收，也不会伪造 NPU 性能数据。

本机发布门禁只接受 `akg experiment run --fake` 和非 NPU 测试。真实 AIPing/Claude CLI、CANN、`msprof`、Triton-Ascend 与 910C 结果必须在远端干净 clone 上完成 G2–G8，不能从开发机缓存、mock 或静态导入推断。

## 设计边界

- 控制端持有模型凭据，构建 Prompt、调用模型、推进每个任务五轮并选择历史最佳。十个任务可同时请求模型；同一任务的五轮保持顺序，后续轮只携带候选代码、关键指标、失败原因和下一轮建议。
- Worker 不加载模型凭据。八个 Worker 各独占一张 NPU，并在全新、受限的子进程中执行每个评测阶段；NPU 阶段并行上限为八。
- 候选必须通过 AST 策略、真实 JIT 编译和公开正确性，才会进入 benchmark；profiler 使用独立进程，避免污染计时。
- 五轮只能看到公开用例。隐藏用例从部署端私有 seed 生成，只在最终候选上运行，失败后不使用隐藏集挑选其他轮次。
- SQLite 保存权威状态和事件；artifact 使用临时文件、`fsync` 与原子 rename 提交。进程中断后从最后一个已提交状态恢复。
- 每张卡同时只允许一个候选运行。首版不依赖 Redis 或 PostgreSQL。

系统内置十类任务：Vector Add、Bias+GELU、SwiGLU、Transpose、Row Softmax、RMSNorm、LayerNorm、RoPE、GEMM、GEMM+Bias+GELU。任务规范、公开用例和隐藏用例模板位于 `task_specs/`。

## 仓库结构

```text
configs/                       实验、profiler 与源码安全策略
task_specs/                    十个版本化 Kernel 任务
src/ascend_kernel_lab/
  backend/ascend/              真实 Triton-Ascend 执行后端
  orchestration/               五轮控制器、baseline、结构化反馈
  worker/                      独立阶段子进程、资源限制、设备锁与健康检查
  probe/                       环境、能力、计时与 profiler 探测
  profiling/                   msprof 调用和跨版本字段归一化
  storage/                     SQLite 状态、租约队列、事件与原子 artifact
  llm/                         Claude CLI/AIPing 与 OpenAI-compatible 适配器
  export/                      SFT、RL 和报告导出
tests/                         CPU 单元测试、集成测试和 profiler fixtures
deploy/                        systemd 与环境文件模板
scripts/                       本机检查、910C 安装、服务入口和板端验收
docs/                          开发、部署、运维、安全与验收说明
runs/                          本地运行状态与产物；默认不提交 Git
```

## 本机开发

要求 Python 3.10 或更高版本。普通开发环境不需要安装 `torch_npu` 或 Triton-Ascend；真实 NPU 路径只在 910C 上运行。

```bash
git clone <your-repository-url> ascend-kernel-lab
cd ascend-kernel-lab

python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'

./scripts/check-local.sh
```

还可以运行不访问模型和 NPU 的十任务五轮控制面演练。`--fake` 自动把默认实验 ID 加上 `_fake`，避免覆盖生产实验：

```bash
akg experiment run --fake
akg verify-run --experiment-root runs/exp_910c_kimi_k3_v1_fake -c configs/experiment_910c_kimi_k3.yaml
```

`check-local.sh` 会执行 Ruff、mypy、非 NPU pytest、CLI 帮助检查和所有 shell 脚本的语法检查。也可分别运行：

```bash
make lint
make typecheck
make test
```

提交前确认没有凭据、隐藏 seed、数据库或运行产物进入版本控制：

```bash
git status --short
git diff --check
git add README.md configs deploy docs scripts src task_specs tests
git commit -m 'build complete Ascend kernel generation pipeline'
git push origin HEAD
```

当前工作区的实现不会替用户自动 `git init`、commit 或 push。发布人必须先检查完整 diff，确保 Git 仓库与 remote 指向正确目标，再执行上述版本控制操作。远端验收必须记录实际 clone 的 commit；dirty/unversioned checkout 不能通过 G1。

更完整的开发约定见 [本机开发指南](docs/development.md)。

## 910C 干净安装

不要让项目安装脚本升级、降级或覆盖板端加速栈。先由运维按机器版本安装并验证 CANN、驱动、固件、PyTorch、`torch_npu` 和 Triton-Ascend，再执行：

```bash
git clone <your-repository-url> /opt/ascend-kernel-lab
cd /opt/ascend-kernel-lab

export AKG_CANN_ENV_FILE=/usr/local/Ascend/ascend-toolkit/set_env.sh
export AKG_VENV_DIR=/opt/ascend-kernel-lab-venv
./scripts/install-910c.sh
```

安装脚本使用 `python3 -m venv --system-site-packages`，先验证系统环境中的 `torch`、`torch_npu` 和 `triton`，只允许在 venv 内补齐 PyYAML、wheel、setuptools 这组纯 Python 构建/运行依赖，再通过 `pip --no-deps --no-build-isolation` 安装本项目。任何加速栈模块缺失或版本身份变化都会直接失败，不会尝试从 PyPI 修复。

也可以从开发机发起一次全新远程 clone。目标目录必须不存在，远端账号需自行具备 Git 认证和目标父目录写权限；脚本不复制 SSH key 或模型凭据：

```bash
./scripts/remote-install-910c.sh \
  npu-user@910c-host \
  git@github.com:your-org/ascend-kernel-lab.git \
  /opt/ascend-kernel-lab \
  /usr/local/Ascend/ascend-toolkit/set_env.sh \
  main
```

venv 默认放在 checkout 外的 `/opt/ascend-kernel-lab-venv`，避免把未跟踪环境目录带进 release clean-check。

激活环境并验证 CLI：

```bash
. /opt/ascend-kernel-lab-venv/bin/activate
akg --help
akg doctor -c configs/experiment_910c_deepseek_v4_pro.yaml --allow-not-ready
```

此时尚未加载 controller 的 AIPing 凭据和隐藏 seed，预检查会把相关项保留为失败证据；完成 controller 环境配置后必须再运行一次不带 `--allow-not-ready` 的 `doctor`。

详细的用户、权限、CANN 环境和 systemd 配置见 [910C 部署指南](docs/deployment-910c.md)。
当前 openEuler/aarch64 主机也可使用彼此隔离的 Controller/Worker 双容器，详见
[910C 双容器部署指南](docs/container-deployment-910c.md)。该方案不依赖 Docker Compose，
不会在 Worker 镜像中安装或替换板端加速栈。

## AIPing + Claude CLI + DeepSeek V4 Pro

冷启动 SFT 生产配置是 `configs/experiment_910c_deepseek_v4_pro.yaml`，模型 ID 精确为
`deepseek-v4-pro`。配置仍走 AIPing 提供的 Anthropic-compatible 地址和现有 Claude CLI
通道；不要把 endpoint 或 token 写进 YAML。

DeepSeek 与旧的 Kimi 配置都使用 `model.provider: claude_cli`。Claude CLI 只是一次性、无工具的
结构化生成通道；它不控制终端、不读取仓库、不执行候选。适配器强制：

- `--print` 单次输出；
- `--output-format json` 与 JSON Schema；
- `--tools ""` 禁用工具；
- `--no-session-persistence` 禁用会话持久化；
- 临时 HOME、临时工作目录和独立配置目录；
- API/传输失败采用有界退避重试，格式修复次数独立受限。

先按 AIPing 租户提供的信息准备控制端专用环境文件。不要把真实值写入仓库或命令历史：

```dotenv
ANTHROPIC_BASE_URL=https://your-aiping-anthropic-compatible-endpoint.example
ANTHROPIC_AUTH_TOKEN=replace-with-a-real-token
ANTHROPIC_MODEL=deepseek-v4-pro
```

另建仅含隐藏 seed 的共享可信环境文件：

```dotenv
AKG_HIDDEN_SEED=REPLACE_WITH_HIGH_ENTROPY_BASE10_INTEGER
```

`AKG_HIDDEN_SEED` 必须是高熵十进制整数；它不是普通标签。可在受控终端用 `python3 -c 'import secrets; print(secrets.randbits(256))'` 生成，然后只写入 root 拥有、权限 `0600` 的独立 `hidden.env`。可信 controller 与可信 Worker 都需要它来一致地派生隐藏 case；Claude CLI 与候选 stage 不会继承它。Worker 仍绝不加载 AIPing/模型 key。

交互式试运行可在当前 shell 中设置这些变量。systemd 部署应分别复制 controller、Worker 和 hidden 三个模板；模型 key 只进入 `controller.env`，hidden seed 只进入 `hidden.env`，两者都设为 root-only `0600`。

检查 Claude CLI 和完整配置：

```bash
claude --version
akg doctor -c configs/experiment_910c_deepseek_v4_pro.yaml
```

若 AIPing 暴露的是 OpenAI-compatible 接口，可把配置中的 provider 改为 `openai_compatible`，并仅通过配置所引用的环境变量提供 key。两种 provider 都不能把 key 写进 YAML、Prompt、日志或 artifact。

## 标准运行流水线

下面命令均从仓库根目录执行。`-c` 是 `--config` 的短参数。

### 1. 配置和数据库检查

```bash
akg doctor -c configs/experiment_910c_deepseek_v4_pro.yaml
akg db upgrade -c configs/experiment_910c_deepseek_v4_pro.yaml
```

`doctor` 检查配置、目录、依赖、Claude CLI/AIPing 引用和 Ascend 工具；`db upgrade` 幂等创建或升级 SQLite schema。数据库必须位于本机文件系统，不要放在 NFS 上。

### 2. 环境、Triton、计时和 profiler 探测

```bash
akg probe all \
  -c configs/experiment_910c_deepseek_v4_pro.yaml \
  -o runs/probe
```

成功后至少检查：

```text
runs/probe/env_manifest.json
runs/probe/capability_matrix.json
runs/probe/profiler_capabilities.json
```

实际 SoC 名、CANN 版本和 profiler 字段由机器探测，不在代码中假定。不可用指标必须是 `null` 或带明确 unavailable reason。

### 3. PyTorch eager baseline

先用 K01 验证计时，再测全部任务：

```bash
akg baseline run -c configs/experiment_910c_deepseek_v4_pro.yaml --task k01_vector_add
akg baseline run -c configs/experiment_910c_deepseek_v4_pro.yaml
```

本轮冷启动 SFT 只测 PyTorch eager NPU 基线，不运行 `torch.compile` 或官方/手写实现对照，
也不要求候选达到指定加速比。benchmark 只报告候选相对 PyTorch eager 的耗时与 speedup。
环境、任务规范、benchmark 配置或 harness commit 变化后，旧 baseline 会因 identity hash
不匹配而失效。

### 4. 启动八个 Worker

容器部署在这台双芯片 910C 八卡主机上使用 `AKG_DEVICE_IDS=0,2,4,6,8,10,12,14`
与 `start-workers`；每个
Worker 只看到一张物理卡，SQLite durable queue 把十个并发任务自然限流到最多八个 NPU
阶段同时运行：

```bash
export AKG_CONFIG_PATH=configs/experiment_910c_deepseek_v4_pro.yaml
export AKG_DEVICE_IDS=0,2,4,6,8,10,12,14
./scripts/run-containers-910c.sh start-workers
./scripts/run-containers-910c.sh status
```

单卡长期运行：

```bash
AKG_CANN_ENV_FILE=/usr/local/Ascend/ascend-toolkit/set_env.sh \
AKG_VENV_DIR=/opt/ascend-kernel-lab-venv \
AKG_CONFIG_PATH=/opt/ascend-kernel-lab/configs/experiment_910c_deepseek_v4_pro.yaml \
./scripts/run-worker.sh
```

只领取一个 job 后退出，用于调试：

```bash
akg worker run -c configs/experiment_910c_deepseek_v4_pro.yaml --once
```

Worker 入口脚本会在启动前拒绝常见模型密钥变量。每个阶段还有独立子进程、清理后的环境、超时、输出上限、进程组终止、私有临时目录、设备独占锁和健康检查。

### 5. 启动或恢复五轮实验

控制端 shell 必须有 AIPing 凭据和 `AKG_HIDDEN_SEED`：

```bash
akg experiment run -c configs/experiment_910c_deepseek_v4_pro.yaml
```

生产默认使用持久 SQLite 队列：controller 以 `task_concurrency: 10` 同时推进十个任务，
八个独立 Worker 领取、执行和提交 NPU 阶段，五轮在各自任务内部顺序迭代。必须先启动上一节的
八个 Worker；不要给生产 controller 加 `--direct` 或 `--with-local-worker`。

只跑指定任务：

```bash
akg experiment run \
  -c configs/experiment_910c_deepseek_v4_pro.yaml \
  --task k01_vector_add
```

查看状态和恢复：

```bash
akg experiment status -c configs/experiment_910c_deepseek_v4_pro.yaml
akg experiment resume -c configs/experiment_910c_deepseek_v4_pro.yaml
```

若要在受控板端调试中临时使用单命令，可加 `--with-local-worker`：它仍经过 durable queue，但 Worker 线程和 controller 位于同一进程，进程级凭据边界不成立。`--direct` 会绕过队列拓扑，仅用于后端调试和定位问题，不用于生产、恢复验收或安全声称。

`--experiment-id <safe-id>` 可为 run/resume 覆盖配置中的持久实验 ID；恢复时必须再次传入完全相同的值。没有显式 override 时使用配置 ID。

```bash
akg experiment status \
  -c configs/experiment_910c_deepseek_v4_pro.yaml \
  --experiment-id <same-safe-id>
```

传输失败不会消耗优化轮；结构化输出经有限修复仍失败时，才记录为该轮模型失败。相同源码会复用已提交评测。第五轮不覆盖历史最佳。

### 6. 独立评测候选

```bash
akg evaluate \
  -c configs/experiment_910c_deepseek_v4_pro.yaml \
  --task k05_row_softmax \
  --candidate /absolute/path/to/candidate.py
```

不要评测来源不可信且超出当前任务策略的 Python 文件；即使有静态和运行时限制，910C Worker 仍应部署在专用主机或受控安全域中。

### 7. 验证并导出

```bash
akg verify-run --experiment-root runs/exp_910c_deepseek_v4_pro_cold_sft_v1

akg export sft \
  --all-samples \
  --experiment-root runs/exp_910c_deepseek_v4_pro_cold_sft_v1 \
  -o runs/exports/exp_910c_deepseek_v4_pro_cold_sft_v1.sft.jsonl

akg export rl \
  --experiment-root runs/exp_910c_deepseek_v4_pro_cold_sft_v1 \
  -o runs/exports/exp_910c_deepseek_v4_pro_cold_sft_v1.rl.jsonl

akg export report \
  --experiment-root runs/exp_910c_deepseek_v4_pro_cold_sft_v1 \
  -o runs/exports/exp_910c_deepseek_v4_pro_cold_sft_v1.report.json
```

导出器只读取已提交 artifact，并拒绝疑似凭据、hidden seed 或隐藏 case 细节。冷启动 SFT
使用 `akg export sft --all-samples` 导出五轮轨迹，不以达到某个 speedup 作为采集前提；
RL 轨迹明确记录最终采用的轮次。

## Artifact 结构

一次实验的典型布局如下；实际阶段目录还会保存每次尝试的 stdout、stderr、IR、Triton cache 和 profiler 原始文件：

```text
runs/
├── metadata.db
└── exp_910c_deepseek_v4_pro_cold_sft_v1/
    ├── experiment.json
    ├── environment_snapshot.json
    ├── baseline_snapshot.json
    ├── events.jsonl
    └── tasks/
        └── k05_row_softmax/
            ├── task_snapshot.json
            ├── best_candidate.py
            ├── final_result.json
            ├── round_01/
            │   ├── prompt.json
            │   ├── raw_response.json
            │   ├── model_response.json
            │   ├── candidate.py
            │   ├── evaluation_result.json
            │   ├── feedback.json
            │   ├── compile/
            │   ├── correctness/
            │   ├── benchmark/
            │   └── profile/
            ├── round_02/
            ├── round_03/
            ├── round_04/
            ├── round_05/
            └── final_evaluation/
                └── final_evaluation.json
```

SQLite 事件是状态恢复的权威来源；`events.jsonl` 是可重建投影。不要在实验运行时手工编辑数据库或已提交 JSON，不要只复制 `best_candidate.py` 而丢弃其环境和任务哈希。

## 安全边界

默认防线包括：

- 候选 import 白名单和 AST 调用链检查，禁止高层 PyTorch 代理、CPU/Numpy 回退、动态导入、文件、网络、subprocess 和 ctypes。
- 候选源码大小限制、固定 `custom_op` 入口和至少一个 Triton JIT kernel。
- 每阶段新进程，argv 调用且不使用 `shell=True`；清除 key、token、proxy 等敏感环境变量。
- 不同随机输入、不同地址偏移、输入不变检查、输出同步、数值校验和隐藏测试。
- profiler 校验候选 kernel 的真实执行和设备时间覆盖率，防止主要计算回退到现有算子。
- systemd Worker 模板不加载 controller 环境文件，并默认禁止 IP 网络。
- Prompt、模型响应、日志和训练导出均不得包含真实隐藏 seed 或凭据。

这些机制是纵深防御，不等同于通用恶意 Python 沙箱。生产环境应使用专用低权限账号、独立 910C 节点或容器/虚机边界，并限制谁能提交候选和修改任务规范。完整威胁模型见 [安全说明](docs/security.md)。

## 常见故障

| 现象 | 处理 |
| --- | --- |
| `doctor` 找不到 `torch_npu`/Triton | 退出项目 venv检查系统 Python；确认 venv 使用 `--system-site-packages`；不要用 pip 自动替换厂商栈 |
| `npu-smi` 正常但 Python NPU 不可用 | 重新加载正确版本的 CANN `set_env.sh`，核对 torch、torch_npu、CANN 和驱动兼容矩阵 |
| Claude CLI 401/404 或模型名错误 | 仅在 controller 环境检查 AIPing URL、token 和模型映射；不要把 key 复制到 Worker |
| 模型 JSON 截断/不合法 | 查看该轮 `raw_response.json` 和 `model_failure.json`；修复网关限制后执行 `experiment resume` |
| 编译或正确性失败 | 查看该轮 `evaluation_result.json`、阶段 stderr 和反馈；系统会跳过性能测试并进入下一轮 |
| benchmark CV 超阈值 | 系统自动重测一次；仍不稳定时检查同卡其他进程、功耗/频率、温度和 host 负载 |
| `msprof` 不可用 | 先检查 `profiler_capabilities.json` 与 `msprof op --help`；结果不会被猜测填充 |
| Worker 异常退出 | 候选子进程错误不会带崩长期 Worker；检查健康检查和 systemd 日志，再重启 Worker、恢复实验 |
| 设备锁超时 | 确认同一张卡只有一个 Worker，避免手工进程与服务同时运行 |
| 最终阶段提示缺少 hidden seed | 给 controller 与 Worker 加载同一份独立 `hidden.env`，重启 Worker 后恢复；不要更改已完成公开轮次 |
| 主机重启或控制器中断 | 先运行 `experiment status` 与 `verify-run`，再运行 `experiment resume` |
| SQLite locked | 确认只有一个 controller 写入，数据库位于本地磁盘且目录权限正确 |

更系统的恢复和诊断步骤见 [运维手册](docs/operations.md)。

## 验收

开发机只能完成 fake/CPU 门禁。下列 acceptance 汇总和 G2–G8 证据必须在实际 910C clone 上生成；尤其不能把本机 `doctor --allow-not-ready` 或 fake `verify-run` 当作真实 G7。

完成运行后的自动证据汇总入口：

```bash
export AKG_CANN_ENV_FILE=/usr/local/Ascend/ascend-toolkit/set_env.sh
export AKG_VENV_DIR=/opt/ascend-kernel-lab-venv
set -a
. /etc/ascend-kernel-lab/controller.env
. /etc/ascend-kernel-lab/hidden.env
set +a
./scripts/acceptance-910c.sh
```

底层 CLI 也可直接运行：

```bash
akg acceptance \
  -c configs/experiment_910c_deepseek_v4_pro.yaml \
  --evidence-root runs/acceptance_evidence \
  -o runs/acceptance_report.json
```

`akg acceptance` 聚合 G0–G8 显式证据、`doctor` 与已完成实验的生命周期/hash/leakage 检查；缺失的硬件或恢复证据保持 `pending`，绝不会因 `verify-run` 通过而冒充完整硬件验收。它不会替你执行 NPU 100 次、进程 kill、设备故障、重启或升级演练。流水线可靠性仍要求按 [910C 验收清单](docs/acceptance-910c.md) 完成并记录这些步骤。验收目标不是要求某次模型生成一定超过官方实现。

## systemd

模板位于：

- `deploy/systemd/ascend-kernel-worker.service`
- `deploy/systemd/ascend-kernel-controller.service`
- `deploy/env/worker.env.example`
- `deploy/env/controller.env.example`
- `deploy/env/hidden.env.example`

复制前必须把 `/opt/ascend-kernel-lab`、两个服务账号、共享运行组、CANN 路径和设备号改为实际值。Worker 与 controller 使用不同环境文件；模型 key 只在 controller 文件中。两者通过 root-only `hidden.env` 接收相同隐藏 seed。

## 文档索引

- [本机开发指南](docs/development.md)
- [910C 部署指南](docs/deployment-910c.md)
- [910C 验收清单](docs/acceptance-910c.md)
- [运维与故障恢复](docs/operations.md)
- [安全模型与凭据边界](docs/security.md)
