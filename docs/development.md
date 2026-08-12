# 本机开发指南

本文描述没有 Ascend NPU 的开发机如何安全完成修改、测试、提交，再把同一 commit 交给 910C 主机验证。

## 开发机与 910C 的职责

开发机可以可靠验证：

- 严格 YAML 配置加载；
- 任务规范与公开/隐藏模板的结构；
- Prompt 与模型响应 Schema；
- 状态机、SQLite 事务、事件和恢复；
- AST 源码策略与已知绕过路径；
- benchmark 统计和候选排序；
- 多版本 `msprof` CSV fixture 解析；
- SFT/RL/report 导出；
- 使用 fake model/fake backend 的完整控制器流程。

只有 910C 可以验证：

- CANN、驱动、固件、PyTorch NPU 与 Triton-Ascend 的真实兼容性；
- `tl.sum`、`tl.max`、`tl.exp`、`tl.dot` 等能力；
- JIT 编译、TTIR/TTAdapter IR、设备同步和异常恢复；
- 真实数值正确性、延迟、CV、三层 baseline；
- `msprof op` 及 Vector/Cube/Scalar/MTE/GM/UB/L2 指标；
- 候选 kernel 设备时间覆盖率。

开发机测试通过不得写成“910C 已验证”。

本机完整控制面演练必须显式带 `--fake`。不连接开发机上的真实 AIPing token，也不把本机是否恰好能导入某个 torch 包当作 CANN/910C 证据。真实 Claude CLI/AIPing、CANN、`msprof` 和 G2–G8 全部在远端干净 clone 执行。

## 初始化环境

```bash
python3 --version
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

推荐使用项目声明支持的最低 Python 版本做一组 CI，再使用团队主力版本做一组 CI。不要在普通开发机为了让 NPU 测试“可导入”而安装随机来源的 `torch_npu` 或 Triton fork。

## 日常检查

```bash
./scripts/check-local.sh
```

或拆分运行：

```bash
python -m ruff check src tests
python -m mypy src
python -m pytest -m 'not npu'
python -m ascend_kernel_lab --help
```

测试规则：

- 所有 bug fix 都应增加回归测试。
- NPU 测试必须标记 `@pytest.mark.npu`，默认 CPU 测试不可意外访问设备。
- 外部命令、时间、随机数和 provider 在单元测试中使用依赖注入或 fixture。
- profiler parser 测试使用脱敏 fixture，不提交真实主机完整 timeline。
- crash recovery 测试需要覆盖重复提交、过期 lease、kill 后恢复和 artifact/DB 不一致。
- benchmark 测试使用合成样本验证 median、percentile、CV、加权几何平均与不稳定重测，不用主机 wall clock 断言性能。

## 修改任务规范

每个任务必须同时维护：

```text
task.yaml
reference.py
input_generator.py
output_validator.py
baseline.py
public_cases.jsonl
hidden_template.json
```

修改时检查：

1. 数学语义是否明确，包括维度顺序、归约轴、epsilon、广播、布局、累加 dtype 和输出 dtype。
2. public case 只包含可公开信息，不携带 hidden seed。
3. hidden template 只描述生成策略，真实 case 在部署端由私有 seed 生成。
4. reference 与 validator 不依赖候选模块。
5. case ID 稳定且唯一；性能权重为有限正数。
6. 任务 digest 改变后，旧 baseline 必须自动失效。

K09/K10 统一采用 A `[M,K]`、B `[K,N]`、bias `[N]`。任何新变体必须在 task spec 中显式表达，不能只写在 Prompt 文本里。

## 修改状态或 Schema

状态和持久格式是恢复协议，不是内部随意重命名的枚举。修改时：

- 为已有数据库和 artifact 考虑兼容或明确拒绝旧版本。
- 先提交 artifact，再提交引用它的状态事件。
- 同一阶段相同幂等键不得生成两个互相矛盾的成功结果。
- `events.jsonl` 是 SQLite 事件的投影，不反向成为权威来源。
- 所有新 JSON 都要有稳定 `schema_version`。
- 不允许用 stdout 中的结束字符串推进状态。

数据库 schema 改动必须通过 `akg db upgrade -c ...` 幂等应用，并增加从空库与现有库升级的测试。

## 修改候选执行路径

候选代码是不可信输入。以下约束不可为了“兼容某个模型输出”而绕开：

- 不使用 `shell=True`。
- 不把 controller 的环境原样传入 stage。
- 每阶段新进程和新私有工作目录。
- timeout 终止整个进程组。
- candidate、artifact、IR 和 profile 路径拒绝 symlink 越界。
- 保留资源上限、输出上限和 NPU 独占锁。
- 正确性失败不能进入 benchmark；profiler 不能与普通计时混跑。

若新增允许的 PyTorch API，必须同时解释它为何不能代理目标计算，并增加反绕过测试。

## Git 工作流

提交前：

```bash
./scripts/check-local.sh
git diff --check
git status --short
```

以下内容不得提交：

- `.env`、controller 环境文件或真实 token；
- `AKG_HIDDEN_SEED` 或从它生成的真实隐藏 case；
- `runs/` 下的数据库、日志、Prompt、响应和 profiler 原始数据；
- CANN/Triton 编译缓存；
- 从 910C 拷回的含主机名、用户路径、凭据或业务输入的日志。

推荐一次提交只包含一个可解释主题，并在提交信息或变更说明里记录：CPU 测试结果、是否需要重新跑 baseline、是否需要 910C 重新验收。

## 交付给 910C

1. 推送经过 CPU CI 的 commit。
2. 在 910C 使用全新目录 `git clone`，不要覆盖正在运行的 checkout。
3. 运行 `scripts/install-910c.sh` 创建该 checkout 专用 venv。
4. 执行 `doctor`、数据库升级、完整 probe 和 K01 baseline。
5. 执行 `scripts/acceptance-910c.sh`。
6. 验收通过后再切换 systemd 的 WorkingDirectory/软链接，并保留旧 checkout 供回滚。

部署和回滚细节见 [deployment-910c.md](deployment-910c.md)。
