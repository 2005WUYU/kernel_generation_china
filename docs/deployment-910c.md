# Ascend 910C 部署指南

本文面向一台已经由平台团队安装好驱动、固件、CANN、PyTorch NPU 和 Triton-Ascend 的 910C 主机。项目脚本不会管理或替换这些厂商组件。

开发机只运行 fake/CPU 门禁。AIPing/Claude CLI 真实调用、CANN、`msprof`、Triton-Ascend 编译和所有 G2–G8 证据必须由本文描述的远端干净 clone 产生。clone 后先核对 `git rev-parse HEAD` 和 `git status --short`，不得把开发目录通过 rsync/scp 当作 clean release。

## 1. 部署前提

用目标服务账号确认：

```bash
npu-smi info
python3 --version
python3 -c 'import torch, torch_npu, triton; print(torch.__version__); print(torch_npu.__version__); print(triton.__version__); print(torch.npu.is_available())'
command -v msprof
msprof op --help
command -v claude
claude --version
```

若任一组件失败，先依据华为和当前 Triton-Ascend 版本的兼容矩阵修复系统环境。不要通过本项目 venv 临时安装另一套 torch 来掩盖问题。

建议准备：

- 两个专用低权限用户 `ascend-kernel-controller`、`ascend-kernel-worker`，以及只用于共享 SQLite/artifact 的组 `ascend-kernel`；
- 本地磁盘上的代码和 `runs/`；
- 一张固定 NPU，例如 `npu:0`；
- controller 可访问 AIPing，Worker 不可访问网络；
- `/etc/ascend-kernel-lab` 仅 root 与服务组可读；
- 时间同步和足够的 artifact 磁盘空间。

## 2. 干净 clone 与安装

共享组必须先于 venv 和状态目录存在；只创建组，此时尚不创建服务账号：

```bash
getent group ascend-kernel >/dev/null || sudo groupadd --system ascend-kernel
```

安装脚本默认把共享 venv 设为 `root:ascend-kernel`，因此应由 root 执行：

```bash
git clone <your-repository-url> /opt/ascend-kernel-lab
cd /opt/ascend-kernel-lab
git rev-parse HEAD
test -z "$(git status --porcelain=v1)"

export AKG_CANN_ENV_FILE=/usr/local/Ascend/ascend-toolkit/set_env.sh
export AKG_VENV_DIR=/opt/ascend-kernel-lab-venv
export AKG_SHARED_GROUP=ascend-kernel
export AKG_VENV_OWNER=root
sudo --preserve-env=AKG_CANN_ENV_FILE,AKG_VENV_DIR,AKG_SHARED_GROUP,AKG_VENV_OWNER \
  ./scripts/install-910c.sh
```

若 venv 缺少 PyYAML、wheel 或足够新的 setuptools，可采用两种显式方式之一：

```bash
# 离线，推荐
export AKG_PURE_PYTHON_WHEELHOUSE=/srv/wheels
sudo --preserve-env=AKG_CANN_ENV_FILE,AKG_VENV_DIR,AKG_SHARED_GROUP,AKG_VENV_OWNER,AKG_PURE_PYTHON_WHEELHOUSE \
  ./scripts/install-910c.sh

# 允许从当前 pip index 获取限定的纯 Python 构建/运行依赖
export AKG_ALLOW_NETWORK_INSTALL=1
sudo --preserve-env=AKG_CANN_ENV_FILE,AKG_VENV_DIR,AKG_SHARED_GROUP,AKG_VENV_OWNER,AKG_ALLOW_NETWORK_INSTALL \
  ./scripts/install-910c.sh
```

脚本只会请求 PyYAML、wheel 和 setuptools，且使用 `--no-deps`；安装本项目时再使用 `--no-deps --no-build-isolation`。它会在安装前后记录并比较 torch、torch_npu 与 triton 的模块路径、版本和 NPU 可用性；任何变化都视为失败。

venv 位于 checkout 外，避免破坏 `git status` 的干净发布证明。它由管理员安装并递归
移除 group/other 写权限，同时把部署组设为 `ascend-kernel`；两个服务 UID 可以读取/执行，
但不能通过共享组篡改 site-packages。不要以任一服务账号运行安装脚本，也不要把 venv
设为 `0770`。从开发机远程执行时可以使用：

```bash
./scripts/remote-install-910c.sh \
  npu-user@910c-host \
  git@github.com:your-org/ascend-kernel-lab.git \
  /opt/ascend-kernel-lab \
  /usr/local/Ascend/ascend-toolkit/set_env.sh \
  <branch-or-tag>
```

脚本只接受全新目标路径，已存在时直接失败，不覆盖、不删除、不拉取更新。远端需已创建
`ascend-kernel` 组；SSH 用户必须是 root 或具备非交互 `sudo -n` 的部署管理员，脚本不会
请求或传送 sudo 密码。SSH 和私有仓库认证使用调用者已有配置；不要把 access token 拼入
Git URL。安装脚本要求 40 位 SHA-1 `HEAD` 且安装前后 `git status
--porcelain=v1 --untracked-files=all` 均为空；发现代码、tracked 或 untracked 文件变化都会
fail-closed。`AKG_ALLOW_NON_GIT_SOURCE_FOR_TESTING=1` 仅用于离线打包测试，生产禁止设置。

## 3. CANN 环境文件

不同安装布局可能是：

```text
/usr/local/Ascend/ascend-toolkit/set_env.sh
/usr/local/Ascend/cann/set_env.sh
/usr/local/Ascend/cann-<version>/set_env.sh
```

必须选择与当前 `torch_npu` 和 Triton-Ascend 匹配的那一份。将绝对路径写入部署环境变量 `AKG_CANN_ENV_FILE`。不要把整份 `set_env.sh` 复制进仓库。

## 4. Controller 凭据

先创建独立服务用户、共享运行组和状态目录。不同发行版的账号管理命令可能不同，下面以常见 Linux 命令为例，执行前由系统管理员核对：

```bash
sudo useradd --system --gid ascend-kernel --home-dir /nonexistent --shell /usr/sbin/nologin ascend-kernel-controller
sudo useradd --system --gid ascend-kernel --home-dir /nonexistent --shell /usr/sbin/nologin ascend-kernel-worker
sudo install -d -m 2770 -o root -g ascend-kernel /opt/ascend-kernel-lab/runs
```

共享组已在安装前创建；两项 `useradd` 只执行一次，账号已存在时不要重复运行或修改 UID/GID。
`runs/` 的 `2` 是 setgid 位：其下新建目录和文件继承 `ascend-kernel`
组。Controller 与 Worker 的启动 umask 均为 `0007`，共享控制面目录/文件分别为
`02770`/`0660`，不会向 other 开放。Worker 为每次候选执行创建的 attempt、work、cache、
IR 和临时目录仍显式使用 `0700`；不要为了排障把这些目录递归改成组可访问。

若这是从旧版本升级，旧的 `umask 077` 可能留下仅创建者可访问的 SQLite、WAL/SHM
或 artifact 父目录。先停止两个服务并做一致性备份，再由 root 一次性修复共享控制面；
下面的 `find` 会剪枝生产队列的私有 candidate attempt 子树：

```bash
sudo systemctl stop ascend-kernel-controller.service ascend-kernel-worker.service
sudo chgrp -R ascend-kernel /opt/ascend-kernel-lab/runs
sudo chmod -R o-rwx /opt/ascend-kernel-lab/runs
sudo chmod 2770 /opt/ascend-kernel-lab/runs
sudo find /opt/ascend-kernel-lab/runs \
  -path '*/worker_jobs/*/attempt_*' -prune -o \
  -type d -exec chmod 2770 '{}' +
sudo find /opt/ascend-kernel-lab/runs \
  -path '*/worker_jobs/*/attempt_*' -prune -o \
  -type f -exec chmod 0660 '{}' +
sudo find /opt/ascend-kernel-lab/runs \
  -path '*/worker_jobs/*/attempt_*' -type d -exec chmod 0700 '{}' +
```

不要在服务运行时修改 SQLite/WAL/SHM 模式，也不要对 `runs/` 无条件执行
`chmod -R 0770`。如果曾使用 controller 进程内直接 NPU backend（而不是生产 durable
queue），其私有 stage 目录不遵循上述 `worker_jobs` 路径；先归档旧运行，勿用这段命令
猜测修复。修复后以两个服务账号分别执行真实读写探针，再启动正式任务：

```bash
sudo -u ascend-kernel-controller test -w /opt/ascend-kernel-lab/runs
sudo -u ascend-kernel-worker test -w /opt/ascend-kernel-lab/runs
stat -c '%A %a %U %G %n' /opt/ascend-kernel-lab/runs \
  /opt/ascend-kernel-lab/runs/metadata.db 2>/dev/null || true
```

检查 NPU 设备节点的属组，并只把 Worker 用户加入板端实际要求的设备组；组名因镜像而异，不要照抄示例值：

```bash
ls -l /dev/davinci* /dev/davinci_manager /dev/devmm_svm 2>/dev/null
sudo usermod -a -G <actual-npu-device-group> ascend-kernel-worker
```

执行 probe/baseline 的维护账号也需要 NPU 权限，并应加入共享组 `ascend-kernel`。

从模板生成模型文件：

```bash
sudo install -d -m 0750 -o root -g ascend-kernel /etc/ascend-kernel-lab
sudo install -m 0600 -o root -g root \
  deploy/env/controller.env.example \
  /etc/ascend-kernel-lab/controller.env
sudoedit /etc/ascend-kernel-lab/controller.env
```

至少填写：

```dotenv
ANTHROPIC_BASE_URL=https://<tenant-endpoint>
ANTHROPIC_AUTH_TOKEN=<secret>
ANTHROPIC_MODEL=Kimi-K3
```

隐藏 seed 使用独立文件，同时由可信 controller 和可信 Worker 加载：

```bash
sudo install -m 0600 -o root -g root \
  deploy/env/hidden.env.example \
  /etc/ascend-kernel-lab/hidden.env
sudoedit /etc/ascend-kernel-lab/hidden.env
```

```dotenv
AKG_HIDDEN_SEED=<high-entropy-base-10-integer>
```

注意：

- AIPing 的 URL、认证方式和模型映射以租户实际配置为准。
- token 与 hidden seed 不得出现在 YAML、Git、systemd unit、Prompt、日志或 shell 历史。
- hidden seed 必须是十进制整数；在受控终端用 `python3 -c 'import secrets; print(secrets.randbits(256))'` 生成。
- 改变 hidden seed 会改变最终测试，正在运行的实验中不得轮换。
- controller service 允许访问 AIPing，但不应暴露通用出站网络。

这些文件是 root-only，正常运行由 systemd manager 读取并只传给对应服务。只读 `doctor` 若需交互验证，应在受控 root 维护 shell中加载；不要 `echo` secret，也不要在该 shell 执行候选：

```bash
set -a
. /etc/ascend-kernel-lab/controller.env
. /etc/ascend-kernel-lab/hidden.env
set +a
/opt/ascend-kernel-lab-venv/bin/akg doctor \
  -c /opt/ascend-kernel-lab/configs/experiment_910c_kimi_k3.yaml
```

## 5. Worker 环境

```bash
sudo install -m 0600 -o root -g root \
  deploy/env/worker.env.example \
  /etc/ascend-kernel-lab/worker.env
sudoedit /etc/ascend-kernel-lab/worker.env
```

Worker 文件只包含项目路径、venv、CANN 环境和非敏感 NPU 设置。严禁加入任何模型凭据：

```text
ANTHROPIC_AUTH_TOKEN
KIMI_API_KEY
任何 AIPing/模型 token
```

可信 Worker 从独立 `hidden.env` 读取 seed，根据队列里的私有 case-set 描述在内存中派生真实隐藏 case。队列不携带真实 hidden shape，Claude CLI 和候选 stage 不继承 seed。`scripts/run-worker.sh` 启动时会检查常见模型变量，只要发现就拒绝启动。

## 6. 首次初始化

```bash
cd /opt/ascend-kernel-lab
. /opt/ascend-kernel-lab-venv/bin/activate
. /usr/local/Ascend/ascend-toolkit/set_env.sh
umask 0007

akg db upgrade -c configs/experiment_910c_kimi_k3.yaml
akg probe all -c configs/experiment_910c_kimi_k3.yaml -o runs/probe
akg baseline run -c configs/experiment_910c_kimi_k3.yaml --task k01_vector_add
```

以上初始化命令由具备 NPU 权限且属于共享组 `ascend-kernel` 的维护账号执行。完整 `doctor` 按上一节由 root 维护 shell 执行；不要把 controller token 导入这个 NPU 初始化 shell。

人工检查 probe 文件中：

- 设备数量和选定设备正确；
- SoC 名来自 `npu-smi`；
- 驱动、固件、CANN、torch、torch_npu、triton 版本完整；
- `tl.dot`、归约、非对齐 shape 等每项都有 compile/run/correct/error；
- timing verified 且空操作开销、CV 合理；
- profiler 能力只报告实际存在字段。

随后按验收清单执行硬件与恢复场景。已完成正式实验后，用脚本汇总自动证据：

```bash
./scripts/acceptance-910c.sh
```

该命令聚合 G0–G8 证据、doctor 和现有实验验证；缺失证据保持 pending，不会自动执行故障注入或重启演练。

## 7. 安装 systemd

先检查模板中的路径、用户、组和配置：

```bash
sudo install -m 0644 deploy/systemd/ascend-kernel-worker.service \
  /etc/systemd/system/ascend-kernel-worker.service
sudo install -m 0644 deploy/systemd/ascend-kernel-controller.service \
  /etc/systemd/system/ascend-kernel-controller.service
sudo systemctl daemon-reload
```

先启动 Worker：

```bash
sudo systemctl enable --now ascend-kernel-worker.service
systemctl status ascend-kernel-worker.service
journalctl -u ascend-kernel-worker.service -n 100 --no-pager
```

控制器默认执行 `experiment resume`，适合初次运行和崩溃恢复：

```bash
sudo systemctl start ascend-kernel-controller.service
systemctl status ascend-kernel-controller.service
journalctl -u ascend-kernel-controller.service -f
```

模板的 Worker 使用 `IPAddressDeny=any`，controller 不使用该限制。若当前 systemd 不支持某项 sandbox 指令，先由安全团队评审后做最小兼容修改，不要直接删除所有 hardening。

unit 只对进程被信号意外终止及意外/基础设施失败（退出码 `4`）做有限自动重启；
配置错误、doctor/not-ready、正常的任务终态失败（退出码 `5`）和凭据缺失不会进入无限
重启循环。Worker 的意外失败同样允许有限重启。修复非重试原因后由运维显式
`restart`/`start`，controller 会从已提交 checkpoint 恢复。

## 8. Git 更新与回滚

不要在运行中的 checkout 执行覆盖式更新。推荐版本目录：

```text
/opt/ascend-kernel-lab-releases/<git-commit>/
/opt/ascend-kernel-lab-current -> <validated-release>
/var/lib/ascend-kernel-lab/runs/
```

本仓库默认配置把 `runs/` 放在项目根。生产采用共享状态目录时，需在版本化配置中把 `storage.database` 和 `storage.artifact_root` 改成绝对部署路径，并保持同一文件系统上的原子 rename 语义。该状态根必须由 root 预建为 `root:ascend-kernel 02770`；应用会修复由当前 UID 拥有的旧模式，但无法替另一 UID `chmod`，所以跨 UID 升级仍必须执行前述停机修复。

升级步骤：

1. 停止 controller，等待或停止 Worker 当前 job。
2. 在新目录 clone 指定 commit，创建新 venv。
3. 备份 SQLite、WAL/SHM 和 artifact；备份时所有写者必须停止。
4. 对新版本运行 `doctor` 和 `db upgrade`。
5. 在独立 acceptance 输出目录执行验收。
6. 切换 systemd WorkingDirectory/软链接，启动 Worker，再恢复 controller。
7. 观察一轮完成后再清理旧版本；清理必须由人工执行，本项目脚本不会删除旧 checkout。

回滚时，若数据库 schema 不向后兼容，不得直接让旧代码打开已升级数据库；从升级前一致性备份恢复到新的显式目录，再切换服务。

## 9. 多卡

每个 Worker 绑定一个 `worker.device`，同卡还会使用跨进程锁。多卡部署应为每张卡准备独立配置、Worker unit instance 和 worker ID；不要让两份配置指向相同 `npu:<index>`。SQLite 适合一台主机的有限并发；跨主机扩展前再引入经过认证的远程队列和集中数据库。
