# 910C Controller + 八 Worker 容器部署

这条路径用于当前 `openEuler 22.03 / aarch64 / Docker 20.10.8` 主机。它不依赖
Compose：Controller 负责 Claude CLI + AIPing，八个 Worker 分别负责一张 910C；两类容器只共享远端
干净 clone 的 `runs/`。Worker 没有网络和模型凭据，Controller 不映射 NPU。

> 目前只完成本地代码和 CPU 门禁，尚未声称 910C 验证。候选 Worker 平台镜像
> `sha256:6cc5e9d45432fe79306890bd616c3d82eea558ac18e3a1a337640f7accabc349`
> 已确认包含 Python 3.11.15、PyYAML 6.0.3、torch 2.9、torch_npu 2.9.0.post2、
> FlagTree/Triton-Ascend、`msprof`、`npu-smi`，但仍必须通过下面的真实 Triton kernel
> smoke 才能批准。它只是这台主机的示例，不是硬编码默认值。已授权研究的 Mentor DNN
> base `sha256:8b5b663f9979d54cf5af3cbfbb4fbdb4884bb64dea311a5b1d81bdeccca65d54`
> 本身缺 Triton，因此不能直接作为合格 Worker 平台镜像。若以它为第一阶段构建平台镜像，
> Triton-Ascend 的来源、版本和构建过程必须另行审批并固化，生成新的完整 image SHA，完成
> 真实 kernel smoke 后，项目第二阶段才可继承。不要把运行中手工改过的 sibling 容器
> `flagtree-dev-flagblas` 当成可复现基础镜像，即使其中可 import FlagTree。

## 1. 先验证候选 Worker 平台镜像

在 910C 主机运行（只启动并删除临时容器）：

```bash
docker run --rm \
  --runtime=ascend \
  --network none \
  -e ASCEND_VISIBLE_DEVICES=0 \
  --entrypoint /bin/bash \
  sha256:6cc5e9d45432fe79306890bd616c3d82eea558ac18e3a1a337640f7accabc349 \
  -lc 'python3 - <<'"'"'PY'"'"'
import torch
import torch_npu
import triton

print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("triton", triton.__version__)
print("npu_available", torch.npu.is_available())
print("npu_count", torch.npu.device_count())
assert torch.npu.is_available()
assert torch.npu.device_count() == 1
x = torch.ones(8, device="npu:0")
assert (x + x).cpu().tolist() == [2.0] * 8
print("NPU_TORCH_SMOKE_OK")
PY'
```

这只证明 torch NPU 基础运算。镜像构建后还必须执行项目的 `akg probe all`，让
Triton JIT、非对齐 shape、归约、计时和 profiler 形成证据；失败就停止，不启动实验。
不要把宿主机 CANN 8.5 的 `set_env.sh` source 到声明其他 CANN 版本的容器中。

## 2. 准备 Controller 基础镜像

Controller 使用不可变的 `CONTROLLER_BASE` 提供 Python 3.10+、PyYAML 和 Git。当前可把
Mentor DNN base
`sha256:8b5b663f9979d54cf5af3cbfbb4fbdb4884bb64dea311a5b1d81bdeccca65d54`
用于 Controller。Claude Code 使用 Anthropic 官方 GitHub Release 的 Linux ARM64 单文件
归档；宿主机先下载并核对固定 SHA-256，Docker 构建只做离线安装，不访问 npm 或其他包仓库。

2026-08-13 锁定的 Claude Code 是 `2.1.228`，官方 `claude-linux-arm64.tar.gz` SHA-256 为
`877d423c35e6d059752f86399352837df5bf1af2a9dbcda5753d898629a439f4`。构建会在宿主和镜像内
各核对一次摘要，再检查归档只含一个 `claude` 文件及 `claude --version`。更新版本时必须
同时更新版本、官方归档和摘要，禁止使用 `latest`、`next` 或 `beta`。

跨主机发布时，三个基础镜像必须写成 `repository@sha256:<64位摘要>`。当前本机镜像没有
RepoDigest，构建脚本也接受完整 `sha256:<64位本地image ID>`：它会 inspect 精确匹配、
打专用临时 tag 并在构建后复验。这个模式只能在同一 Docker daemon 上复现。

## 3. 构建

从远端干净 clone 根目录执行：

```bash
export WORKER_BASE=sha256:6cc5e9d45432fe79306890bd616c3d82eea558ac18e3a1a337640f7accabc349
export CONTROLLER_BASE=sha256:8b5b663f9979d54cf5af3cbfbb4fbdb4884bb64dea311a5b1d81bdeccca65d54
export CLAUDE_CODE_ARCHIVE=/var/cache/ascend-kernel-lab/claude-linux-arm64-2.1.228.tar.gz
export CLAUDE_CODE_VERSION=2.1.228
export CLAUDE_CODE_SHA256=877d423c35e6d059752f86399352837df5bf1af2a9dbcda5753d898629a439f4
./scripts/build-container-images.sh
```

构建上下文固定为 `deploy/container/`；`.git`、源码、`runs/`、环境文件和密钥不会发送给
Docker daemon。镜像分两层：平台层由平台负责人根据批准的兼容矩阵构建并形成自己的
不可变 SHA；项目 Worker 层只继承已审批且通过真实 kernel smoke 的平台层。项目构建不
安装或替换 torch、torch_npu、Triton、CANN、驱动或固件，只检查平台镜像已经提供完整
兼容栈和 PyYAML。镜像构建阶段没有 Ascend 设备及驱动挂载，因此这里只检查加速包存在，
不导入 `torch_npu`；真正的导入、单卡可见性和 NPU 运算必须在 `--runtime=ascend` 的
临时容器及后续 `probe` 中通过。

## 4. 准备运行文件和权限

必须从 GitHub 全新 clone；容器内正式命令会验证 Git top-level、完整 commit 和 clean
状态。源码以只读方式挂载，`runs/` 再作为嵌套可写挂载，因此 tracked
`runs/.gitkeep` 不会被卷遮掉。

```bash
cd /opt/ascend-kernel-lab
test -z "$(git status --porcelain=v1 --untracked-files=all)"
sudo install -d -m 0700 /etc/ascend-kernel-lab
sudo install -m 0600 deploy/container/controller.env.example /etc/ascend-kernel-lab/controller.env
sudo install -m 0600 deploy/container/hidden.env.example /etc/ascend-kernel-lab/hidden.env
sudo install -m 0600 deploy/container/worker.env.example /etc/ascend-kernel-lab/worker-container.env
sudoedit /etc/ascend-kernel-lab/controller.env
sudoedit /etc/ascend-kernel-lab/hidden.env
sudoedit /etc/ascend-kernel-lab/worker-container.env
```

Controller 文件填写 `ANTHROPIC_BASE_URL`、`ANTHROPIC_AUTH_TOKEN`、
`ANTHROPIC_MODEL=deepseek-v4-pro`；hidden 文件只放 `AKG_HIDDEN_SEED`。Worker 文件不得出现
任何 `ANTHROPIC_`、`KIMI_`、`OPENAI_` 或 `AIPING_` 变量，也不要 source 宿主 CANN。
环境文件必须是普通文件且权限为 `0600` 或 `0400`。

两个容器使用不同的专用数字 UID，并加入同一个共享数字 GID 来写 SQLite、WAL 和
artifact。把远端 `runs/` 设为共享组、setgid、other 无权限；启动前分别用两个镜像执行
一次文件创建/读取探针。不要对候选 attempt 私有目录递归放宽权限。

当前单机部署可固定使用下面三个未占用的数字身份。它们只用于本项目，不改变现有
FlagBLAS 容器：

```bash
getent group 27100 >/dev/null || groupadd --system --gid 27100 akg-shared
getent passwd 27101 >/dev/null || useradd --system --uid 27101 --gid 27100 \
  --home-dir /nonexistent --shell /sbin/nologin akg-controller
getent passwd 27102 >/dev/null || useradd --system --uid 27102 --gid 27100 \
  --home-dir /nonexistent --shell /sbin/nologin akg-worker
chown root:27100 /opt/ascend-kernel-lab/runs
chmod 2770 /opt/ascend-kernel-lab/runs
```

同一张 NPU 的 probe、baseline 和 Worker 还会共享宿主设备锁目录。该目录必须预先创建为
普通目录，属组与 `runs/` 相同且权限精确为 `2770`；启动脚本会拒绝缺失、符号链接或错误
权限。示例：

```bash
install -d -o root -g <共享数字GID> -m 2770 /var/lock/ascend-kernel-lab
```

Worker 还需加入 NPU 设备节点的数字属组，先读取实际值，不能猜组名：

```bash
stat -c '%a %u %g %n' /dev/davinci0 /dev/davinci_manager /dev/devmm_svm
```

## 5. 启动顺序

```bash
export AKG_PROJECT_ROOT=/opt/ascend-kernel-lab
export AKG_CONFIG_PATH=configs/experiment_910c_deepseek_v4_pro.yaml
export AKG_DEVICE_IDS=0,2,4,6,8,10,12,14
export AKG_CONTROLLER_UID=<controller数字UID>
export AKG_WORKER_UID=<worker数字UID>
export AKG_SHARED_GID=<runs共享数字GID>
export AKG_NPU_DEVICE_GID=<NPU设备节点数字GID>
export AKG_DEVICE_LOCK_ROOT=/var/lock/ascend-kernel-lab

./scripts/run-containers-910c.sh init
./scripts/run-containers-910c.sh probe
./scripts/run-containers-910c.sh baseline
./scripts/run-containers-910c.sh start-workers
./scripts/run-containers-910c.sh status
for DEVICE_ID in 0 1 2 3 4 5 6 7; do
  docker logs --tail 100 "ascend-kernel-worker-$DEVICE_ID"
done
```

`start-workers` 为八张物理卡各启动一个 Worker；每个容器内部仍使用 `npu:0`。Controller
把十个任务并发提交到 SQLite durable queue，NPU 阶段由八个 Worker 自然限流到最多八路并行，
同一任务的五轮保持顺序。

Worker 启动会强制验证：没有模型变量、隐藏 seed 合法、只暴露一张 NPU、
`torch.npu.is_available()` 为真、clean Git clone。它使用 `--runtime=ascend`、
`--network none`、只读根文件系统、drop capabilities、no-new-privileges、PID/内存/日志
上限，且不挂 Docker socket。

`probe` 会执行真实 Triton JIT、NPU feature smoke、计时和 profiler 能力探测；
`baseline` 会对配置中的全部任务生成环境绑定的 PyTorch eager NPU 基线；不再测
`torch.compile` 或官方/手写实现。实验轨迹用于冷启动 SFT，不设置硬加速目标，benchmark
只报告候选相对 PyTorch eager 的耗时与 speedup。任何一步非零退出都停止。
`probe` 使用可执行的临时 Triton cache、验证子进程恰好只能看到一张 NPU，并要求所有必需
feature、计时方法和 profiler 指标通过；固定输出目录必须为空，避免旧 profiler 数据混入。
维护期间若项目 Worker 正在运行会直接拒绝。
若 probe 失败，先保留整个失败证据目录再重跑，禁止直接删除：

```bash
if [ -d runs/probe ]; then
  mv runs/probe "runs/probe.failed.$(date -u +%Y%m%dT%H%M%SZ).$$"
fi
```

profile 使用 quick 模式，只做一次 warmup、一次采集，保留 task time、pipe utilization、
kernel 数和候选 kernel coverage 等短时间可取信息，不运行 full profile。

检查 `runs/probe/` 和 baseline 证据后，再按验收指南完成板端验收。八个 Worker 状态都变成
`healthy` 后运行：

```bash
./scripts/run-containers-910c.sh start-controller
./scripts/watch-experiment-910c.sh "$AKG_EXPERIMENT_ID"
```

观察脚本在前台每五秒采样，只在状态变化时打印十个任务的五轮进度、八个 Worker 健康数和
当前模型请求数；SSH 断开只会终止观察脚本，重新登录后执行同一条命令即可继续观察，容器内
实验不受影响。

Controller 有 AIPing 网络但不挂 NPU；即使宿主 Docker 把 Ascend 设为默认 runtime，启动
脚本也会为 Controller 和数据库初始化显式选择 `runc`。生产上再用主机防火墙或 egress
proxy 只允许租户端点。停止顺序是 Controller 再 Worker：

```bash
./scripts/run-containers-910c.sh stop
```

## 6. 发布阻断

以下任一情况都不得开始正式实验：基础镜像只有 tag/短 ID、架构不是 arm64、clean Git
验证失败、Worker 看到零张或多张 NPU、Worker 获得模型变量、Controller 挂载 NPU、
Worker 有网络、共享卷双角色读写失败、Triton kernel smoke/probe/baseline/profiler 任一失败。
CPU/fake 测试不能替代这些 910C 证据。
