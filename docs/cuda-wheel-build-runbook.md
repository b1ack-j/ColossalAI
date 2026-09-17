# B4：Linux 无 GPU 构建操作说明

状态：脚本及本地模拟测试已准备；下面的 Linux 命令尚未实际执行。
本轮固定 Linux x86_64、Python 3.10、Torch 2.5.1 cu124、Toolkit 12.4。
执行者提供 Linux 主机和 Docker；不需要 GPU。Mac ARM 不作为本组合验收机器。

## 1. 准备源码与输出目录

使用本地交接包中的 `integration-source.tar.gz`，它包含六份改动及独立的 `.git`。
解压后必须能在目录中执行 `git rev-parse HEAD` 和 `git status --short`。
这些改动尚未提交，看到已暂存改动是预期行为。不要只挂载普通 linked worktree：
其 `.git` 文件可能指向容器外的宿主路径。

在 Linux 上以 Bash 执行，替换源码目录：

```bash
export LOCAL_SOURCE=/absolute/path/to/integration-source
export ARTIFACTS="$(mktemp -d /tmp/colossal-b4.XXXXXX)"
test "$(uname -m)" = x86_64
test -d "$LOCAL_SOURCE/.git"
docker pull nvidia/cuda:12.4.1-devel-ubuntu22.04
docker image inspect nvidia/cuda:12.4.1-devel-ubuntu22.04 > "$ARTIFACTS/container-image.json"
```

容器 tag 和构建工具版本与 CI 一致；保存 inspect 结果记录本次实际镜像 digest。
再次构建时重新创建输出根目录，不混用旧产物。

## 2. 执行两个独立构建

先定义函数，然后分别运行默认、显式目标。没有 `--gpus` 参数；两项环境变量也禁用 GPU 可见性。

```bash
run_build() {
  docker run --rm \
    -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void \
    -e MAX_JOBS=2 -e ARCH_MODE="$1" \
    -v "$LOCAL_SOURCE:/workspace:ro" -v "$ARTIFACTS:/artifacts" \
    -w /workspace nvidia/cuda:12.4.1-devel-ubuntu22.04 bash -lc '
      set -euo pipefail
      apt-get update
      apt-get install -y --no-install-recommends build-essential python3 python3-dev python3-venv git
      git config --global --add safe.directory /workspace
      python3 -m venv /opt/wheel-venv
      /opt/wheel-venv/bin/python -m pip install pip==25.0.1 setuptools==75.8.0 wheel==0.45.1 build==1.2.2.post1 packaging==24.2 ninja==1.11.1.3
      /opt/wheel-venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
      args=()
      if [[ "$ARCH_MODE" == explicit ]]; then args+=(--arch-list "8.0;9.0"); fi
      /opt/wheel-venv/bin/python scripts/build_cuda_wheel.py --output "/artifacts/$ARCH_MODE" "${args[@]}"
    '
}
run_build default
run_build explicit
```

默认模式会清除继承的 `TORCH_CUDA_ARCH_LIST`，真正检查 B1 默认逻辑。
显式模式检查用户指定的目标。两次不能互相代替；一次失败也应单独保留失败证据。

## 3. 判断结果和交付

每个输出子目录必须同时满足：命令退出码为 0、`result.json` 为 passed、
`build-report.json` 为 passed。`wheel-inspection.json` 中七个扩展齐全，
六个 GPU 扩展分别含 sm80、sm90；CPU Adam 不要求 GPU 代码。

交付整个输出目录，不只拷贝 wheel：

- wheel、`SHA256SUMS`、`build-report.json`：给 B5 核对产物与环境。
- `environment.json`、`build.log`、`wheel-inspection.json`、`result.json`：定位构建问题。
- `source-snapshot.json`、`source.tar.gz`：还原包含未提交改动的实际源码。
- 根目录 `container-image.json`：记录实际容器版本。

编译失败先看 `result.json` 和 `build.log`；缺目标看具体扩展的 cuobjdump 输出。
依赖安装失败可能尚未产生 result.json，此时保存 Docker 命令的完整终端输出。
不要删除失败目录后把另一次成功结果放进去。

本步骤不证明 GPU 数值正确、不证明所有 CPU/驱动可用，也不证明 CUDA 12.8/13 可构建。
先用默认构建产物进入 [B5 操作说明](cuda-wheel-consumer-runbook.md)；显式构建作为另一项构建验收证据。
公开发布前还需检查 CPU Adam 的 `-march=native`、动态库依赖及目标索引接受的平台标签。
分析后若产物发生改变，重新检查和 GPU 验收最终 wheel，不能只改标签。
