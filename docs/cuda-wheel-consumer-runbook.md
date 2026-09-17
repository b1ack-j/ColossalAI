# B5：GPU 安装与计算验收操作说明

状态：本地工具已准备；GPU 成功路径和 TestPyPI 试发均尚未执行。
输入：B4 的完整成功输出目录、同一份整合源码中的 scripts/requirements、
一台 Linux x86_64 NVIDIA GPU 机器。宿主必须已配好驱动和容器 GPU 支持。
不指定未经确认的 GPU 型号；机器提供后，逐台记录实际型号和计算能力。

## 1. 先验收本地产物

以下命令用于 B4 首个 Python 3.10 / Torch 2.5.1 cu124 产物。
不要用 Python 3.12 构建的 wheel 套用这个环境；环境不匹配会失败。
替换三个绝对路径，在 Linux 的 Bash 中执行：

```bash
export LOCAL_SOURCE=/absolute/path/to/integration-source
export BUILD_ARTIFACTS=/absolute/path/to/b4/default
export GPU_RESULTS="$(mktemp -d /tmp/colossal-b5.XXXXXX)"
docker pull nvidia/cuda:12.4.1-base-ubuntu22.04
docker image inspect nvidia/cuda:12.4.1-base-ubuntu22.04 > "$GPU_RESULTS/container-image.json"
docker run --rm --gpus all \
  -v "$LOCAL_SOURCE:/tools:ro" -v "$BUILD_ARTIFACTS:/input:ro" \
  -v "$GPU_RESULTS:/results" -w /tmp \
  nvidia/cuda:12.4.1-base-ubuntu22.04 bash -lc '
    set -euo pipefail
    apt-get update
    apt-get install -y --no-install-recommends python3 python3-venv ca-certificates
    python3 -m venv /opt/runtime
    export PATH="/opt/runtime/bin:$PATH"
    export TORCH_EXTENSIONS_DIR=/tmp/empty-jit-cache
    mkdir "$TORCH_EXTENSIONS_DIR"
    python -m pip install pip==25.0.1
    python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    printf "torch==2.5.1\n" > /tmp/runtime-constraints.txt
    python -m pip install -c /tmp/runtime-constraints.txt -r /tools/requirements/requirements.txt
    python - <<PY
import sys
from pathlib import Path
sys.path.insert(0, "/tools/scripts")
from cuda_wheel_contract import load_build_report, verify_wheel
report = load_build_report("/input/build-report.json")
wheel = Path("/input") / report["wheel"]["filename"]
verify_wheel(report, wheel)
Path("/tmp/wheel-path.txt").write_text(str(wheel))
PY
    wheel="$(cat /tmp/wheel-path.txt)"
    python -m pip install --no-index --no-deps "$wheel"
    python -m pip freeze > /results/runtime-packages.txt
    python -m pip check > /results/pip-check.txt
    python /tools/scripts/smoke_prebuilt_wheel.py "$wheel" \
      --build-report /input/build-report.json --output /results/gpu-smoke.json --device 0
  '
```

此运行容器没有 nvcc。运行依赖按仓库声明安装并记录实际版本，
Torch 固定为 2.5.1；这是第一轮验收，尚不声称全部间接依赖已锁定或实际安装通过。
若依赖安装失败，保存终端日志并作为独立问题处理，不跳过依赖后宣布验收成功。

脚本及其相邻 `cuda_wheel_contract.py`、`check_cuda_wheel.py` 必须一起保留。
不从源码目录启动 Python，不使用 editable 安装，不提供 JIT 回退。

## 2. 如何判断成功

必须退出码为 0，且 `gpu-smoke.json` 的 status 为 passed；检查：

- build_report 中 wheel 哈希、源码标识与 B4 交付一致。
- Python 主次版本、Torch release、torch.version.cuda、OS/CPU 与构建一致。
- 记录实际 GPU、驱动；七个模块来自当前环境安装的 `.so`，文件内容匹配给定 wheel。
- `jit_attempts` 为空，FusedAdam 的多步参数更新与 LayerNorm 前向/反向数值检查通过。

其余五个模块只有导入/加载检查。此脚本成功不代表所有算子、所有 dtype、
完整训练或分布式训练通过。没有 GPU、配置不匹配、缺文件和数值错误都必须失败。
不同 GPU 机器分别执行、分别存报告，不合并成一份不注明机器的“通过”。

## 3. 后续 TestPyPI 复验：单独记录

本轮不上传、不创建试发版本。先解决 CPU 指令/动态库/平台标签及变体选择问题，
然后确定试发版本。若重新打包或更改版本，必须重新产生对应的 B4 构建清单和哈希，
不能拿旧清单验证新 wheel。

维护者提供已发布的准确版本及其验收清单后，在匹配 Python 3.10 的干净 Linux 环境下载：

```bash
export TRIAL_VERSION=维护者提供的准确试发版本
export DOWNLOAD_DIR="$(mktemp -d /tmp/colossal-testpypi.XXXXXX)"
python -m pip download --no-deps --only-binary=:all: \
  --index-url https://test.pypi.org/simple/ \
  --dest "$DOWNLOAD_DIR" "colossalai==$TRIAL_VERSION"
```

索引只用于下载这个包，依赖从正常来源安装。不要混用 extra-index 查找被验收的 wheel。
把下载的 wheel 与维护者提供的 `build-report.json` 放入同一个新目录，
使用本页第一节再创建全新运行容器。下载的文件名、哈希必须匹配，后续数值检查同样必须通过。
如果下载到的是不含 `.so` 的普通 wheel，不得算作预编译 wheel 的 B5。

结果分别命名并记录来源：本地产物验收 / TestPyPI 下载验收。只有后者真实完成，
才能勾选原表的 TestPyPI 试发验收。
