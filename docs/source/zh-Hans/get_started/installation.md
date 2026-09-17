# 安装

环境要求:

- 2.2 <= PyTorch <= 2.5.1
- Python >= 3.10
- CUDA >= 11.0
- [NVIDIA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) >= 7.0 (V100/RTX20 and higher)
- Linux OS

如果你遇到安装问题，可以向本项目 [反馈](https://github.com/hpcaitech/ColossalAI/issues/new/choose)。

## 从PyPI上安装

标准 wheel 包含 Python 代码和扩展源码，不包含提前编译的 CUDA 扩展。可以使用以下命令安装 Colossal-AI：

```shell
pip install colossalai
```

**注：现在只支持Linux。**

如果要在安装时编译 PyTorch 扩展，请先安装受支持的 CUDA 版 PyTorch、与其匹配的 CUDA Toolkit、C++ 编译器，以及 Python 构建工具（`setuptools`、`wheel`、`ninja`、`packaging`）。使用构建时可以访问目标 GPU 的 Linux 环境，然后明确选择源码包：

```shell
BUILD_EXT=1 pip install --no-binary=colossalai --no-build-isolation colossalai
```

`BUILD_EXT=1` 不会重新编译下载好的 wheel。`--no-binary=colossalai` 用于选择源码包，`--no-build-isolation` 让构建过程使用已经安装的 PyTorch。如果已经安装了 ColossalAI，请先卸载，再切换为源码构建。

不提前编译时，需要的扩展会在首次使用时编译。这同样需要上述开发工具；安装包成功不代表已经验证了 GPU 执行。

## 从源安装

> 此文档将与版本库的主分支保持一致。如果您遇到任何问题，欢迎给我们提 issue。

```shell
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# install dependency
pip install -r requirements/requirements.txt

# install colossalai
BUILD_EXT=1 pip install --no-build-isolation .
```

如果希望把扩展编译推迟到首次使用，可以不设置 `BUILD_EXT`。这不会禁用融合算子，也不会消除其编译环境要求：

```shell
pip install .
```

<!-- doc-test-command: echo "installation.md does not need test" -->
