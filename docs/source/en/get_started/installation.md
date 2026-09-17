# Setup

Requirements:
- PyTorch >= 2.1
- Python >= 3.7
- CUDA >= 11.0
- [NVIDIA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) >= 7.0 (V100/RTX20 and higher)
- Linux OS

If you encounter any problem about installation, you may want to raise an [issue](https://github.com/hpcaitech/ColossalAI/issues/new/choose) in this repository.


## Download From PyPI

The standard wheel contains Python code and extension sources, without precompiled CUDA extensions. Install it with

```shell
pip install colossalai
```

**Note: only Linux is supported for now**

To compile PyTorch extensions during installation, first install a supported CUDA-enabled PyTorch version, a matching CUDA Toolkit, a C++ compiler, and the Python build tools (`setuptools`, `wheel`, `ninja`, and `packaging`). Use a Linux environment where the build can access the target GPU. Then explicitly select the source distribution:

```shell
BUILD_EXT=1 pip install --no-binary=colossalai --no-build-isolation colossalai
```


`BUILD_EXT=1` does not rebuild a downloaded wheel. `--no-binary=colossalai` selects source, and `--no-build-isolation` allows the build to use your installed PyTorch. If ColossalAI is already installed, uninstall it before switching to a source build.

Without ahead-of-time compilation, extensions are compiled on first use when needed. This requires the same development tools; successful installation alone does not verify GPU execution.

## Download From Source

> The version of Colossal-AI will be in line with the main branch of the repository. Feel free to raise an issue if you encounter any problem.

```shell
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# install dependency
pip install -r requirements/requirements.txt

# install colossalai
BUILD_EXT=1 pip install --no-build-isolation .
```

To defer extension compilation until first use, omit `BUILD_EXT`. This does not disable fused operators or remove their build requirements:

```shell
pip install .
```

<!-- doc-test-command: echo "installation.md does not need test" -->
