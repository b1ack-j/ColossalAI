# CUDA wheel build and acceptance

This is the B4 build check, not a release workflow. It uploads CI artifacts only.
Step-by-step handoff: [Linux build](cuda-wheel-build-runbook.md) and
[GPU consumer acceptance](cuda-wheel-consumer-runbook.md).
The baseline is Linux x86_64, Python 3.10, PyTorch 2.5.1 cu124 and CUDA Toolkit 12.4.
The scripts also work with a matching Python 3.12 environment, but each Python
version needs its own binary wheel and acceptance run.

## Prerequisites and current scope

- B1 must be integrated before the **default architecture** build can pass. On
  the old implementation CUDA 12.4 defaults omit sm80/sm90, and some extensions
  query a real GPU while constructing compiler flags. Explicit targets alone do
  not repair all those old code paths.
- Use a Linux CUDA **devel** image with a host compiler and `cuobjdump`, without
  exposing GPU devices. The CI image is `nvidia/cuda:12.4.1-devel-ubuntu22.04`.
- The workflow installs the exact build-tool versions and Torch variant; run
  those same installation commands when reproducing locally. Runtime project
  dependencies are not needed just to compile and inspect the wheel.
- CUDA 12.8/13 and Torch 2.11 are later matrix entries requiring real C2 compiler
  compatibility work. This baseline does not claim those builds have passed.
- A Mac can run the parser tests. It cannot validate this Linux CUDA build or
  execute these GPU kernels natively.

## Run the build

From a Git checkout with B1 and this change, using the prepared Linux environment:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/build_cuda_wheel.py --output /tmp/colossal-wheel-default
CUDA_VISIBLE_DEVICES='' python scripts/build_cuda_wheel.py --output /tmp/colossal-wheel-explicit --arch-list '8.0;9.0'
```

Use a new output directory for every run. It must be outside the checkout.
The script verifies the Torch/Toolkit versions and that no GPU is visible,
sets `BUILD_EXT=1 FORCE_CUDA=1`, then builds in a temporary copy of the current
Git source files (including uncommitted edits). It excludes old build outputs.
Default mode removes any inherited `TORCH_CUDA_ARCH_LIST`, so a developer's
environment cannot accidentally hide the default-selection bug. Explicit mode
uses the requested list, but B4 acceptance still requires sm80 and sm90.

The CI workflow runs all CPU tests in `tests/test_extensions` on relevant pull
requests, including the B1 and precompiled-loading tests once those changes are
integrated. Trigger
**Validate CUDA wheel without a GPU** manually for the expensive build, choosing
default or explicit mode. Both should be run before accepting the build path.
The manual workflow must exist on the repository's default branch before GitHub
exposes its dispatch entry; a local invocation can be used during initial review.

Successful output contains the wheel, `environment.json`, `build.log`,
`wheel-inspection.json`, `SHA256SUMS`, `result.json`, and `build-report.json`.
The version-1 build report records Python major/minor, Torch release and CUDA
build version, Toolkit, OS/CPU architecture, requested targets and per-extension
targets actually observed by cuobjdump, wheel hash, source commit and source
snapshot hash. `source-snapshot.json` lists source file hashes, executable bits
and symlink targets; `source.tar.gz` preserves that copied source before building.
Only a successfully built and inspected wheel receives a passing build report.
These hashes associate artifacts; they are not publisher signatures.
Failures retain the
evidence produced so far. Compare `result.json`, not merely the workflow artifact
upload status. `environment.json` records the source commit/worktree state,
Python/Torch/Toolkit/compiler versions and installed distribution versions.

## What the inspector proves

```bash
python scripts/check_cuda_wheel.py /path/to/colossalai.whl --output /tmp/inspection.json
```

It requires a platform wheel and exactly these seven AOT extension modules:

| Module | Required GPU images |
| --- | --- |
| cpu_adam_x86 | None: it has only C++ source, although it links CUDA libraries |
| layernorm_cuda | sm_80 and sm_90 |
| moe_cuda | sm_80 and sm_90 |
| fused_optim_cuda | sm_80 and sm_90 |
| inference_ops_cuda | sm_80 and sm_90 |
| scaled_masked_softmax_cuda | sm_80 and sm_90 |
| scaled_upper_triangle_masked_softmax_cuda | sm_80 and sm_90 |

It runs `cuobjdump --list-elf` separately for each of the six GPU extensions.
Finding a target in only one extension is insufficient. FlashAttention Dao,
SDPA and NPU adapters are not AOT extensions and are not expected as `.so` files.
The module inventory is an explicit acceptance contract; update it when the
project adds or removes a precompiled extension.

This checks packaged GPU images, not GPU numerical correctness, compatibility
with every driver, or whether every callable kernel has every target. It does
not relabel a Linux wheel as manylinux. Check linked libraries and CPU portability
before public distribution: CPU Adam currently uses `-march=native`, which can
require CPU instructions specific to the build machine.

## B5 consumer acceptance still required

After repairing the precompiled-loading path, test the **same wheel hash** on
A100 and H100 in fresh runtime containers with matching Python/PyTorch and a
compatible driver. These containers must lack `nvcc` and a CUDA development
Toolkit; CUDA runtime libraries and the driver are still necessary.

1. Install dependencies from their normal sources, then download the exact trial
   wheel from TestPyPI using `--only-binary=:all: --no-deps`. Do not mix indexes
   when identifying the wheel being accepted. Compare its SHA256 with the build.
2. Start outside the source checkout and assert `colossalai.__file__` and loaded
   extension paths belong to the installed environment, not an editable checkout.
3. Use empty JIT caches and make `_CppExtension.build_jit` and
   `_CudaExtension.build_jit` fail immediately during the smoke run. This ensures
   a missing/broken wheel cannot pass by compiling a replacement on the GPU node.
4. Load all seven modules through the normal loaders and verify their `.so` paths.
   Run FusedAdam updates and LayerNorm forward/backward against Torch references;
   run at least one actual computation for each remaining GPU extension before
   declaring all packaged extensions usable. Synchronize CUDA to surface errors.
5. Record GPU model/capability, driver, Python/Torch, wheel hash and numerical test
   results separately for each GPU. A successful import alone is not acceptance.

The initial consumer smoke script is `scripts/smoke_prebuilt_wheel.py`. Run it
with the exact installed wheel from a directory outside the source checkout:

```bash
cd /tmp
python /path/to/checkout/scripts/smoke_prebuilt_wheel.py /path/to/exact.whl \
  --build-report /path/to/build-report.json --output /tmp/gpu-smoke.json
```

The build report is required. Wheel name/hash and runtime Python major/minor,
Torch release, torch.version.cuda and OS/CPU architecture must match before GPU
access or extension loading. An optional local `+cu124` Torch suffix may differ
only when the release and CUDA build version still match. The runtime does not
need the build Toolkit; its absence is intentional.
It checks the installed extension bytes against that wheel, prohibits JIT, loads
all seven modules and runs FusedAdam/AdamW and fused LayerNorm numerical checks.
The other extensions are imported but their kernels are not executed by this
initial smoke script. Those execution cases remain separate B5 work; neither a
successful script run nor a CPU syntax check proves all operators usable.

Without the separate precompiled-loading fix, the loader checks `CUDA_HOME`/`nvcc`
before importing precompiled modules; that fix is a prerequisite for this
no-Toolkit consumer test.
No TestPyPI/PyPI upload, GPU run, or CUDA compilation is implied by the CPU tests.

## CPU tests and references

```bash
python -m unittest discover -s tests/test_extensions -v
```

The tests use synthetic ZIP entries and mocked `cuobjdump` output; they validate
inventory and parsing failures, not real CUDA binaries.

- [PyTorch release installation combinations](https://pytorch.org/get-started/previous-versions/)
- [NVIDIA CUDA 12.4 binary inspection tools](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-binary-utilities/index.html)
- [CUDA container image](https://hub.docker.com/r/nvidia/cuda/tags?name=12.4.1-devel-ubuntu22.04)
