#!/usr/bin/env python3
"""Validate an installed Linux x86_64 CUDA wheel on one real NVIDIA GPU.

Run outside a source checkout, in a fresh environment without nvcc:
    python /path/to/smoke_prebuilt_wheel.py /path/to/colossalai.whl \
        --build-report /path/to/build-report.json --output /tmp/smoke.json

All seven native modules are imported and loaded with JIT forbidden. Numerical
checks cover only FP32 FusedAdam and MixedFusedLayerNorm, not every CUDA kernel.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import time
import traceback
import zipfile
from contextlib import ExitStack
from datetime import datetime, timezone
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from cuda_wheel_contract import load_build_report, runtime_identity, verify_runtime, verify_wheel

MODULE_LOADERS = {
    "cpu_adam_x86": "CPUAdamLoader",
    "layernorm_cuda": "LayerNormLoader",
    "moe_cuda": "MoeLoader",
    "fused_optim_cuda": "FusedOptimizerLoader",
    "inference_ops_cuda": "InferenceOpsLoader",
    "scaled_masked_softmax_cuda": "ScaledMaskedSoftmaxLoader",
    "scaled_upper_triangle_masked_softmax_cuda": "ScaledUpperTriangleMaskedSoftmaxLoader",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path):
    with path.open("rb") as stream:
        return sha256_stream(stream)


def within(path, directory):
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def check(report, name, function):
    record = {"name": name, "status": "running"}
    report["checks"].append(record)
    try:
        result = function()
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    record.update(status="passed", details=result)
    return result


def inspect_runtime(report, device_index, build_report):
    import torch

    environment = report["environment"]
    environment.update(torch=str(torch.__version__), torch_cuda=torch.version.cuda)
    identity = runtime_identity(torch)
    environment["identity"] = identity
    verify_runtime(build_report, identity)
    environment["cuda_available"] = torch.cuda.is_available()
    require(environment["cuda_available"], "A real, visible NVIDIA GPU is required; CPU/Mac execution cannot pass.")
    require(torch.version.cuda is not None, "CUDA-enabled PyTorch is required.")
    require(sys.platform == "linux" and platform.machine() == "x86_64", "This wheel contract requires Linux x86_64.")
    require(0 <= device_index < torch.cuda.device_count(), f"CUDA device {device_index} does not exist.")
    torch.cuda.set_device(device_index)
    torch.cuda.synchronize(device_index)
    properties = torch.cuda.get_device_properties(device_index)
    environment["gpu"] = {
        "index": device_index,
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": properties.total_memory,
    }
    nvcc = shutil.which("nvcc")
    compiler_candidates = [Path(nvcc)] if nvcc else []
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(variable):
            compiler_candidates.append(Path(os.environ[variable]) / "bin/nvcc")
    compiler_candidates.append(Path("/usr/local/cuda/bin/nvcc"))
    compilers = [str(path) for path in compiler_candidates if path.is_file()]
    environment["visible_nvcc_paths"] = compilers
    require(not compilers, "Use a runtime environment without nvcc; a development image is not B5 consumer acceptance.")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            environment["nvidia_smi"] = {
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            environment["nvidia_smi"] = {"error": str(error)}
    else:
        environment["nvidia_smi"] = {"available": False}
    return torch


class InstalledWheel:
    def __init__(self, archive):
        self.archive = archive
        names = archive.namelist()
        require(len(names) == len(set(names)), "The wheel contains duplicate ZIP paths.")
        metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
        require(len(metadata_paths) == 1, "The wheel must have exactly one distribution METADATA file.")
        metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
        require(metadata["Name"].lower().replace("_", "-") == "colossalai", "Expected a ColossalAI wheel.")
        self.version = metadata["Version"]
        site_roots = {Path(sysconfig.get_path(key)).resolve() for key in ("purelib", "platlib")}
        spec = importlib.util.find_spec("colossalai")
        require(spec is not None and spec.origin is not None, "ColossalAI is not installed.")
        origin = Path(spec.origin).resolve()
        require(any(within(origin, root) for root in site_roots), f"Source/editable import is forbidden: {origin}")
        self.package_root = origin.parent
        self.site_root = self.package_root.parent
        distribution = importlib.metadata.distribution("colossalai")
        metadata_root = Path(distribution.locate_file("")).resolve()
        require(metadata_root == self.site_root, f"Distribution metadata is outside this installation: {metadata_root}")
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        require(not direct_url.get("dir_info", {}).get("editable", False), "Editable installations cannot pass.")
        require(distribution.version == self.version, "Installed version differs from the supplied wheel.")
        self.verified_files = {}
        self.verify_file(origin)

    def verify_file(self, path):
        path = Path(path).resolve()
        require(within(path, self.package_root), f"Loaded file is outside the installed package: {path}")
        relative_path = path.relative_to(self.site_root).as_posix()
        require(
            relative_path in self.archive.namelist(), f"Loaded file is absent from the supplied wheel: {relative_path}"
        )
        with self.archive.open(relative_path) as stream:
            expected = sha256_stream(stream)
        actual = sha256_file(path)
        require(actual == expected, f"Installed file SHA256 differs from the wheel: {relative_path}")
        self.verified_files[relative_path] = {"installed_path": str(path), "sha256": actual}
        return path

    def native_file(self, name):
        matches = [
            path
            for path in self.archive.namelist()
            if PurePosixPath(path).parent == PurePosixPath("colossalai/_C")
            and PurePosixPath(path).name.split(".", 1)[0] == name
            and path.endswith(".so")
        ]
        require(len(matches) == 1, f"Expected exactly one .so for {name}, found {matches}.")
        return self.verify_file(self.site_root / matches[0])

    def python_module(self, name):
        module = importlib.import_module(name)
        self.verify_file(module.__file__)
        return module


def load_native_modules(report, installation, torch):
    loader_module = installation.python_module("colossalai.kernel.kernel_loader")
    loaded = {}
    for name, loader_name in MODULE_LOADERS.items():

        def load_one():
            expected_path = installation.native_file(name)
            module = importlib.import_module(f"colossalai._C.{name}")
            require(Path(module.__file__).resolve() == expected_path, f"Unexpected native module path for {name}.")
            require(module.__file__.endswith(".so"), f"{name} is not a native .so module.")
            loader = getattr(loader_module, loader_name)()
            require(loader.load() is module, f"Default {loader_name} did not load the verified module.")
            require(loader.load(name) is module, f"Explicit {loader_name} did not load the verified module.")
            torch.cuda.synchronize()
            loaded[name] = module
            return {"module": module.__name__, "file": str(expected_path), "execution": "import_and_loader_only"}

        check(report, f"prebuilt_module:{name}", load_one)
    return loaded


def check_fused_adam(torch, installation, loaded):
    FusedAdam = installation.python_module("colossalai.nn.optimizer.fused_adam").FusedAdam
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(20260915)
    shapes = [(37,), (17, 19), (262145,)]
    fused_parameters = [torch.nn.Parameter(torch.randn(shape, device=device, dtype=torch.float32)) for shape in shapes]
    reference_parameters = [torch.nn.Parameter(parameter.detach().clone()) for parameter in fused_parameters]
    options = {"lr": 0.003, "betas": (0.85, 0.97), "eps": 1e-6, "weight_decay": 0.01}
    fused = FusedAdam(fused_parameters, adamw_mode=True, bias_correction=True, **options)
    reference = torch.optim.AdamW(reference_parameters, foreach=False, fused=False, **options)
    require(
        fused.multi_tensor_adam is loaded["fused_optim_cuda"].multi_tensor_adam, "FusedAdam used an unverified kernel."
    )
    differences = []
    for _ in range(8):
        for actual, expected in zip(fused_parameters, reference_parameters):
            gradient = torch.randn_like(actual)
            actual.grad = gradient.clone()
            expected.grad = gradient.clone()
        fused.step()
        torch.cuda.synchronize()
        reference.step()
        torch.cuda.synchronize()
        maximum = 0.0
        for actual, expected in zip(fused_parameters, reference_parameters):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            maximum = max(maximum, (actual - expected).abs().max().item())
        differences.append(maximum)
    return {
        "dtype": "float32",
        "steps": 8,
        "parameter_shapes": shapes,
        "rtol": 1e-5,
        "atol": 1e-6,
        "max_absolute_difference_per_step": differences,
    }


def check_layernorm(torch, installation, loaded):
    layernorm_module = installation.python_module("colossalai.nn.layer.layernorm")
    device = torch.device("cuda", torch.cuda.current_device())
    results = []
    for shape in ((2, 5, 128), (3, 7, 1024)):
        fused = layernorm_module.MixedFusedLayerNorm(shape[-1], eps=1e-5, device=device, dtype=torch.float32)
        reference = torch.nn.LayerNorm(shape[-1], eps=1e-5, device=device, dtype=torch.float32)
        with torch.no_grad():
            fused.weight.uniform_(0.5, 1.5)
            fused.bias.uniform_(-0.1, 0.1)
        reference.load_state_dict(fused.state_dict())
        actual_input = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
        expected_input = actual_input.detach().clone().requires_grad_(True)
        actual_output = fused(actual_input)
        torch.cuda.synchronize()
        require(layernorm_module.layer_norm is loaded["layernorm_cuda"], "LayerNorm used an unverified kernel.")
        expected_output = reference(expected_input)
        gradient = torch.randn_like(actual_output)
        actual_output.backward(gradient.clone())
        torch.cuda.synchronize()
        expected_output.backward(gradient.clone())
        torch.cuda.synchronize()
        comparisons = {
            "output": (actual_output, expected_output),
            "input_gradient": (actual_input.grad, expected_input.grad),
            "weight_gradient": (fused.weight.grad, reference.weight.grad),
            "bias_gradient": (fused.bias.grad, reference.bias.grad),
        }
        differences = {}
        for name, (actual, expected) in comparisons.items():
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            differences[name] = (actual - expected).abs().max().item()
        results.append({"input_shape": shape, "max_absolute_difference": differences})
    return {"dtype": "float32", "rtol": 1e-4, "atol": 1e-5, "cases": results}


def run(args, report):
    wheel = args.wheel.resolve()
    require(wheel.is_file() and wheel.suffix == ".whl", f"Wheel does not exist: {wheel}")
    report["wheel"] = {"path": str(wheel), "sha256": sha256_file(wheel)}
    build_report = load_build_report(args.build_report)
    report["build_report"] = {
        "path": str(args.build_report.resolve()),
        "sha256": sha256_file(args.build_report),
        "contents": build_report,
    }
    verify_wheel(build_report, wheel)
    torch = inspect_runtime(report, args.device, build_report)
    with zipfile.ZipFile(wheel) as archive:
        installation = InstalledWheel(archive)
        report["installation"] = {"package_root": str(installation.package_root), "version": installation.version}
        report["verified_files"] = installation.verified_files
        cpp = installation.python_module("colossalai.kernel.extensions.cpp_extension")
        cuda = installation.python_module("colossalai.kernel.extensions.cuda_extension")
        installation.python_module("colossalai.kernel.extensions.base_extension")

        def forbid_jit(extension, *unused_args, **unused_kwargs):
            report["jit_attempts"].append(extension.name)
            raise RuntimeError(f"JIT compilation is forbidden in wheel acceptance: {extension.name}")

        with ExitStack() as stack:
            stack.enter_context(patch.object(cpp._CppExtension, "build_jit", forbid_jit))
            stack.enter_context(patch.object(cuda._CudaExtension, "build_jit", forbid_jit))
            loaded = load_native_modules(report, installation, torch)
            check(report, "FusedAdam_vs_AdamW_fp32", lambda: check_fused_adam(torch, installation, loaded))
            check(
                report,
                "MixedFusedLayerNorm_forward_backward_fp32",
                lambda: check_layernorm(torch, installation, loaded),
            )
            torch.cuda.synchronize()
        require(not report["jit_attempts"], "A JIT attempt occurred during acceptance.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wheel", type=Path, help="Exact wheel installed into the current environment")
    parser.add_argument("--build-report", required=True, type=Path, help="B4 build-report.json for this exact wheel")
    parser.add_argument("--output", required=True, type=Path, help="JSON report path, written on success or failure")
    parser.add_argument("--device", type=int, default=0, help="Visible CUDA device index (default: 0)")
    args = parser.parse_args(argv)
    started = time.monotonic()
    report = {
        "status": "failed",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "checks": [],
        "jit_attempts": [],
        "scope": {
            "native_modules_required": list(MODULE_LOADERS),
            "numerical_checks_planned": [
                "fused_optim_cuda:multi_tensor_adam",
                "layernorm_cuda:forward_affine/backward_affine",
            ],
            "other_modules": "Import and Loader validation only; their computational kernels are not tested.",
            "full_training_or_all_kernel_acceptance": False,
        },
    }
    try:
        run(args, report)
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        print(report["error"], file=sys.stderr)
    report["elapsed_seconds"] = time.monotonic() - started
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    except OSError as error:
        print(f"Could not write report: {error}", file=sys.stderr)
        return 1
    print(f"GPU wheel smoke: {report['status']}; report: {args.output.resolve()}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
