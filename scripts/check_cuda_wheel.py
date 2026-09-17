"""Inspect a Linux x86_64 AOT wheel without importing it or using a GPU."""

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path

CUDA_MODULES = (
    "layernorm_cuda",
    "moe_cuda",
    "fused_optim_cuda",
    "inference_ops_cuda",
    "scaled_masked_softmax_cuda",
    "scaled_upper_triangle_masked_softmax_cuda",
)
# CPU Adam links CUDA libraries but contains no .cu source / GPU machine code.
EXPECTED_MODULES = ("cpu_adam_x86",) + CUDA_MODULES
MODULE_PATH = re.compile(r"^colossalai/_C/([a-z_][a-z_0-9]*)(?:\.[^/]+)?\.so$")


def inspect_wheel(wheel, required_arches=("sm_80", "sm_90"), cuobjdump="cuobjdump"):
    wheel = Path(wheel)
    required_arches = set(required_arches)
    if not required_arches or any(not re.fullmatch(r"sm_\d+[af]?", arch) for arch in required_arches):
        raise ValueError("Required architectures must use names such as sm_80 and sm_90")
    report = {
        "wheel": wheel.name,
        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "modules": {},
    }
    with zipfile.ZipFile(wheel) as archive, tempfile.TemporaryDirectory(prefix="colossal-wheel-inspect-") as temp:
        names = archive.namelist()
        wheel_metadata = [name for name in names if name.endswith(".dist-info/WHEEL")]
        if len(wheel_metadata) != 1:
            raise ValueError("Expected exactly one WHEEL metadata file")
        metadata = Parser().parsestr(archive.read(wheel_metadata[0]).decode())
        tags = metadata.get_all("Tag", [])
        if metadata.get("Root-Is-Purelib", "").lower() != "false":
            raise ValueError("Expected a platform wheel with Root-Is-Purelib: false")
        if not tags or any("linux" not in tag or not tag.endswith("x86_64") for tag in tags):
            raise ValueError(f"Expected Linux x86_64 wheel tags, got {tags}")
        report["tags"] = tags
        modules = {}
        for name in names:
            match = MODULE_PATH.fullmatch(name)
            if match:
                module = match.group(1)
                if module in modules:
                    raise ValueError(f"Duplicate extension module: {module}")
                modules[module] = name
        missing = set(EXPECTED_MODULES) - modules.keys()
        unexpected = modules.keys() - set(EXPECTED_MODULES)
        if missing or unexpected:
            raise ValueError(
                f"Extension inventory mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        for module, member in sorted(modules.items()):
            data = archive.read(member)
            entry = {
                "member": member,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            report["modules"][module] = entry
            if module not in CUDA_MODULES:
                entry["cuda_arch_check"] = "not applicable: CPU-only source"
                continue
            # Write only a fixed basename; never extract arbitrary archive paths.
            binary = Path(temp) / f"{module}.so"
            binary.write_bytes(data)
            result = subprocess.run([cuobjdump, "--list-elf", str(binary)], capture_output=True, text=True)
            entry["cuobjdump_output"] = result.stdout + result.stderr
            if result.returncode:
                raise RuntimeError(f"cuobjdump failed for {module}: {entry['cuobjdump_output']}")
            arches = {f"sm_{arch}" for arch in re.findall(r"(?<![a-z0-9])sm_(\d+[af]?)(?![a-z0-9])", result.stdout)}
            entry["architectures"] = sorted(arches)
            absent = required_arches - arches
            if absent:
                raise ValueError(f"{module} is missing {sorted(absent)}; found {sorted(arches)}")
    report["required_architectures"] = sorted(required_arches)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--require", nargs="+", default=["sm_80", "sm_90"])
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_wheel(args.wheel, args.require, args.cuobjdump)
    result = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()
