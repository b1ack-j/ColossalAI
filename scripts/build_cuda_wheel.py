"""Build and inspect the torch 2.5.1 / CUDA 12.4 wheel on a GPU-less Linux node."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from check_cuda_wheel import inspect_wheel
from cuda_wheel_contract import source_snapshot, validate_build_report


def capture(command, cwd=None):
    return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def copy_sources(source, destination):
    """Copy current Git source files, including edits, but never prior build outputs."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    files = capture(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=source,
    ).split("\0")
    for filename in filter(None, files):
        relative = Path(filename)
        if (
            relative.parts[0] in {"build", "dist"}
            or any(part.endswith(".egg-info") or part == "__pycache__" for part in relative.parts)
            or relative.suffix in {".so", ".o", ".obj", ".pyd", ".pyc"}
        ):
            continue
        original = source / relative
        target = destination / relative
        if not original.exists() and not original.is_symlink():
            continue  # A tracked source may have been deleted in the worktree.
        target.parent.mkdir(parents=True, exist_ok=True)
        if original.is_symlink():
            # Reject links outside this checkout and keep the copy self-contained.
            relative_target = original.resolve().relative_to(source)
            target.symlink_to(os.path.relpath(destination / relative_target, target.parent))
        elif original.is_file():
            shutil.copy2(original, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="A new directory; existing directories are rejected",
    )
    parser.add_argument(
        "--arch-list",
        help="Explicit TORCH_CUDA_ARCH_LIST; omit to exercise B1 defaults",
    )
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.is_relative_to(source):
        parser.error("--output must be outside the source checkout")
    if args.arch_list is not None and not args.arch_list.strip():
        parser.error("--arch-list cannot be empty")
    output.mkdir(parents=True, exist_ok=False)
    status = {
        "status": "failed",
        "source": str(source),
        "arch_mode": "explicit" if args.arch_list else "default",
    }
    try:
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("This baseline requires Linux x86_64; parser tests can run on other platforms")
        import torch
        from torch.utils.cpp_extension import CUDA_HOME

        nvcc = str(Path(CUDA_HOME or "/missing-cuda") / "bin/nvcc")
        versions = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_home": CUDA_HOME,
            "gpu_visible": torch.cuda.is_available(),
            "nvcc": capture([nvcc, "--version"]),
            "compiler": capture([os.environ.get("CXX", "c++"), "--version"]),
            "git_commit": capture(["git", "rev-parse", "HEAD"], source),
            "git_status": capture(["git", "status", "--short"], source),
            "packages": sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()),
        }
        (output / "environment.json").write_text(json.dumps(versions, indent=2) + "\n")
        if versions["gpu_visible"]:
            raise RuntimeError("A GPU is visible; run without GPU devices or set CUDA_VISIBLE_DEVICES='' before launch")
        if torch.__version__.split("+")[0] != "2.5.1" or torch.version.cuda != "12.4":
            raise RuntimeError("Install the torch==2.5.1 cu124 build for this baseline")
        if not re.search(r"release 12\.4(?:,|\s)", versions["nvcc"]):
            raise RuntimeError("Use CUDA Toolkit 12.4 for this baseline")
        env = os.environ.copy()
        env.update(BUILD_EXT="1", FORCE_CUDA="1", CUDA_VISIBLE_DEVICES="")
        env.setdefault("MAX_JOBS", "2")
        # In default mode an inherited value must not hide the B1 regression.
        env.pop("TORCH_CUDA_ARCH_LIST", None)
        if args.arch_list:
            env["TORCH_CUDA_ARCH_LIST"] = args.arch_list
        status["arch_list"] = env.get("TORCH_CUDA_ARCH_LIST", "<project default>")
        cuobjdump = str(Path(CUDA_HOME) / "bin/cuobjdump")
        with tempfile.TemporaryDirectory(prefix="colossal-wheel-build-") as temp:
            clean_source = Path(temp) / "source"
            clean_source.mkdir()
            copy_sources(source, clean_source)
            snapshot = source_snapshot(clean_source)
            (output / "source-snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
            with tarfile.open(output / "source.tar.gz", "w:gz", dereference=False) as archive:
                archive.add(clean_source, arcname="source")
            command = [sys.executable, "-m", "build", "--wheel", "--no-isolation"]
            with (output / "build.log").open("w") as log:
                log.write(f"command={command!r}\narch_list={status['arch_list']}\n")
                with subprocess.Popen(
                    command,
                    cwd=clean_source,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                ) as process:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    if process.wait():
                        raise RuntimeError(f"Wheel build failed; see {output / 'build.log'}")
            wheels = list((clean_source / "dist").glob("*.whl"))
            if len(wheels) != 1:
                raise RuntimeError(f"Expected exactly one fresh wheel, found {len(wheels)}")
            wheel = output / wheels[0].name
            shutil.copy2(wheels[0], wheel)
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            status.update(wheel=wheel.name, sha256=digest)
            (output / "SHA256SUMS").write_text(f"{digest}  {wheel.name}\n")
            report = inspect_wheel(wheel, cuobjdump=cuobjdump)
            (output / "wheel-inspection.json").write_text(json.dumps(report, indent=2) + "\n")
            build_report = validate_build_report(
                {
                    "schema_version": 1,
                    "artifact_kind": "cuda-aot",
                    "status": "passed",
                    "source": {
                        "git_commit": versions["git_commit"],
                        "snapshot_sha256": snapshot["sha256"],
                    },
                    "environment": {
                        "python": list(sys.version_info[:2]),
                        "torch": versions["torch"],
                        "torch_cuda": versions["torch_cuda"],
                        "toolkit": "12.4",
                        "system": platform.system(),
                        "machine": platform.machine(),
                    },
                    "targets": {
                        "mode": status["arch_mode"],
                        "requested": args.arch_list,
                        "observed": {
                            name: entry["architectures"]
                            for name, entry in report["modules"].items()
                            if "architectures" in entry
                        },
                    },
                    "wheel": {"filename": wheel.name, "sha256": digest},
                }
            )
            (output / "build-report.json").write_text(json.dumps(build_report, indent=2) + "\n")
            status["status"] = "passed"
    except Exception as error:
        status["error"] = str(error)
        raise
    finally:
        (output / "result.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
