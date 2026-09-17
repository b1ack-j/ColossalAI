"""Versioned B4/B5 handoff contract. No Torch import or GPU access is needed here."""

import hashlib
import json
import platform
import re
import sys
from pathlib import Path

from check_cuda_wheel import CUDA_MODULES


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_snapshot(directory):
    """Hash copied source contents, relative symlink targets and executable bits."""
    root = Path(directory)
    files = []
    for path in sorted(root.rglob("*")):
        entry = {"path": path.relative_to(root).as_posix()}
        if path.is_symlink():
            entry.update(kind="symlink", target=str(path.readlink()))
        elif path.is_file():
            entry.update(kind="file", sha256=sha256_file(path), executable=bool(path.stat().st_mode & 0o111))
        else:
            continue
        files.append(entry)
    serialized = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(serialized).hexdigest(), "files": files}


def validate_build_report(report):
    require(isinstance(report, dict), "Build report must be a JSON object")
    require(type(report.get("schema_version")) is int and report["schema_version"] == 1, "Unsupported schema_version")
    require(report.get("artifact_kind") == "cuda-aot", "Build report must describe a cuda-aot wheel")
    require(report.get("status") == "passed", "Build report must have status=passed")
    for key in ("source", "environment", "targets", "wheel"):
        require(isinstance(report.get(key), dict), f"Build report is missing object: {key}")
    source, environment, targets, wheel = (report[key] for key in ("source", "environment", "targets", "wheel"))
    for key, length in (("git_commit", 40), ("snapshot_sha256", 64)):
        require(bool(re.fullmatch(rf"[0-9a-f]{{{length}}}", str(source.get(key, "")))), f"Invalid source.{key}")
    python = environment.get("python")
    require(
        isinstance(python, list) and len(python) == 2 and all(type(value) is int for value in python),
        "environment.python must contain [major, minor]",
    )
    require(python[0] == 3 and python[1] >= 10, "Build Python must be >=3.10")
    for key in ("torch", "torch_cuda", "toolkit", "system", "machine"):
        require(isinstance(environment.get(key), str) and bool(environment[key]), f"Missing environment.{key}")
    require(environment["torch"].split("+", 1)[0] == "2.5.1", "Build profile requires Torch 2.5.1")
    require(environment["torch_cuda"] == environment["toolkit"] == "12.4", "Build profile requires CUDA 12.4")
    require(
        environment["system"] == "Linux" and environment["machine"] == "x86_64", "Build profile requires Linux x86_64"
    )
    require(targets.get("mode") in ("default", "explicit"), "Invalid targets.mode")
    require("requested" in targets, "Missing targets.requested")
    if targets["mode"] == "default":
        require(targets["requested"] is None, "Default targets must have requested=null")
    else:
        require(
            isinstance(targets["requested"], str) and bool(targets["requested"].strip()), "Missing explicit targets"
        )
    observed = targets.get("observed")
    require(isinstance(observed, dict) and bool(observed), "Missing observed GPU targets")
    require(set(observed) == set(CUDA_MODULES), "Observed targets must cover exactly the six GPU extensions")
    for module, arches in observed.items():
        require(
            isinstance(arches, list) and all(isinstance(arch, str) for arch in arches),
            f"Invalid observed targets for {module}",
        )
        require({"sm_80", "sm_90"}.issubset(arches), f"Missing sm80/sm90 in observed targets for {module}")
    name = wheel.get("filename")
    require(isinstance(name, str) and Path(name).name == name and name.endswith(".whl"), "Invalid wheel.filename")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", str(wheel.get("sha256", "")))), "Invalid wheel.sha256")
    return report


def load_build_report(path):
    try:
        report = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read build report {path}: {error}") from error
    return validate_build_report(report)


def verify_wheel(report, wheel):
    validate_build_report(report)
    require(Path(wheel).name == report["wheel"]["filename"], "Wheel filename differs from the build report")
    require(sha256_file(wheel) == report["wheel"]["sha256"], "Wheel SHA256 differs from the build report")


def runtime_identity(torch):
    return {
        "python": list(sys.version_info[:2]),
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "system": platform.system(),
        "machine": platform.machine(),
    }


def verify_runtime(report, actual):
    validate_build_report(report)
    expected = report["environment"]
    for key in ("python", "torch", "torch_cuda", "system", "machine"):
        wanted, found = expected[key], actual.get(key)
        # Wheel installation may include or omit the local +cu124 version suffix.
        # Compare the release AND torch.version.cuda, never the release alone.
        if key == "torch":
            wanted = wanted.split("+", 1)[0]
            found = found.split("+", 1)[0] if isinstance(found, str) else found
        require(found == wanted, f"Runtime {key} mismatch: built for {wanted!r}, found {found!r}")
