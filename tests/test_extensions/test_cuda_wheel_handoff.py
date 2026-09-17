"""Handoff failures are tested with synthetic artifacts, never GPU acceptance."""

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = load("check_cuda_wheel")
with patch.dict(sys.modules, {"check_cuda_wheel": checker}):
    contract = load("cuda_wheel_contract")
with patch.dict(sys.modules, {"cuda_wheel_contract": contract, "check_cuda_wheel": checker}):
    smoke = load("smoke_prebuilt_wheel")
    builder = load("build_cuda_wheel")


class TestWheelHandoff(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.wheel = self.root / "colossalai-0.5.0-cp310-cp310-linux_x86_64.whl"
        self.wheel.write_bytes(b"synthetic: not an executable CUDA wheel")
        self.report = {
            "schema_version": 1,
            "status": "passed",
            "artifact_kind": "cuda-aot",
            "source": {"git_commit": "a" * 40, "snapshot_sha256": "b" * 64},
            "environment": {
                "python": [3, 10],
                "torch": "2.5.1+cu124",
                "torch_cuda": "12.4",
                "toolkit": "12.4",
                "system": "Linux",
                "machine": "x86_64",
            },
            "targets": {
                "mode": "default",
                "requested": None,
                "observed": {name: ["sm_80", "sm_90"] for name in checker.CUDA_MODULES},
            },
            "wheel": {"filename": self.wheel.name, "sha256": contract.sha256_file(self.wheel)},
        }
        self.manifest = self.root / "build-report.json"
        self.manifest.write_text(json.dumps(self.report))

    def arguments(self):
        return [str(self.wheel), "--build-report", str(self.manifest), "--output", str(self.root / "smoke.json")]

    def test_matching_identity_accepts_optional_cuda_version_suffix(self):
        report = contract.load_build_report(self.manifest)
        contract.verify_wheel(report, self.wheel)
        runtime = dict(report["environment"], torch="2.5.1")
        contract.verify_runtime(report, runtime)

    def test_each_environment_mismatch_is_rejected_before_gpu_access(self):
        wrong = {
            "python": [3, 11],
            "torch": "2.2.0",
            "torch_cuda": "12.1",
            "system": "Darwin",
            "machine": "aarch64",
        }
        torch = SimpleNamespace(__version__="2.5.1", version=SimpleNamespace(cuda="12.4"), cuda=Mock())
        for key, value in wrong.items():
            with self.subTest(field=key), patch.dict(sys.modules, {"torch": torch}), patch.object(
                smoke, "runtime_identity", return_value=dict(self.report["environment"], **{key: value})
            ), self.assertRaisesRegex(ValueError, f"Runtime {key} mismatch"):
                smoke.inspect_runtime({"environment": {}}, 0, self.report)
        torch.cuda.is_available.assert_not_called()

    def test_hash_mismatch_fails_before_runtime_and_writes_failure_report(self):
        self.wheel.write_bytes(b"different wheel with the same filename")
        with patch.object(smoke, "inspect_runtime") as runtime:
            self.assertEqual(smoke.main(self.arguments()), 1)
        runtime.assert_not_called()
        result = json.loads((self.root / "smoke.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertIn("SHA256", result["error"])
        self.assertEqual(result["checks"], [])

    def test_filename_mismatch_is_rejected(self):
        renamed = self.root / "different.whl"
        renamed.write_bytes(self.wheel.read_bytes())
        with self.assertRaisesRegex(ValueError, "filename"):
            contract.verify_wheel(self.report, renamed)

    def test_missing_or_invalid_json_has_actionable_error(self):
        self.manifest.unlink()
        with self.assertRaisesRegex(ValueError, "Cannot read build report"):
            contract.load_build_report(self.manifest)
        self.manifest.write_text("{broken")
        with self.assertRaisesRegex(ValueError, "Cannot read build report"):
            contract.load_build_report(self.manifest)

    def test_incomplete_or_failed_reports_are_rejected(self):
        cases = []
        for key in ("schema_version", "source", "environment", "targets", "wheel"):
            report = copy.deepcopy(self.report)
            del report[key]
            cases.append(report)
        for section, key in (("environment", "python"), ("environment", "toolkit"), ("source", "snapshot_sha256")):
            report = copy.deepcopy(self.report)
            del report[section][key]
            cases.append(report)
        cases.extend([dict(self.report, status="failed"), dict(self.report, artifact_kind="python-jit"), []])
        for report in cases:
            with self.subTest(report=report), self.assertRaises(ValueError):
                contract.validate_build_report(report)

    def test_no_gpu_cannot_produce_success_even_with_matching_manifest(self):
        torch = SimpleNamespace(
            __version__="2.5.1", version=SimpleNamespace(cuda="12.4"), cuda=Mock(is_available=Mock(return_value=False))
        )
        with patch.dict(sys.modules, {"torch": torch}), patch.object(
            smoke, "runtime_identity", return_value=self.report["environment"]
        ), patch.object(smoke, "load_native_modules") as native:
            self.assertEqual(smoke.main(self.arguments()), 1)
        native.assert_not_called()
        result = json.loads((self.root / "smoke.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertIn("real, visible NVIDIA GPU", result["error"])
        self.assertEqual(result["checks"], [])

    def test_build_report_is_required_by_cli(self):
        with self.assertRaises(SystemExit) as error:
            smoke.main([str(self.wheel), "--output", str(self.root / "result.json")])
        self.assertEqual(error.exception.code, 2)

    def test_source_snapshot_tracks_contents_and_symlinks_not_location(self):
        source = self.root / "source"
        source.mkdir()
        (source / "kernel.cu").write_text("first")
        (source / "link").symlink_to("kernel.cu")
        original = contract.source_snapshot(source)
        (source / "kernel.cu").write_text("other")
        self.assertNotEqual(original["sha256"], contract.source_snapshot(source)["sha256"])
        (source / "kernel.cu").write_text("first")
        moved = self.root / "moved"
        source.rename(moved)
        self.assertEqual(original, contract.source_snapshot(moved))
        (moved / "link").unlink()
        (moved / "link").symlink_to("different.cu")
        self.assertNotEqual(original["sha256"], contract.source_snapshot(moved)["sha256"])

    def test_builder_report_is_consumable_and_contains_observed_targets(self):
        source = self.root / "checkout"
        source.mkdir()
        output = self.root / "output"
        torch = SimpleNamespace(
            __version__="2.5.1+cu124",
            version=SimpleNamespace(cuda="12.4"),
            cuda=Mock(is_available=Mock(return_value=False)),
        )

        def capture(command, cwd=None):
            if command[:3] == ["git", "rev-parse", "HEAD"]:
                return "a" * 40
            return "Cuda compilation tools, release 12.4, V12.4.131"

        def copy_sources(original, destination):
            (destination / "setup.py").write_text("# synthetic source")

        def popen(command, cwd, **kwargs):
            dist = cwd / "dist"
            dist.mkdir()
            (dist / self.wheel.name).write_bytes(self.wheel.read_bytes())
            process = Mock(stdout=iter(["synthetic compile log\n"]))
            process.wait.return_value = 0
            manager = Mock()
            manager.__enter__ = Mock(return_value=process)
            manager.__exit__ = Mock(return_value=False)
            return manager

        inspected = {"modules": {name: {"architectures": ["sm_80", "sm_90"]} for name in checker.CUDA_MODULES}}
        modules = {
            "torch": torch,
            "torch.utils": SimpleNamespace(),
            "torch.utils.cpp_extension": SimpleNamespace(CUDA_HOME="/mock/cuda"),
        }
        with patch.dict(sys.modules, modules), patch.object(
            builder.platform, "system", return_value="Linux"
        ), patch.object(builder.platform, "machine", return_value="x86_64"), patch.object(
            builder, "capture", side_effect=capture
        ), patch.object(
            builder, "copy_sources", side_effect=copy_sources
        ), patch.object(
            builder.platform, "platform", return_value="Linux-x86_64-mocked"
        ), patch.object(
            builder.importlib.metadata, "distributions", return_value=[]
        ), patch.object(
            builder.subprocess, "Popen", side_effect=popen
        ), patch.object(
            builder, "inspect_wheel", return_value=inspected
        ), patch.object(
            sys, "argv", ["build", "--source", str(source), "--output", str(output)]
        ):
            builder.main()
        report = contract.load_build_report(output / "build-report.json")
        contract.verify_wheel(report, output / self.wheel.name)
        snapshot = json.loads((output / "source-snapshot.json").read_text())
        self.assertEqual(report["source"]["snapshot_sha256"], snapshot["sha256"])
        self.assertEqual(report["targets"]["observed"], self.report["targets"]["observed"])
        self.assertTrue((output / "source.tar.gz").is_file())


if __name__ == "__main__":
    unittest.main()
