"""CPU-only tests of the wheel inspector; synthetic bytes are not real CUDA binaries."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_cuda_wheel.py"
SPEC = importlib.util.spec_from_file_location("check_cuda_wheel", SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)
BUILD_SPEC = importlib.util.spec_from_file_location("build_cuda_wheel", SCRIPT.with_name("build_cuda_wheel.py"))
builder = importlib.util.module_from_spec(BUILD_SPEC)
CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "cuda_wheel_contract", SCRIPT.with_name("cuda_wheel_contract.py")
)
contract = importlib.util.module_from_spec(CONTRACT_SPEC)
with patch.dict(sys.modules, {"check_cuda_wheel": checker}):
    CONTRACT_SPEC.loader.exec_module(contract)
with patch.dict(sys.modules, {"check_cuda_wheel": checker, "cuda_wheel_contract": contract}):
    BUILD_SPEC.loader.exec_module(builder)


class TestCudaWheelInspection(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.wheel = Path(self.temp.name) / "colossalai-0.5.0-cp310-cp310-linux_x86_64.whl"

    def make_wheel(self, modules=checker.EXPECTED_MODULES, pure=False, duplicate=None):
        with zipfile.ZipFile(self.wheel, "w") as archive:
            archive.writestr(
                "colossalai-0.5.0.dist-info/WHEEL",
                f"Wheel-Version: 1.0\nRoot-Is-Purelib: {str(pure).lower()}\nTag: cp310-cp310-linux_x86_64\n",
            )
            for module in modules:
                archive.writestr(
                    f"colossalai/_C/{module}.cpython-310-x86_64-linux-gnu.so",
                    b"synthetic binary",
                )
            if duplicate:
                archive.writestr(f"colossalai/_C/{duplicate}.so", b"duplicate binary")

    @staticmethod
    def cuobjdump(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            "ELF file 1: kernel.sm_80.cubin\nELF file 2: kernel.sm_90.cubin\n",
            "",
        )

    def test_every_cuda_module_is_checked_and_cpu_module_is_exempt(self):
        self.make_wheel()
        with patch.object(checker.subprocess, "run", side_effect=self.cuobjdump) as run:
            report = checker.inspect_wheel(self.wheel)
        self.assertEqual(len(report["modules"]), 7)
        self.assertEqual(run.call_count, 6)
        self.assertEqual(
            {Path(call.args[0][-1]).stem for call in run.call_args_list},
            set(checker.CUDA_MODULES),
        )
        self.assertNotIn("architectures", report["modules"]["cpu_adam_x86"])

    def test_missing_module_fails_before_inspection(self):
        self.make_wheel(modules=checker.EXPECTED_MODULES[:-1])
        with patch.object(checker.subprocess, "run") as run, self.assertRaisesRegex(
            ValueError, "missing=.*scaled_upper"
        ):
            checker.inspect_wheel(self.wheel)
        run.assert_not_called()

    def test_one_cuda_module_missing_sm90_fails(self):
        self.make_wheel()

        def output(command, **kwargs):
            if Path(command[-1]).stem == "moe_cuda":
                return subprocess.CompletedProcess(command, 0, "ELF file 1: kernel.sm_80.cubin\n", "")
            return self.cuobjdump(command, **kwargs)

        with patch.object(checker.subprocess, "run", side_effect=output), self.assertRaisesRegex(
            ValueError, "moe_cuda.*sm_90"
        ):
            checker.inspect_wheel(self.wheel)

    def test_no_cuda_code_fails(self):
        self.make_wheel()
        result = subprocess.CompletedProcess([], 0, "No ELF files found", "")
        with patch.object(checker.subprocess, "run", return_value=result), self.assertRaisesRegex(
            ValueError, "missing.*sm_80"
        ):
            checker.inspect_wheel(self.wheel)

    def test_cpu_module_alone_is_not_a_cuda_wheel(self):
        self.make_wheel(modules=["cpu_adam_x86"])
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            checker.inspect_wheel(self.wheel)

    def test_duplicate_module_fails(self):
        self.make_wheel(duplicate="layernorm_cuda")
        with self.assertRaisesRegex(ValueError, "Duplicate.*layernorm_cuda"):
            checker.inspect_wheel(self.wheel)

    def test_pure_python_metadata_is_rejected(self):
        self.make_wheel(pure=True)
        with self.assertRaisesRegex(ValueError, "Root-Is-Purelib"):
            checker.inspect_wheel(self.wheel)

    def test_cuobjdump_failure_is_not_ignored(self):
        self.make_wheel()
        result = subprocess.CompletedProcess([], 1, "", "invalid ELF")
        with patch.object(checker.subprocess, "run", return_value=result), self.assertRaisesRegex(
            RuntimeError, "invalid ELF"
        ):
            checker.inspect_wheel(self.wheel)

    def test_architecture_suffix_does_not_count_as_generic_target(self):
        self.make_wheel()
        result = subprocess.CompletedProcess([], 0, "kernel.sm_80.cubin kernel.sm_90a.cubin", "")
        with patch.object(checker.subprocess, "run", return_value=result), self.assertRaisesRegex(
            ValueError, "missing.*sm_90"
        ):
            checker.inspect_wheel(self.wheel)

    def test_invalid_zip_fails(self):
        self.wheel.write_bytes(b"not a zip file")
        with self.assertRaises(zipfile.BadZipFile):
            checker.inspect_wheel(self.wheel)

    def test_unexpected_module_requires_an_inventory_update(self):
        self.make_wheel(modules=checker.EXPECTED_MODULES + ("new_cuda",))
        with self.assertRaisesRegex(ValueError, "unexpected=.*new_cuda"):
            checker.inspect_wheel(self.wheel)


class TestCleanBuildSources(unittest.TestCase):
    def test_edited_source_copied_but_old_build_outputs_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            source, destination = Path(temp) / "source", Path(temp) / "copy"
            source.mkdir()
            files = {
                "setup.py": "edited current source",
                "build/lib/old.py": "stale build",
                "dist/old.whl": "stale wheel",
                "colossalai/_C/old.so": "stale binary",
                "package.egg-info/PKG-INFO": "stale metadata",
            }
            for filename, content in files.items():
                path = source / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            with patch.object(builder, "capture", return_value="\0".join(files)):
                builder.copy_sources(source, destination)
            self.assertEqual((destination / "setup.py").read_text(), "edited current source")
            self.assertEqual(
                [p.relative_to(destination).as_posix() for p in destination.rglob("*")],
                ["setup.py"],
            )
            self.assertTrue((source / "build/lib/old.py").exists())

    def test_extension_symlink_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            source, destination = Path(temp) / "source", Path(temp) / "copy"
            (source / "extensions").mkdir(parents=True)
            (source / "extensions/utils.py").write_text("current source")
            (source / "colossalai/kernel").mkdir(parents=True)
            (source / "colossalai/kernel/extensions").symlink_to("../../extensions")
            files = "extensions/utils.py\0colossalai/kernel/extensions"
            with patch.object(builder, "capture", return_value=files):
                builder.copy_sources(source, destination)
            self.assertTrue((destination / "colossalai/kernel/extensions").is_symlink())
            self.assertEqual(
                (destination / "colossalai/kernel/extensions/utils.py").read_text(),
                "current source",
            )


if __name__ == "__main__":
    unittest.main()
