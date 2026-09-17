"""Exercise the real loader with mocked GPU/runtime and native-module imports.

These tests require neither PyTorch nor CUDA. They do not execute GPU kernels.
Run with: python -m unittest discover -s tests/test_extensions -v
"""

import importlib.util
import sys
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import extensions
from extensions import CpuAdamArmExtension, CpuAdamX86Extension, FusedOptimizerCudaExtension
from extensions.base_extension import _Extension


def load_kernel_loader():
    # Execute the production loader without importing unrelated training modules.
    package_name = "_extension_loading_tests"
    package = types.ModuleType(package_name)
    package.__path__ = []
    loader_path = Path(__file__).resolve().parents[2] / "colossalai/kernel/kernel_loader.py"
    spec = importlib.util.spec_from_file_location(f"{package_name}.kernel_loader", loader_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            package_name: package,
            f"{package_name}.extensions": extensions,
            f"{package_name}.extensions.base_extension": sys.modules[_Extension.__module__],
        },
    ):
        spec.loader.exec_module(module)
    return module


class TestExtensionLoading(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        torch = types.ModuleType("torch")
        torch.__version__ = "2.5.1"
        torch.version = types.SimpleNamespace(cuda="12.4")
        torch.cuda = Mock()
        torch.cuda.is_available.return_value = True
        torch_utils = types.ModuleType("torch.utils")
        cpp_extension = types.ModuleType("torch.utils.cpp_extension")
        cpp_extension.CUDA_HOME = None
        torch_utils.cpp_extension = cpp_extension
        torch.utils = torch_utils
        stack.enter_context(
            patch.dict(
                sys.modules,
                {"torch": torch, "torch.utils": torch_utils, "torch.utils.cpp_extension": cpp_extension},
            )
        )
        self.torch = torch
        self.loader_module = load_kernel_loader()
        self.extension_cls = FusedOptimizerCudaExtension
        self.native_module = object()

    def test_prebuilt_load_does_not_require_compiler(self):
        for name in (None, "fused_optim_cuda"):
            with self.subTest(explicit_name=name):
                with (
                    patch.object(self.extension_cls, "import_op", return_value=self.native_module),
                    patch.object(self.extension_cls, "assert_build_compatible") as build_check,
                    patch.object(self.extension_cls, "build_jit") as build,
                ):
                    actual = self.loader_module.FusedOptimizerLoader().load(name)
                    self.assertIs(actual, self.native_module)
                    build_check.assert_not_called()
                    build.assert_not_called()

    def test_missing_prebuilt_module_checks_build_requirements_before_jit(self):
        for missing_name in ("colossalai._C", "colossalai._C.fused_optim_cuda"):
            with self.subTest(missing_name=missing_name):
                events = []
                error = ModuleNotFoundError("No prebuilt module", name=missing_name)
                with (
                    patch.object(self.extension_cls, "import_op", side_effect=error),
                    patch.object(
                        self.extension_cls, "assert_build_compatible", side_effect=lambda: events.append("check")
                    ),
                    patch.object(
                        self.extension_cls,
                        "build_jit",
                        side_effect=lambda: events.append("compile") or self.native_module,
                    ),
                ):
                    actual = self.loader_module.FusedOptimizerLoader().load()
                    self.assertIs(actual, self.native_module)
                    self.assertEqual(events, ["check", "compile"])

    def test_missing_compiler_still_blocks_jit(self):
        error = ModuleNotFoundError("No prebuilt module", name="colossalai._C")
        with (
            patch.object(self.extension_cls, "import_op", side_effect=error),
            patch.object(self.extension_cls, "build_jit") as build,
        ):
            with self.assertRaisesRegex(AssertionError, "CUDA_HOME"):
                self.loader_module.FusedOptimizerLoader().load()
            build.assert_not_called()

    def test_import_failures_preserve_the_original_error_without_jit(self):
        errors = [
            ImportError("undefined symbol: example_torch_symbol"),
            ImportError("libexample.so: cannot open shared object file"),
            ModuleNotFoundError("Missing extension dependency", name="extension_dependency"),
        ]
        for error in errors:
            with self.subTest(error=error):
                with (
                    patch.object(self.extension_cls, "import_op", side_effect=error),
                    patch.object(self.extension_cls, "build_jit") as build,
                    patch.object(self.extension_cls, "assert_build_compatible") as build_check,
                ):
                    with self.assertRaises(type(error)) as caught:
                        self.loader_module.FusedOptimizerLoader().load()
                    self.assertIs(caught.exception, error)
                    build.assert_not_called()
                    build_check.assert_not_called()

    def test_cuda_extension_rejects_cpu_only_torch_before_import(self):
        self.torch.version.cuda = None
        with patch.object(self.extension_cls, "import_op") as import_op:
            with self.assertRaisesRegex(RuntimeError, "CUDA-enabled PyTorch"):
                self.loader_module.FusedOptimizerLoader().load("fused_optim_cuda")
            import_op.assert_not_called()

    def test_cpu_architecture_checks_are_preserved_for_explicit_selection(self):
        for extension_cls, machine in ((CpuAdamX86Extension, "aarch64"), (CpuAdamArmExtension, "x86_64")):
            with self.subTest(extension=extension_cls.__name__):
                with patch("platform.machine", return_value=machine), patch.object(extension_cls, "import_op") as op:
                    with self.assertRaisesRegex(AssertionError, "CPU architecture"):
                        self.loader_module.CPUAdamLoader().load(extension_cls().name)
                    op.assert_not_called()

    def test_existing_custom_cuda_hardware_check_is_not_bypassed(self):
        class CustomExtension(FusedOptimizerCudaExtension):
            def assert_compatible(self):
                raise RuntimeError("custom hardware is incompatible")

        class CustomLoader(self.loader_module.KernelLoader):
            REGISTRY = [CustomExtension]

        for name in (None, "fused_optim_cuda"):
            with self.subTest(explicit_name=name), patch.object(CustomExtension, "import_op") as op:
                with self.assertRaisesRegex(RuntimeError, "custom hardware is incompatible"):
                    CustomLoader().load(name)
                op.assert_not_called()

    def test_real_importlib_uses_existing_prebuilt_module(self):
        native_module = types.ModuleType("colossalai._C.fused_optim_cuda")
        with (
            patch.dict(sys.modules, {native_module.__name__: native_module}),
            patch.object(self.extension_cls, "assert_build_compatible") as build_check,
            patch.object(self.extension_cls, "build_jit") as build,
        ):
            actual = self.loader_module.FusedOptimizerLoader().load()
            self.assertIs(actual, native_module)
            build_check.assert_not_called()
            build.assert_not_called()

    def test_real_importlib_missing_namespace_or_module_falls_back_to_jit(self):
        for namespace_present in (False, True):
            for name in (None, "fused_optim_cuda"):
                with self.subTest(namespace_present=namespace_present, explicit_name=name):
                    package = types.ModuleType("colossalai")
                    package.__path__ = []
                    namespace = types.ModuleType("colossalai._C")
                    namespace.__path__ = []
                    with (
                        patch.dict(sys.modules, {"colossalai": package}),
                        patch.object(self.extension_cls, "assert_build_compatible") as build_check,
                        patch.object(self.extension_cls, "build_jit", return_value=self.native_module) as build,
                    ):
                        sys.modules.pop("colossalai._C.fused_optim_cuda", None)
                        sys.modules.pop("colossalai._C", None)
                        if namespace_present:
                            sys.modules["colossalai._C"] = namespace
                        self.assertIs(self.loader_module.FusedOptimizerLoader().load(name), self.native_module)
                        build_check.assert_called_once_with()
                        build.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
