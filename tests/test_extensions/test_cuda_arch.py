"""Build configuration tests; run without PyTorch, CUDA, or a GPU:

python -m unittest discover -s tests/test_extensions -v
"""

import os
import sys
import types
import unittest
import warnings
from contextlib import ExitStack
from unittest.mock import Mock, patch

from extensions import ALL_EXTENSIONS, utils
from extensions.cuda_extension import _CudaExtension


class TestCudaArch(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(os.environ))
        os.environ.pop("TORCH_CUDA_ARCH_LIST", None)

        self.torch = types.ModuleType("torch")
        self.torch.__version__ = "2.7.0"
        self.torch.cuda = Mock()
        self.torch.cuda.is_available.return_value = False
        for name in ("get_device_capability", "get_arch_list", "device_count"):
            getattr(self.torch.cuda, name).side_effect = AssertionError("Build configuration must not query a GPU")

        torch_utils = types.ModuleType("torch.utils")
        cpp_extension = types.ModuleType("torch.utils.cpp_extension")
        cpp_extension.CUDA_HOME = "/mock/cuda"
        cpp_extension.CUDAExtension = lambda **kwargs: kwargs
        torch_utils.cpp_extension = cpp_extension
        self.torch.utils = torch_utils
        stack.enter_context(
            patch.dict(
                sys.modules,
                {"torch": self.torch, "torch.utils": torch_utils, "torch.utils.cpp_extension": cpp_extension},
            )
        )
        self.cuda_version = stack.enter_context(
            patch.object(utils, "get_cuda_bare_metal_version", return_value=("12", "8"))
        )
        stack.enter_context(warnings.catch_warnings())
        warnings.simplefilter("ignore")

    def test_default_targets_by_toolkit_version(self):
        legacy = "6.0;6.1;6.2;7.0;7.5"
        modern = "8.0;8.6;8.9;9.0"
        cases = [
            ("10", "2", legacy),
            ("11", "0", legacy + ";8.0"),
            ("11", "1", legacy + ";8.0;8.6"),
            ("11", "7", legacy + ";8.0;8.6"),
            ("11", "8", legacy + ";" + modern),
            ("12", "0", legacy + ";" + modern),
            ("12", "7", legacy + ";" + modern),
            ("12", "8", legacy + ";" + modern + ";10.0;12.0"),
            ("13", "0", "7.5;" + modern + ";10.0;12.0"),
        ]
        for major, minor, expected in cases:
            with self.subTest(cuda=f"{major}.{minor}"):
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
                self.cuda_version.return_value = (major, minor)
                self.assertFalse(utils.set_cuda_arch_list("/mock/cuda"))
                self.assertEqual(os.environ["TORCH_CUDA_ARCH_LIST"], expected)

    def test_older_torch_does_not_receive_unknown_blackwell_targets(self):
        for version in ("2.5.1+cu124", "2.6.0"):
            with self.subTest(torch=version):
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
                self.torch.__version__ = version
                utils.set_cuda_arch_list("/mock/cuda")
                self.assertEqual(os.environ["TORCH_CUDA_ARCH_LIST"], "6.0;6.1;6.2;7.0;7.5;8.0;8.6;8.9;9.0")

    def test_blackwell_gate_accepts_prerelease_torch_versions(self):
        for version in ("2.7.0rc1", "2.8.0.dev20250101+cu128", "2.11.0"):
            with self.subTest(torch=version):
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
                self.torch.__version__ = version
                utils.set_cuda_arch_list("/mock/cuda")
                self.assertTrue(os.environ["TORCH_CUDA_ARCH_LIST"].endswith(";10.0;12.0"))

    def test_explicit_targets_are_preserved_with_or_without_gpu(self):
        for available in (False, True):
            for arch_list in ("8.0;9.0+PTX", "8.0 8.6+PTX", "Ampere"):
                with self.subTest(gpu=available, targets=arch_list):
                    self.torch.cuda.is_available.return_value = available
                    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
                    self.assertEqual(utils.set_cuda_arch_list("/mock/cuda"), available)
                    self.assertEqual(os.environ["TORCH_CUDA_ARCH_LIST"], arch_list)
        self.cuda_version.assert_not_called()

    def test_visible_gpu_without_override_is_left_to_pytorch(self):
        self.torch.cuda.is_available.return_value = True
        self.assertTrue(utils.set_cuda_arch_list("/mock/cuda"))
        self.assertNotIn("TORCH_CUDA_ARCH_LIST", os.environ)
        self.cuda_version.assert_not_called()

    def test_empty_override_uses_default_targets(self):
        os.environ["TORCH_CUDA_ARCH_LIST"] = ""
        utils.set_cuda_arch_list("/mock/cuda")
        self.assertIn("8.0", os.environ["TORCH_CUDA_ARCH_LIST"].split(";"))

    def test_warning_reports_actual_targets(self):
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;9.0+PTX"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            utils.set_cuda_arch_list("/mock/cuda")
        self.assertTrue(any("8.0;9.0+PTX" in str(w.message) for w in caught))

    def test_all_aot_cuda_extensions_leave_arch_flags_to_pytorch(self):
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;9.0+PTX"
        for extension_cls in ALL_EXTENSIONS:
            if not issubclass(extension_cls, _CudaExtension):
                continue
            with self.subTest(extension=extension_cls.__name__):
                extension = extension_cls()
                if not extension.support_aot:
                    continue
                configuration = extension.build_aot()
                flags = configuration["extra_compile_args"]["nvcc"]
                # Handwritten flags cause PyTorch's AOT builder to ignore the
                # architecture override, and duplicate flags in its JIT builder.
                self.assertFalse(any("arch" in flag or "gencode" in flag for flag in flags))
                self.assertEqual(os.environ["TORCH_CUDA_ARCH_LIST"], "8.0;9.0+PTX")


if __name__ == "__main__":
    unittest.main()
