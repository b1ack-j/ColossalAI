"""CPU import checks; locally run with pytest --noconftest on CPU-only hosts."""

import ast
import importlib.util
import pickle
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]


def test_distributed_galore_compatibility_alias():
    from colossalai.nn import optimizer
    from colossalai.nn.optimizer import distributed_galore

    assert optimizer.DistGaloreAdamW is optimizer.DistGaloreAwamW
    assert distributed_galore.DistGaloreAwamW is optimizer.DistGaloreAdamW
    assert optimizer.optim2DistOptim[optimizer.GaLoreAdamW8bit] is optimizer.DistGaloreAdamW
    assert all(hasattr(distributed_galore, name) for name in distributed_galore.__all__)
    # A previously serialized reference must still resolve through the old name.
    old_reference = b"ccolossalai.nn.optimizer.distributed_galore\nDistGaloreAwamW\n."
    assert pickle.loads(old_reference) is optimizer.DistGaloreAdamW


def test_sequence_parallel_modes_share_one_definition():
    from colossalai.booster.plugin import hybrid_parallel_plugin, moe_hybrid_parallel_plugin
    from colossalai.shardformer.shard.shard_config import SUPPORT_SP_MODE

    assert hybrid_parallel_plugin.SUPPORT_SP_MODE is SUPPORT_SP_MODE
    assert moe_hybrid_parallel_plugin.SUPPORT_SP_MODE is SUPPORT_SP_MODE


def test_moe_exports_are_lazy_and_match_existing_operations():
    # A fresh interpreter verifies that importing the package does not eagerly
    # import _operation, even if another test has already imported MoE code.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import torch
import colossalai.moe as moe

assert 'colossalai.moe._operation' not in sys.modules
assert set(moe.__all__).issubset(dir(moe))
assert not hasattr(moe, 'nonexistent_operation')
namespace = {}
exec('from colossalai.moe import *', namespace)
from colossalai.moe import _operation
for name in moe.__all__:
    assert namespace[name] is getattr(_operation, name)
assert _operation.MOE_KERNEL is None
inputs = torch.tensor([[0, 1], [1, 0], [1, 1]])
torch.testing.assert_close(moe.moe_cumsum(inputs), torch.cumsum(inputs, dim=0) - 1)
""",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_policy_wildcard_exports_with_stubbed_optional_models(monkeypatch):
    # Verify __init__.py's export contract independently of optional GPU model
    # implementations. This is deliberately not a real inference-model import.
    path = ROOT / "colossalai/inference/modeling/policy/__init__.py"
    module_name = "_test_inference_policy_exports"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    for statement in ast.parse(path.read_text()).body:
        if isinstance(statement, ast.ImportFrom) and statement.level == 1:
            dependency_name = f"{module_name}.{statement.module}"
            dependency = ModuleType(dependency_name)
            for imported in statement.names:
                setattr(dependency, imported.name, type(imported.name, (), {}))
            monkeypatch.setitem(sys.modules, dependency_name, dependency)
    spec.loader.exec_module(module)
    namespace = {}
    exec(f"from {module_name} import *", namespace)
    assert namespace["model_policy_map"] is module.model_policy_map
    assert "model_polic_map" not in namespace


def test_dreambooth_requirements_parse():
    path = ROOT / "examples/images/dreambooth/requirements.txt"
    requirements = [Requirement(line) for line in path.read_text().splitlines() if line and not line.startswith("#")]
    diffusers = next(requirement for requirement in requirements if requirement.name == "diffusers")
    assert "0.5.0" in diffusers.specifier
    assert "0.4.0" not in diffusers.specifier
