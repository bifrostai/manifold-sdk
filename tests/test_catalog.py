import importlib
import pkgutil

import pytest

import manifold.embodiments
from manifold.benchmarks import ALL
from manifold.core import Benchmark, Compatibility, PolicySignature, check_compatibility
from manifold.core.embodiment import Embodiment
from manifold.core.names import ControlMode, EmbodimentName
from manifold.embodiments import ALL as EMBODIMENTS

# Read off the package directory, not off `ALL`, so a newly added file is checked
# whether or not the catalog has been told about it.
MODULES = sorted(module.name for module in pkgutil.iter_modules(manifold.embodiments.__path__))


@pytest.mark.parametrize("benchmark", ALL, ids=lambda benchmark: benchmark.name)
def test_a_native_policy_is_compatible_with_each_shipped_benchmark(benchmark: Benchmark) -> None:
    # A policy that declares exactly what the benchmark publishes must pair,
    # with no adapter — this is the coherence check on the catalog as it grows.
    native = PolicySignature(
        action_space=benchmark.embodiment.action,
        proprioception=benchmark.observation_space.proprioception,
        cameras=list(benchmark.observation_space.cameras),
        instruction=benchmark.observation_space.instruction,
    )
    assert check_compatibility(native, benchmark).status is Compatibility.COMPATIBLE


@pytest.mark.parametrize("embodiment", EMBODIMENTS, ids=lambda embodiment: embodiment.name)
def test_an_embodiments_module_constant_and_name_agree(embodiment: Embodiment) -> None:
    # The catalog's naming convention, asserted mechanically: `Embodiment.name` is the
    # module it lives in and the upper-case constant that module exports, so a reader
    # who has one of the three can find the other two. What the name MEANS — the robot
    # or assembly, then its canonical control mode — a reader still has to check.
    module = importlib.import_module(f"manifold.embodiments.{embodiment.name}")
    assert getattr(module, embodiment.name.upper()) is embodiment
    assert getattr(manifold.embodiments, embodiment.name.upper()) is embodiment


def test_every_embodiment_in_the_catalog_has_a_distinct_name() -> None:
    # A repeated name is a silent pairing bug rather than an import error: a policy
    # server keys its pipelines on (benchmark name, embodiment name), so two entries
    # sharing a name make one of them unreachable.
    names = [embodiment.name for embodiment in EMBODIMENTS]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("module_name", MODULES)
def test_an_embodiments_name_ends_in_a_canonical_control_mode(module_name: str) -> None:
    # The half of the convention the per-entry check cannot see: a name that agrees with
    # itself can still invent a control mode. The suffix has to be one the SDK defines,
    # and the name has to start with a robot or an assembly.
    mode = next((mode for mode in ControlMode if module_name.endswith(f"_{mode}")), None)
    assert mode is not None, f"{module_name}: must end in one of {[m.value for m in ControlMode]}"
    assert module_name[: -len(mode) - 1], f"{module_name}: must start with a robot or an assembly"


def test_every_embodiments_module_is_listed_in_the_catalog() -> None:
    # `ALL` is both what a consumer enumerates and what the per-entry checks iterate, so a
    # module missing from it is invisible to a user and to every other test in this file.
    assert sorted(embodiment.name for embodiment in EMBODIMENTS) == MODULES


def test_embodiment_names_keep_the_strings_that_registered_versions_pair_on():
    # `EmbodimentName` lists exactly the shipped embodiments, under the strings a hello
    # and a registered version already carry.
    assert {name.value for name in EmbodimentName} == {
        "droid_joint_absolute",
        "franka_ee_delta",
        "franka_joint_absolute",
        "panda_omron_whole_body",
        "widowx_ee_delta",
    }
    assert {name.value for name in EmbodimentName} == set(MODULES)


def test_camera_names_keep_the_strings_that_registered_versions_pair_on():
    from manifold.sensors import CameraName

    assert {name.value for name in CameraName} == {
        "agentview",
        "agentview_depth",
        "agentview_left",
        "agentview_right",
        "over_shoulder_left",
        "wrist",
        "wrist_depth",
    }
