import importlib
import pkgutil

import pytest

import manifold.embodiments
import manifold.embodiments as embodiments
from manifold.core.conventions import Frame, GripperFormat, RotationFormat
from manifold.core.embodiment import (
    BodyActionSpace,
    CompositeActionSpace,
    EEActionSpace,
    Embodiment,
    JointActionSpace,
)
from manifold.core.names import ControlMode, EmbodimentName
from manifold.embodiments import ALL as EMBODIMENTS
from manifold.embodiments import (
    DUAL_FRANKA_EE_DELTA,
    FRANKA_EE_DELTA,
    FRANKA_MOBILE_WHOLE_BODY,
    PANDA_OMRON_WHOLE_BODY,
)

# Read off the package directory, not off `ALL`, so a newly added file is checked
# whether or not the catalog has been told about it.
MODULES = sorted(module.name for module in pkgutil.iter_modules(manifold.embodiments.__path__))


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
        "aloha_agilex_joint_absolute",
        "droid_joint_absolute",
        "dual_franka_ee_delta",
        "dual_franka_leap_ee_delta",
        "dual_franka_mobile_leap_whole_body",
        "dual_franka_mobile_whole_body",
        "dual_franka_right_leap_ee_delta",
        "franka_allegro_joint_delta",
        "franka_ee_delta",
        "franka_joint_absolute",
        "franka_leap_ee_delta",
        "franka_mobile_leap_whole_body",
        "franka_mobile_whole_body",
        "panda_omron_whole_body",
        "rby1_wuji2_joint_absolute",
        "widowx_ee_delta",
    }
    assert {name.value for name in EmbodimentName} == set(MODULES)


def test_camera_names_keep_the_strings_that_registered_versions_pair_on():
    from manifold.sensors import CameraName

    assert {name.value for name in CameraName} == {
        "agentview",
        "agentview_depth",
        "agentview_left",
        "agentview_left_depth",
        "agentview_right",
        "agentview_right_depth",
        "head",
        "head_left",
        "head_right",
        "left_wrist",
        "over_shoulder_left",
        "over_shoulder_right",
        "right_wrist",
        "wrist",
        "wrist_depth",
    }


def test_the_robosuite_panda_grippers_close_at_plus_one():
    # PandaOmron and the LIBERO Franka both drive robosuite's PandaGripper, whose
    # `format_action` closes at +1 and opens at -1.
    omron = PANDA_OMRON_WHOLE_BODY.action
    franka = FRANKA_EE_DELTA.action
    assert isinstance(omron, CompositeActionSpace) and isinstance(omron.groups[0], EEActionSpace)
    assert isinstance(franka, EEActionSpace)

    assert omron.groups[0].gripper is GripperFormat.SIGNED_OPEN_LOW
    assert franka.gripper is GripperFormat.SIGNED_OPEN_LOW


def test_panda_omron_declares_its_whole_body():
    action = PANDA_OMRON_WHOLE_BODY.action
    proprio = PANDA_OMRON_WHOLE_BODY.proprioception

    assert isinstance(action, CompositeActionSpace)
    assert action.slices() == [
        ("ee", 0, 7),
        ("mobile_base", 7, 10),
        ("torso", 10, 11),
        ("mode_slot", 11, 12),
    ]
    widths = [getattr(proprio, field) for field in proprio.spec_fields()]
    assert sum(spec.expected_length() for spec in widths if spec is not None) == 24


def test_the_mobile_franka_moves_its_base_from_the_mount_point_then_its_arm():
    action = FRANKA_MOBILE_WHOLE_BODY.action

    assert isinstance(action, CompositeActionSpace)
    assert action.slices() == [("mobile_base", 0, 3), ("ee", 3, 10)]
    base = action.groups[0]
    assert isinstance(base, BodyActionSpace)
    assert base.frame is Frame.MOUNT


def test_the_dual_franka_drives_each_arm_in_turn_in_its_own_base_frame():
    action = DUAL_FRANKA_EE_DELTA.action

    assert isinstance(action, EEActionSpace)
    assert (action.per_arm_length(), action.expected_length()) == (7, 14)
    assert (action.rotation, action.gripper) == (RotationFormat.EULER_XYZ, GripperFormat.SIGNED)
    assert (action.frame, action.delta) == (Frame.BASE, True)


@pytest.mark.parametrize(
    ("embodiment", "slices"),
    [
        (embodiments.FRANKA_LEAP_EE_DELTA, [("ee", 0, 6), ("joint", 6, 22)]),
        (
            embodiments.DUAL_FRANKA_LEAP_EE_DELTA,
            [("ee0", 0, 6), ("joint0", 6, 22), ("ee1", 22, 28), ("joint1", 28, 44)],
        ),
        (
            embodiments.DUAL_FRANKA_RIGHT_LEAP_EE_DELTA,
            [("ee0", 0, 6), ("joint0", 6, 22), ("ee1", 22, 28), ("joint1", 28, 44)],
        ),
        (
            embodiments.FRANKA_MOBILE_LEAP_WHOLE_BODY,
            [("mobile_base", 0, 3), ("ee", 3, 9), ("joint", 9, 25)],
        ),
        (
            embodiments.DUAL_FRANKA_MOBILE_LEAP_WHOLE_BODY,
            [
                ("mobile_base0", 0, 3),
                ("ee0", 3, 9),
                ("joint0", 9, 25),
                ("mobile_base1", 25, 28),
                ("ee1", 28, 34),
                ("joint1", 34, 50),
            ],
        ),
        (
            embodiments.DUAL_FRANKA_MOBILE_WHOLE_BODY,
            [("mobile_base0", 0, 3), ("ee0", 3, 10), ("mobile_base1", 10, 13), ("ee1", 13, 20)],
        ),
    ],
    ids=lambda value: value.name if isinstance(value, Embodiment) else "",
)
def test_each_maniladder_robot_reads_its_groups_in_the_order_of_its_controllers(
    embodiment: Embodiment, slices: list[tuple[str, int, int]]
) -> None:
    action = embodiment.action

    assert isinstance(action, CompositeActionSpace)
    assert action.slices() == slices


def test_the_allegro_hand_takes_its_sixteen_joint_deltas_alone():
    action = embodiments.FRANKA_ALLEGRO_JOINT_DELTA.action

    assert isinstance(action, JointActionSpace)
    assert (action.dof, action.gripper, action.delta) == (16, None, True)
