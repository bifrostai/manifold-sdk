"""Every adapter transforms every arm.

Each test feeds a two-armed value whose arms are IDENTICAL, then requires every arm to
come out identical too -- and different from what went in. An adapter that strides by
the whole step, or reads arm 0 alone, leaves some arm in its source form, so that arm
differs from the others whatever the adapter's own maths. Width assertions cannot see
this: a half-transformed value keeps its length.

Before these existed, three of the six adapters PR #16 fixed could be reverted to their
single-arm bodies with every test still passing, and two more with the same bug were
never fixed at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from manifold.adapters.action import (
    GripperPolarityAdapter,
    GripperThresholdAdapter,
    RotationFormatAdapter,
)
from manifold.adapters.action.gripper_threshold import UnifiedGripperThresholdAdapter
from manifold.adapters.observation import (
    DynamicFrameRebaseAdapter,
    FrameRebaseAdapter,
    ObservedGripperAdapter,
    ProprioRotationAdapter,
)
from manifold.core import (
    EEActionSpace,
    EEObservationSpec,
    Frame,
    GripperFormat,
    GripperObservationSpec,
    ObservationAdapter,
    ObservationSpace,
    Proprioception,
    RotationFormat,
    UnifiedActionSpace,
)
from manifold.core.values import Observation

_AA = RotationFormat.AXIS_ANGLE
_Q = RotationFormat.QUATERNION

# One arm's worth of each layout. The rotation is well away from identity and the
# gripper away from any target's fixed point, so every transform below changes it.
_EE_ARM = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9]  # position, axis-angle, gripper
_POSE_ARM = [0.1, 0.2, 0.3, 0.0, 0.0, 0.38268343, 0.92387953, 0.5, -0.5]  # pos, quat, 2 fingers


def _assert_every_arm_alike(out: object, per_arm: int, arms: int, one_arm_in: list[float]) -> None:
    """Every arm came out identical, and not as it went in."""
    values = np.asarray(out, dtype=np.float64)
    assert values.size == per_arm * arms, "the transform changed the width"
    groups = values.reshape(arms, per_arm)
    for arm in range(1, arms):
        assert np.allclose(groups[arm], groups[0]), f"arm {arm} was not transformed as arm 0 was"
    if per_arm == len(one_arm_in):
        assert not np.allclose(groups[0], one_arm_in), "the transform changed nothing"


def _pose_space(
    *,
    rotation: RotationFormat = _Q,
    frame: Frame = Frame.WORLD,
    encoding: GripperFormat | None = None,
    arm_count: int = 2,
) -> ObservationSpace:
    ee = EEObservationSpec(
        rotation=rotation,
        frame=frame,
        gripper=GripperObservationSpec(dim=2, encoding=encoding),
        arm_count=arm_count,
    )
    return ObservationSpace(proprioception=Proprioception(ee_pose=ee))


def _rebase_matrix() -> np.ndarray:
    # A quarter turn about z, then a shift: moves both position and rotation.
    matrix = np.eye(4)
    matrix[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    matrix[:3, 3] = [0.5, -0.25, 0.1]
    return matrix


# ------------------------------------------------------------------ actions


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_rotation_adapter_converts_every_arm(chunk_size: int) -> None:
    spec = EEActionSpace(
        rotation=_AA, gripper=GripperFormat.SIGNED, arm_count=2, chunk_size=chunk_size
    )
    adapter = RotationFormatAdapter(target=_Q)
    out = adapter.adapt(_EE_ARM * 2 * chunk_size, source=spec)
    produced = adapter.produce(spec)
    _assert_every_arm_alike(out, produced.per_arm_length(), 2 * chunk_size, _EE_ARM)


def test_gripper_polarity_remaps_every_arm() -> None:
    # Striding by the whole step remapped only the last arm's gripper, so the left
    # gripper reached the environment in the source polarity -- exactly inverted.
    spec = EEActionSpace(rotation=_AA, gripper=GripperFormat.SIGNED, arm_count=2)
    out = GripperPolarityAdapter(target=GripperFormat.SIGNED_OPEN_LOW).adapt(
        _EE_ARM * 2, source=spec
    )
    _assert_every_arm_alike(out, spec.per_arm_length(), 2, _EE_ARM)


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_gripper_threshold_thresholds_every_arm(chunk_size: int) -> None:
    spec = EEActionSpace(
        rotation=_AA, gripper=GripperFormat.UNSIGNED, arm_count=2, chunk_size=chunk_size
    )
    out = GripperThresholdAdapter(target=GripperFormat.SIGNED).adapt(
        _EE_ARM * 2 * chunk_size, source=spec
    )
    _assert_every_arm_alike(out, spec.per_arm_length(), 2 * chunk_size, _EE_ARM)


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_unified_gripper_threshold_thresholds_every_arm(chunk_size: int) -> None:
    # Offsetting by the whole payload step thresholded the last arm alone, handing the
    # first arm's raw continuous score to an environment reading a binary command.
    payload = EEActionSpace(
        rotation=_AA, gripper=GripperFormat.UNSIGNED, arm_count=2, chunk_size=chunk_size
    )
    spec = UnifiedActionSpace(width=16, payload=payload)
    padding = [0.0, 0.0]
    out = UnifiedGripperThresholdAdapter(target=GripperFormat.SIGNED).adapt(
        (_EE_ARM * 2 + padding) * chunk_size, source=spec
    )
    for step in range(chunk_size):
        base = step * spec.width
        payload_out = out[base : base + payload.per_step_length()]
        _assert_every_arm_alike(payload_out, payload.per_arm_length(), 2, _EE_ARM)
        assert out[base + payload.per_step_length() :][:2] == padding, "padding was touched"


# ------------------------------------------------------------- observations


def test_proprio_rotation_reencodes_every_arm() -> None:
    source = _pose_space()
    observation = Observation(state={"ee_pose": np.asarray(_POSE_ARM * 2, dtype=np.float32)})
    adapter = ProprioRotationAdapter(target=_AA)
    out = adapter.adapt(observation, source=source).state["ee_pose"]
    produced = adapter.produce(source).proprioception.ee_pose
    assert produced is not None
    per_arm = produced.per_arm_length()
    _assert_every_arm_alike(out, per_arm, 2, _POSE_ARM)


def test_frame_rebase_rebases_every_arm() -> None:
    source = _pose_space(frame=Frame.BASE)
    adapter = FrameRebaseAdapter(Frame.BASE, Frame.WIDOWX_BRIDGE_EE, _rebase_matrix()[:3, :3])
    observation = Observation(state={"ee_pose": np.asarray(_POSE_ARM * 2, dtype=np.float32)})
    out = adapter.adapt(observation, source=source).state["ee_pose"]
    _assert_every_arm_alike(out, len(_POSE_ARM), 2, _POSE_ARM)


def test_dynamic_frame_rebase_rebases_every_arm() -> None:
    source = _pose_space(frame=Frame.WORLD)
    adapter = DynamicFrameRebaseAdapter(Frame.WORLD, Frame.BASE)
    observation = Observation(
        state={
            "ee_pose": np.asarray(_POSE_ARM * 2, dtype=np.float32),
            "base_pose": _rebase_matrix().reshape(-1).astype(np.float32),
        }
    )
    out = adapter.adapt(observation, source=source).state["ee_pose"]
    _assert_every_arm_alike(out, len(_POSE_ARM), 2, _POSE_ARM)


def test_observed_gripper_remaps_every_arm() -> None:
    source = _pose_space(encoding=GripperFormat.SIGNED)
    observation = Observation(state={"ee_pose": np.asarray(_POSE_ARM * 2, dtype=np.float32)})
    out = (
        ObservedGripperAdapter(target=GripperFormat.UNSIGNED)
        .adapt(observation, source=source)
        .state["ee_pose"]
    )
    _assert_every_arm_alike(out, len(_POSE_ARM), 2, _POSE_ARM)


# --------------------------------------------- a spec that forgot arm_count


def _one_arm_spec_fed_two_arms() -> tuple[ObservationSpace, Observation]:
    # The easiest way to reach this: a two-armed benchmark whose author left arm_count
    # at its default of 1. The value carries two arms; the spec declares one.
    source = _pose_space(frame=Frame.WORLD, encoding=GripperFormat.SIGNED, arm_count=1)
    observation = Observation(
        state={
            "ee_pose": np.asarray(_POSE_ARM * 2, dtype=np.float32),
            "base_pose": np.eye(4).reshape(-1).astype(np.float32),
        }
    )
    return source, observation


@pytest.mark.parametrize(
    "adapter",
    [
        ProprioRotationAdapter(target=_AA),
        DynamicFrameRebaseAdapter(Frame.WORLD, Frame.BASE),
        ObservedGripperAdapter(target=GripperFormat.UNSIGNED),
    ],
    ids=type,
)
def test_an_ee_pose_longer_than_its_spec_is_refused(adapter: ObservationAdapter) -> None:
    # A per-arm loop takes exactly `arm_count` arms. Without a length check the second
    # arm would vanish -- no error, a shorter array handed on as if it were whole.
    source, observation = _one_arm_spec_fed_two_arms()
    with pytest.raises(ValueError, match="length mismatch"):
        adapter.adapt(observation, source=source)


def test_a_rebase_of_an_ee_pose_longer_than_its_spec_is_refused() -> None:
    source = _pose_space(frame=Frame.BASE, arm_count=1)
    observation = Observation(state={"ee_pose": np.asarray(_POSE_ARM * 2, dtype=np.float32)})
    adapter = FrameRebaseAdapter(Frame.BASE, Frame.WIDOWX_BRIDGE_EE, _rebase_matrix()[:3, :3])
    with pytest.raises(ValueError, match="length mismatch"):
        adapter.adapt(observation, source=source)
