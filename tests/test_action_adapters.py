"""Tests for the gripper threshold adapters and DiscreteBinarize.

Covers the action-adapter layer's lossy binarizers:

  1. GripperThresholdAdapter — thresholds an open-high gripper head on a plain
     EEActionSpace.

  2. UnifiedGripperThresholdAdapter — its whole-body sibling, thresholding the
     gripper inside a UnifiedActionSpace payload (the RLDX/RoboCasa use case: 12-D
     buffer, 7-D EE payload). Declared with from_spec=UnifiedActionSpace so it is
     reachable through check_compatibility.

  3. DiscreteBinarize — a new adapter that snaps one dimension of a
     UnifiedActionSpace per-step vector to a discrete low/high value.
"""

from __future__ import annotations

import pytest

from manifold.adapters.action import (
    DiscreteBinarize,
    GripperThresholdAdapter,
    UnifiedGripperThresholdAdapter,
    UnifiedSliceAdapter,
)
from manifold.core.benchmark import Benchmark
from manifold.core.conventions import GripperFormat, RotationFormat
from manifold.core.embodiment import (
    EEActionSpace,
    EEObservationSpec,
    Embodiment,
    GripperObservationSpec,
    Proprioception,
    UnifiedActionSpace,
)
from manifold.core.names import EmbodimentName
from manifold.core.pipeline import Pipeline
from manifold.core.policy import PolicySignature
from manifold.core.sensor import Camera
from manifold.sensors import CameraName

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _ee_unsigned() -> EEActionSpace:
    """Open-high UNSIGNED EE space: 3 pos + 3 axis-angle + 1 gripper = 7 D."""
    return EEActionSpace(
        rotation=RotationFormat.AXIS_ANGLE,
        gripper=GripperFormat.UNSIGNED,
        delta=True,
    )


def _unified_12d() -> UnifiedActionSpace:
    """12-D whole-body buffer with a 7-D UNSIGNED EE payload (RoboCasa shape).

    Layout of an action: [pos3, rot3, gripper1, base5] = 12 dims.
    Gripper sits at index 6 (payload.expected_length() - 1 = 6).
    """
    return UnifiedActionSpace(width=12, payload=_ee_unsigned())


def _one_step_values(gripper: float = 0.9) -> list[float]:
    """7-D arm values + 5-D base-pin tail for an action in the 12-D buffer."""
    arm = [0.1, -0.2, 0.3, 0.4, -0.4, 0.5, gripper]
    base = [0.0, 0.0, 0.0, 0.0, -1.0]
    return arm + base


# ---------------------------------------------------------------------------
# UnifiedGripperThresholdAdapter — UnifiedActionSpace path
# ---------------------------------------------------------------------------


def test_gripper_threshold_applies_to_unified_with_ee_payload() -> None:
    source = _unified_12d()
    adapter = UnifiedGripperThresholdAdapter(target=GripperFormat.SIGNED)
    assert adapter.applies(source)


def test_gripper_threshold_does_not_apply_when_unified_payload_has_no_gripper() -> None:
    payload_no_gripper = EEActionSpace(rotation=RotationFormat.AXIS_ANGLE, delta=True)
    source = UnifiedActionSpace(width=12, payload=payload_no_gripper)
    adapter = UnifiedGripperThresholdAdapter(target=GripperFormat.SIGNED)
    assert not adapter.applies(source)


def test_gripper_threshold_does_not_apply_when_unified_gripper_already_matches() -> None:
    payload_signed = EEActionSpace(
        rotation=RotationFormat.AXIS_ANGLE,
        gripper=GripperFormat.SIGNED,
        delta=True,
    )
    source = UnifiedActionSpace(width=12, payload=payload_signed)
    adapter = UnifiedGripperThresholdAdapter(target=GripperFormat.SIGNED)
    assert not adapter.applies(source)


def test_gripper_threshold_does_not_apply_to_unified_without_ee_payload() -> None:
    source = UnifiedActionSpace(width=32)  # no payload
    adapter = UnifiedGripperThresholdAdapter(target=GripperFormat.SIGNED)
    assert not adapter.applies(source)


def test_gripper_threshold_unified_does_not_apply_to_ee() -> None:
    # The unified adapter only matches UnifiedActionSpace; a plain EE space is the
    # EE adapter's job. (check_compatibility matches on exact from_spec, but applies
    # must also reject the wrong form for a hand-built chain.)
    source = _ee_unsigned()
    adapter = UnifiedGripperThresholdAdapter(target=GripperFormat.SIGNED)
    assert not adapter.applies(source)


def test_gripper_threshold_produce_returns_unified_with_updated_payload_gripper() -> None:
    source = _unified_12d()
    adapter = UnifiedGripperThresholdAdapter(target=GripperFormat.SIGNED)

    produced = adapter.produce(source)

    assert isinstance(produced, UnifiedActionSpace)
    assert produced.width == 12
    assert isinstance(produced.payload, EEActionSpace)
    assert produced.payload.gripper is GripperFormat.SIGNED


def test_gripper_threshold_unified_terminates_after_produce() -> None:
    source = _unified_12d()
    adapter = UnifiedGripperThresholdAdapter(target=GripperFormat.SIGNED)

    assert adapter.applies(source)
    produced = adapter.produce(source)
    # Once the payload gripper is the target, the adapter no longer fires.
    assert not adapter.applies(produced)


def test_gripper_threshold_unified_adapt_binarizes_gripper_dim_open() -> None:
    # Gripper value 0.9 > 0.5 cutoff → open → SIGNED maps open to +1.
    source = _unified_12d()
    adapter = UnifiedGripperThresholdAdapter(target=GripperFormat.SIGNED)
    values = _one_step_values(gripper=0.9)

    out = adapter.adapt(values, source=source)

    assert len(out) == 12
    # Gripper is at index 6 (payload.expected_length() - 1).
    assert out[6] == 1.0
    # All other dims pass through unchanged.
    assert out[:6] == values[:6]
    assert out[7:] == values[7:]


def test_gripper_threshold_unified_adapt_binarizes_gripper_dim_closed() -> None:
    # Gripper value 0.1 <= 0.5 cutoff → closed → SIGNED maps closed to -1.
    source = _unified_12d()
    adapter = UnifiedGripperThresholdAdapter(target=GripperFormat.SIGNED)
    values = _one_step_values(gripper=0.1)

    out = adapter.adapt(values, source=source)

    assert out[6] == -1.0


def test_gripper_threshold_unified_raises_when_payload_overflows_width() -> None:
    # A malformed buffer: the 7-D EE payload does not fit within a width-5 buffer.
    # The gripper offset (6) would index past the step, so adapt must raise rather
    # than corrupt a neighbouring dim. (validate_value sizes the buffer to width.)
    source = UnifiedActionSpace(width=5, payload=_ee_unsigned())
    adapter = UnifiedGripperThresholdAdapter(target=GripperFormat.SIGNED)
    with pytest.raises(ValueError, match="does not fit within"):
        adapter.adapt([0.1, 0.2, 0.3, 0.4, 0.5], source=source)


# ---------------------------------------------------------------------------
# GripperThresholdAdapter — EEActionSpace regression (byte-identical)
# ---------------------------------------------------------------------------


def test_gripper_threshold_ee_path_still_applies() -> None:
    source = _ee_unsigned()
    adapter = GripperThresholdAdapter(target=GripperFormat.SIGNED)
    assert adapter.applies(source)


def test_gripper_threshold_ee_path_produce_returns_ee() -> None:
    source = _ee_unsigned()
    adapter = GripperThresholdAdapter(target=GripperFormat.SIGNED)
    produced = adapter.produce(source)
    assert isinstance(produced, EEActionSpace)
    assert produced.gripper is GripperFormat.SIGNED


def test_gripper_threshold_ee_path_adapt_open() -> None:
    source = _ee_unsigned()
    adapter = GripperThresholdAdapter(target=GripperFormat.SIGNED)
    out = adapter.adapt([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9], source=source)
    assert out[-1] == 1.0


def test_gripper_threshold_ee_path_adapt_closed() -> None:
    source = _ee_unsigned()
    adapter = GripperThresholdAdapter(target=GripperFormat.SIGNED)
    out = adapter.adapt([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1], source=source)
    assert out[-1] == -1.0


# ---------------------------------------------------------------------------
# DiscreteBinarize
# ---------------------------------------------------------------------------


def _unified_32d() -> UnifiedActionSpace:
    """32-D whole-body buffer (pi0.5 style), no payload."""
    return UnifiedActionSpace(width=32)


def _unified_12d_no_payload() -> UnifiedActionSpace:
    return UnifiedActionSpace(width=12)


def test_discrete_binarize_applies_when_dim_in_range() -> None:
    source = _unified_32d()
    adapter = DiscreteBinarize(dim_index=0)
    assert adapter.applies(source)

    adapter_last = DiscreteBinarize(dim_index=31)
    assert adapter_last.applies(source)


def test_discrete_binarize_does_not_apply_when_dim_out_of_range() -> None:
    source = _unified_32d()
    assert not DiscreteBinarize(dim_index=32).applies(source)
    assert not DiscreteBinarize(dim_index=-1).applies(source)


def test_discrete_binarize_does_not_apply_to_non_unified() -> None:
    source = _ee_unsigned()
    adapter = DiscreteBinarize(dim_index=0)
    assert not adapter.applies(source)


def test_discrete_binarize_produce_returns_same_spec() -> None:
    source = _unified_32d()
    adapter = DiscreteBinarize(dim_index=5)
    produced = adapter.produce(source)
    assert produced is source


def test_discrete_binarize_adapt_snaps_target_dim_high() -> None:
    # dim_index=11 (control-mode slot in a 12-D buffer), value 0.8 > 0.5 → high (+1.0).
    source = _unified_12d_no_payload()
    adapter = DiscreteBinarize(dim_index=11)
    values = [0.0] * 11 + [0.8]

    out = adapter.adapt(values, source=source)

    assert out[11] == 1.0
    assert out[:11] == values[:11]


def test_discrete_binarize_adapt_snaps_target_dim_low() -> None:
    source = _unified_12d_no_payload()
    adapter = DiscreteBinarize(dim_index=11)
    values = [0.0] * 11 + [0.3]

    out = adapter.adapt(values, source=source)

    assert out[11] == -1.0


def test_discrete_binarize_at_threshold_maps_to_low() -> None:
    # Exactly at the threshold → low (strictly-above rule).
    source = _unified_12d_no_payload()
    adapter = DiscreteBinarize(dim_index=0, threshold=0.5)
    values = [0.5] + [0.0] * 11

    out = adapter.adapt(values, source=source)

    assert out[0] == -1.0


def test_discrete_binarize_custom_threshold() -> None:
    source = _unified_12d_no_payload()
    adapter = DiscreteBinarize(dim_index=0, threshold=0.8)

    values_below = [0.7] + [0.0] * 11
    values_above = [0.9] + [0.0] * 11

    assert adapter.adapt(values_below, source=source)[0] == -1.0
    assert adapter.adapt(values_above, source=source)[0] == 1.0


def test_discrete_binarize_custom_low_high() -> None:
    source = _unified_12d_no_payload()
    adapter = DiscreteBinarize(dim_index=3, threshold=0.5, low=0.0, high=2.0)

    values_high = [0.0] * 3 + [0.9] + [0.0] * 8
    values_low = [0.0] * 3 + [0.2] + [0.0] * 8

    assert adapter.adapt(values_high, source=source)[3] == 2.0
    assert adapter.adapt(values_low, source=source)[3] == 0.0


def test_discrete_binarize_is_lossy() -> None:
    assert DiscreteBinarize.lossless is False


# ---------------------------------------------------------------------------
# UnifiedGripperThresholdAdapter through check_compatibility / verify
# ---------------------------------------------------------------------------
#
# The regression that the earlier generalisation missed: a whole-body gripper
# threshold must be reachable through `check_compatibility`, which matches an
# adapter by exact from_spec type. Declaring from_spec=EEActionSpace made the
# unified branch dead through any checked pipeline; routing a UnifiedActionSpace
# policy through the checker and verify — not just calling adapt/applies directly —
# is what catches that. The chain is
# UnifiedGripperThresholdAdapter (UNSIGNED→SIGNED) then UnifiedSliceAdapter, landing
# on the benchmark's signed EE action space.


def _signed_ee() -> EEActionSpace:
    """The benchmark's EE action: signed gripper, axis-angle, matching the payload shape."""
    return EEActionSpace(
        rotation=RotationFormat.AXIS_ANGLE, gripper=GripperFormat.SIGNED, delta=True
    )


def _unified_through_check_pipeline() -> tuple[PolicySignature, Benchmark, Pipeline]:
    proprio = Proprioception(
        ee_pose=EEObservationSpec(
            rotation=RotationFormat.AXIS_ANGLE, gripper=GripperObservationSpec(dim=2)
        )
    )
    benchmark = Benchmark(
        name="suite",
        embodiment=Embodiment(
            name=EmbodimentName.FRANKA_EE_DELTA, action=_signed_ee(), proprioception=proprio
        ),
        sensors=[Camera(name=CameraName.AGENTVIEW, shape=(8, 8, 3))],
        instruction=False,
    )
    policy = PolicySignature(
        action_space=UnifiedActionSpace(width=12, payload=_ee_unsigned()),
        proprioception=proprio,
        cameras=[Camera(name=CameraName.AGENTVIEW, shape=(8, 8, 3))],
        instruction=False,
    )
    pipeline = Pipeline(
        action=[
            UnifiedGripperThresholdAdapter(target=GripperFormat.SIGNED),
            UnifiedSliceAdapter(),
        ]
    )
    return policy, benchmark, pipeline


def test_unified_gripper_threshold_routes_through_check_compatibility() -> None:
    from manifold.core.check import Compatibility, check_compatibility

    policy, benchmark, pipeline = _unified_through_check_pipeline()
    report = check_compatibility(policy, benchmark, pipeline)

    # The chain is accepted end to end: the unified threshold adapter is reachable
    # through the checker (its from_spec is UnifiedActionSpace), then the slice lands
    # on the benchmark's signed EE action space.
    assert report.ok, report.reasons
    assert report.status is Compatibility.COMPATIBLE_VIA_PIPELINE


def test_unified_gripper_threshold_verifies_through_pipeline() -> None:
    from manifold.core.verify import verify

    policy, benchmark, pipeline = _unified_through_check_pipeline()
    report = verify(policy, benchmark, pipeline)

    # verify folds an asymmetric probe through the real chain. The gripper-meaning
    # comparison lives on EE specs, so a unified SOURCE is length-only here; the
    # point is that the threshold-then-slice chain runs end to end and lands on the
    # benchmark's action length without raising — exercised through the pipeline, not
    # by calling adapt() in isolation.
    assert report.ok, report.reasons
    assert any(c.name == "action.length" and c.passed for c in report.checks)


@pytest.mark.parametrize(("gripper", "expected"), [(0.9, 1.0), (0.1, -1.0)])
def test_unified_gripper_threshold_thresholds_through_pipeline(
    gripper: float, expected: float
) -> None:
    # Drive an open and a closed action through the actual pipeline (threshold then
    # slice) and confirm the gripper is binarized to the benchmark's SIGNED
    # convention at the right offset.
    from manifold.core.state import PipelineState
    from manifold.core.values import Action

    source = UnifiedActionSpace(width=12, payload=_ee_unsigned())
    pipeline = Pipeline(
        action=[
            UnifiedGripperThresholdAdapter(target=GripperFormat.SIGNED),
            UnifiedSliceAdapter(),
        ]
    )
    result = pipeline.apply_action(
        Action.from_array(_one_step_values(gripper=gripper)), source=source, state=PipelineState()
    )
    out = [float(v) for v in result.values]
    # After slicing to the 7-D EE payload, the gripper is the last dim.
    assert len(out) == 7
    assert out[6] == expected
