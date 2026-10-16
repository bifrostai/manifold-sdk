"""A whole-body action: arm, torso and mobile-base groups, and the hybrid mode slot."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from manifold.core import (
    ActionAdapter,
    ActionSpace,
    Benchmark,
    Camera,
    Command,
    CompositeActionSpace,
    EEActionSpace,
    Embodiment,
    Frame,
    GripperFormat,
    HybridModeSlot,
    JointActionSpace,
    MobileBaseActionSpace,
    Pipeline,
    PolicySignature,
    Proprioception,
    RotationFormat,
    TorsoActionSpace,
    UnifiedActionSpace,
    check_compatibility,
    verify,
)
from manifold.core.names import EmbodimentName
from manifold.sensors import CameraName

_CAMERA = Camera(name=CameraName.AGENTVIEW, shape=(8, 8, 3))

# RoboCasa365's PandaOmron: the arm, the base, the torso lift, then the mode slot.
_RC365 = CompositeActionSpace(
    groups=(
        EEActionSpace(
            rotation=RotationFormat.AXIS_ANGLE,
            gripper=GripperFormat.SIGNED_OPEN_LOW,
            frame=Frame.BASE,
            delta=True,
        ),
        MobileBaseActionSpace(dof=3, command=Command.VELOCITY),
        TorsoActionSpace(command=Command.DELTA),
    ),
    hybrid_mode_slot=HybridModeSlot.BINARY_0_5,
)


def _benchmark(action: Any) -> Benchmark:
    return Benchmark(
        name="suite",
        embodiment=Embodiment(
            name=EmbodimentName.PANDA_OMRON_WHOLE_BODY,
            action=action,
            proprioception=Proprioception(),
        ),
        sensors=[_CAMERA],
        instruction=False,
    )


def _policy(action: Any) -> PolicySignature:
    return PolicySignature(
        action_space=action, proprioception=Proprioception(), cameras=[_CAMERA], instruction=False
    )


def _check(report: Any, name: str):  # type: ignore[no-untyped-def]
    matches = [c for c in report.checks if c.name == name]
    assert matches, f"no check named {name!r}; have {[c.name for c in report.checks]}"
    return matches[0]


class _Rewrite(ActionAdapter):
    """A bug simulation: rewrites the values of a composite action, spec unchanged."""

    from_spec: ClassVar[type[BaseModel]] = CompositeActionSpace
    to_spec: ClassVar[type[BaseModel]] = CompositeActionSpace
    lossless: ClassVar[bool] = True

    def __init__(self, rewrite: Callable[[list[float]], list[float]]) -> None:
        self.rewrite = rewrite

    def applies(self, source: BaseModel) -> bool:
        return isinstance(source, CompositeActionSpace)

    def produce(self, source: BaseModel) -> BaseModel:
        return source

    def adapt(self, values: Any, *, source: BaseModel) -> list[float]:
        return self.rewrite([float(v) for v in values])


def _set(index: int, value: float) -> Callable[[list[float]], list[float]]:
    def rewrite(values: list[float]) -> list[float]:
        values[index] = value
        return values

    return rewrite


def test_the_rc365_action_is_12_values_in_robocasa_order() -> None:
    assert _RC365.expected_length() == 12
    assert _RC365.slices() == [
        ("ee", 0, 7),
        ("mobile_base", 7, 10),
        ("torso", 10, 11),
        ("hybrid_mode", 11, 12),
    ]


def test_repeated_group_kinds_are_numbered() -> None:
    arm = EEActionSpace(rotation=RotationFormat.AXIS_ANGLE, gripper=GripperFormat.SIGNED)
    action = CompositeActionSpace(
        groups=(arm, MobileBaseActionSpace(dof=2, command=Command.VELOCITY), arm)
    )

    assert [name for name, _, _ in action.slices()] == ["ee0", "mobile_base", "ee1"]


def test_a_composite_needs_a_group() -> None:
    with pytest.raises(ValidationError):
        CompositeActionSpace(groups=())


def test_a_composite_round_trips_through_the_action_space_union() -> None:
    adapter: TypeAdapter[Any] = TypeAdapter(ActionSpace)

    parsed = adapter.validate_json(_RC365.model_dump_json())

    assert parsed == _RC365


@pytest.mark.parametrize(
    ("slot", "value", "high"),
    [
        (HybridModeSlot.BINARY_0, 0.0, False),
        (HybridModeSlot.BINARY_0, 0.01, True),
        (HybridModeSlot.BINARY_0_5, 0.49, False),
        (HybridModeSlot.BINARY_0_5, 0.5, True),
        (HybridModeSlot.BINARY_0, -1.0, False),
        (HybridModeSlot.BINARY_0_5, -1.0, False),
        (HybridModeSlot.BINARY_0, 1.0, True),
        (HybridModeSlot.BINARY_0_5, 1.0, True),
    ],
)
def test_the_mode_slot_cuts(slot: HybridModeSlot, value: float, high: bool) -> None:
    assert slot.is_high(value) is high


def test_a_composite_benchmark_reports_its_arm_gripper() -> None:
    assert _benchmark(_RC365).gripper_format is GripperFormat.SIGNED_OPEN_LOW


def test_an_identical_composite_is_compatible() -> None:
    report = check_compatibility(_policy(_RC365), _benchmark(_RC365))

    assert report.status.value == "compatible", report.reasons


def test_a_composite_with_another_mode_cut_is_refused() -> None:
    kitchen = _RC365.model_copy(update={"hybrid_mode_slot": HybridModeSlot.BINARY_0})

    report = check_compatibility(_policy(_RC365), _benchmark(kitchen))

    assert report.status.value == "incompatible"


def test_an_identical_rc365_pairing_verifies_every_group() -> None:
    report = verify(_policy(_RC365), _benchmark(_RC365), Pipeline())

    assert report.ok, report.reasons
    for name in (
        "action.ee.position",
        "action.ee.rotation",
        "action.ee.gripper",
        "action.ee.gripper.closed",
        "action.mobile_base",
        "action.torso",
        "action.hybrid_mode",
        "action.hybrid_mode.low",
    ):
        assert _check(report, name).passed, name


def test_a_joint_arm_group_is_compared_arm_by_arm() -> None:
    body = CompositeActionSpace(
        groups=(
            MobileBaseActionSpace(dof=3, command=Command.VELOCITY),
            JointActionSpace(dof=14, arm_count=2, gripper=GripperFormat.UNSIGNED),
        )
    )

    def swap(values: list[float]) -> list[float]:
        return values[:3] + values[11:19] + values[3:11]

    same = verify(_policy(body), _benchmark(body), Pipeline())
    swapped = verify(_policy(body), _benchmark(body), Pipeline(action=[_Rewrite(swap)]))

    assert same.ok, same.reasons
    assert _check(same, "action.joint.gripper.closed.arm1").passed
    assert not _check(swapped, "action.joint.joints.arm0").passed


def test_verify_catches_the_base_and_torso_swapped() -> None:
    def swap(values: list[float]) -> list[float]:
        return [*values[:7], values[10], values[7], values[8], values[9], values[11]]

    report = verify(_policy(_RC365), _benchmark(_RC365), Pipeline(action=[_Rewrite(swap)]))

    assert not report.ok
    assert _check(report, "action.ee.position").passed
    assert not _check(report, "action.mobile_base").passed
    assert not _check(report, "action.torso").passed


def test_verify_catches_a_mode_slot_held_high() -> None:
    report = verify(_policy(_RC365), _benchmark(_RC365), Pipeline(action=[_Rewrite(_set(11, 1.0))]))

    assert not report.ok
    assert _check(report, "action.hybrid_mode").passed
    assert not _check(report, "action.hybrid_mode.low").passed


def test_verify_catches_a_gripper_held_open() -> None:
    report = verify(_policy(_RC365), _benchmark(_RC365), Pipeline(action=[_Rewrite(_set(6, -1.0))]))

    assert not report.ok
    assert _check(report, "action.ee.gripper").passed
    assert not _check(report, "action.ee.gripper.closed").passed


class _Relabel(_Rewrite):
    """Produces `target`'s spec, and rewrites the values with `rewrite`."""

    def __init__(
        self, target: BaseModel, rewrite: Callable[[list[float]], list[float]] = list
    ) -> None:
        super().__init__(rewrite)
        self.target = target

    def produce(self, source: BaseModel) -> BaseModel:
        return self.target


def test_verify_catches_the_gripper_and_the_mode_slot_trading_places() -> None:
    arm = _RC365.groups[0].model_copy(update={"gripper": GripperFormat.SIGNED})
    body = _RC365.model_copy(
        update={"groups": (arm, *_RC365.groups[1:]), "hybrid_mode_slot": HybridModeSlot.BINARY_0}
    )

    def swap(values: list[float]) -> list[float]:
        values[6], values[11] = values[11], values[6]
        return values

    report = verify(_policy(body), _benchmark(body), Pipeline(action=[_Rewrite(swap)]))

    assert not report.ok
    assert not _check(report, "action.ee.gripper.closed").passed


@pytest.mark.parametrize(("first", "second"), [(slice(3, 6), slice(10, 13)), (6, 13)])
def test_verify_catches_two_arm_groups_trading_rotations_or_grippers(
    first: Any, second: Any
) -> None:
    arm = _RC365.groups[0]
    two_arms = CompositeActionSpace(groups=(arm, arm), hybrid_mode_slot=HybridModeSlot.BINARY_0)

    def swap(values: list[float]) -> list[float]:
        values[first], values[second] = values[second], values[first]
        return values

    report = verify(_policy(two_arms), _benchmark(two_arms), Pipeline(action=[_Rewrite(swap)]))

    assert not report.ok


def test_a_chain_that_reshapes_the_groups_is_not_checked() -> None:
    no_mode = _RC365.model_copy(update={"hybrid_mode_slot": None})
    drop_mode = _Relabel(no_mode, lambda values: values[:-1])

    report = verify(_policy(_RC365), _benchmark(no_mode), Pipeline(action=[drop_mode]))

    assert report.ok, report.reasons
    assert any("no group was compared" in reason for reason in report.not_checked)


def test_a_mismatched_arm_count_is_reported_not_raised() -> None:
    arm = _RC365.groups[0]
    assert isinstance(arm, EEActionSpace)
    two_arms = _RC365.model_copy(
        update={"groups": (arm.model_copy(update={"arm_count": 2}), *_RC365.groups[1:])}
    )

    def drop_second_arm(values: list[float]) -> list[float]:
        return [*values[:7], *values[14:]]

    report = verify(
        _policy(two_arms), _benchmark(_RC365), Pipeline(action=[_Relabel(_RC365, drop_second_arm)])
    )

    assert not report.ok
    assert not _check(report, "action.ee.arms").passed
    assert not any(c.name.startswith("action.ee.gripper.closed") for c in report.checks)


def test_a_probe_shorter_than_the_benchmark_is_reported_not_raised() -> None:
    def shorten_closed(values: list[float]) -> list[float]:
        # The arm's gripper is open-low, so a closed one is above 0.
        return values if values[6] < 0 else values[:-1]

    report = verify(
        _policy(_RC365), _benchmark(_RC365), Pipeline(action=[_Rewrite(shorten_closed)])
    )

    assert not report.ok
    assert not _check(report, "action.closed").passed


def test_verify_catches_a_mode_cut_that_was_only_relabelled() -> None:
    raw = _RC365.model_copy(update={"hybrid_mode_slot": HybridModeSlot.BINARY_0})

    report = verify(_policy(raw), _benchmark(_RC365), Pipeline(action=[_Relabel(_RC365)]))

    assert not report.ok
    assert _check(report, "action.hybrid_mode").passed
    assert _check(report, "action.hybrid_mode.low").passed
    assert not _check(report, "action.hybrid_mode.middle").passed


def test_a_torso_command_change_is_not_checked() -> None:
    torso = TorsoActionSpace(command=Command.VELOCITY)
    target = _RC365.model_copy(update={"groups": (*_RC365.groups[:2], torso)})

    report = verify(_policy(_RC365), _benchmark(target), Pipeline(action=[_Relabel(target)]))

    assert report.ok, report.reasons
    assert not any(c.name == "action.torso" for c in report.checks)
    assert any(reason.startswith("action.torso:") for reason in report.not_checked)


def test_a_unit_clip_on_the_base_and_torso_passes_the_probe() -> None:
    def clip(values: list[float]) -> list[float]:
        body = [min(1.0, max(-1.0, v)) for v in values[7:11]]
        return [*values[:7], *body, *values[11:]]

    report = verify(_policy(_RC365), _benchmark(_RC365), Pipeline(action=[_Rewrite(clip)]))

    assert report.ok, report.reasons


@pytest.mark.parametrize(
    ("model", "spec"),
    [
        (CompositeActionSpace, {"groups": [{"kind": "torso", "command": "delta"}]}),
        (TorsoActionSpace, {"command": "delta"}),
        (MobileBaseActionSpace, {"dof": 3, "command": "velocity"}),
    ],
)
def test_a_body_action_refuses_a_chunk(model: type[BaseModel], spec: dict[str, Any]) -> None:
    assert model.model_validate({**spec, "chunk_size": 1}) == model.model_validate(spec)
    with pytest.raises(ValidationError, match="chunk_size"):
        model.model_validate({**spec, "chunk_size": 8})


def test_a_composite_with_different_arm_grippers_has_no_one_gripper() -> None:
    arm = EEActionSpace(rotation=RotationFormat.AXIS_ANGLE, gripper=GripperFormat.SIGNED)
    other = EEActionSpace(rotation=RotationFormat.AXIS_ANGLE, gripper=GripperFormat.UNSIGNED)

    with pytest.raises(TypeError, match="do not share one gripper"):
        _ = _benchmark(CompositeActionSpace(groups=(arm, other))).gripper_format


def test_a_composite_with_one_arm_gripper_has_no_one_gripper() -> None:
    arm = EEActionSpace(rotation=RotationFormat.AXIS_ANGLE, gripper=GripperFormat.SIGNED)
    bare = EEActionSpace(rotation=RotationFormat.AXIS_ANGLE)

    with pytest.raises(TypeError, match="do not share one gripper"):
        _ = _benchmark(CompositeActionSpace(groups=(arm, bare))).gripper_format


def test_a_composite_against_a_unified_action_says_its_groups_went_unchecked() -> None:
    arm = _RC365.groups[0]
    assert isinstance(arm, EEActionSpace)
    unified = UnifiedActionSpace(width=12, payload=arm)

    report = verify(_policy(_RC365), _benchmark(unified), Pipeline(action=[_Relabel(unified)]))

    assert report.ok, report.reasons
    assert any("whole-body action on one side only" in r for r in report.not_checked)


def test_a_composite_without_grippers_or_mode_runs_one_probe() -> None:
    body = CompositeActionSpace(
        groups=(
            MobileBaseActionSpace(dof=3, command=Command.VELOCITY),
            TorsoActionSpace(command=Command.DELTA),
        )
    )
    calls: list[int] = []

    def count(values: list[float]) -> list[float]:
        calls.append(1)
        return values

    report = verify(_policy(body), _benchmark(body), Pipeline(action=[_Rewrite(count)]))

    assert report.ok, report.reasons
    assert len(calls) == 1


_TWO_ARMS = CompositeActionSpace(
    groups=(
        EEActionSpace(rotation=RotationFormat.AXIS_ANGLE, gripper=GripperFormat.SIGNED_OPEN_LOW),
        EEActionSpace(rotation=RotationFormat.AXIS_ANGLE, gripper=GripperFormat.SIGNED_OPEN_LOW),
        MobileBaseActionSpace(dof=3, command=Command.VELOCITY),
    )
)


def test_verify_catches_two_arm_groups_swapped() -> None:
    def swap(values: list[float]) -> list[float]:
        return [*values[7:14], *values[:7], *values[14:]]

    report = verify(_policy(_TWO_ARMS), _benchmark(_TWO_ARMS), Pipeline(action=[_Rewrite(swap)]))

    assert not report.ok
    assert not _check(report, "action.ee0.position").passed
    assert not _check(report, "action.ee1.position").passed


def test_verify_catches_an_arm_rotation_value_written_into_the_base() -> None:
    # The axis-angle arm probe has -0.5 at slot 4. With three base slots, an evenly
    # spaced base probe would put -0.5 at slot 7, so this write would go unseen.
    arm_and_base = CompositeActionSpace(groups=_RC365.groups[:2])
    rewrite = _Rewrite(_set(7, -0.5))

    report = verify(_policy(arm_and_base), _benchmark(arm_and_base), Pipeline(action=[rewrite]))

    assert not report.ok
    assert not _check(report, "action.mobile_base").passed
