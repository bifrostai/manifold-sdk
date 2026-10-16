"""A whole-body action: arm, torso and mobile-base groups, and the mode slot."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from manifold.adapters.action import (
    BasePinWiden,
    GripperPolarityAdapter,
    GripperThresholdAdapter,
    GroupAdapter,
    RotationFormatAdapter,
)
from manifold.core import (
    ActionAdapter,
    ActionSpace,
    Benchmark,
    BodyActionSpace,
    BodyPart,
    Camera,
    Command,
    CompositeActionSpace,
    EEActionSpace,
    Embodiment,
    Frame,
    GripperFormat,
    JointActionSpace,
    ModeCut,
    Pipeline,
    PolicySignature,
    Proprioception,
    RotationFormat,
    UnifiedActionSpace,
    check_compatibility,
    verify,
)
from manifold.core.names import EmbodimentName
from manifold.core.verify import _probe_composite
from manifold.recipes.resolve import resolve
from manifold.sensors import CameraName

_CAMERA = Camera(name=CameraName.AGENTVIEW, shape=(8, 8, 3))

_BASE = BodyActionSpace(
    part=BodyPart.MOBILE_BASE, dof=3, command=Command.VELOCITY, frame=Frame.BASE
)
_TORSO = BodyActionSpace(part=BodyPart.TORSO, dof=1, command=Command.DELTA)

# RoboCasa365's PandaOmron: the arm, the base, the torso lift, then the mode slot.
_RC365 = CompositeActionSpace(
    groups=(
        EEActionSpace(
            rotation=RotationFormat.AXIS_ANGLE,
            gripper=GripperFormat.SIGNED_OPEN_LOW,
            frame=Frame.BASE,
            delta=True,
        ),
        _BASE,
        _TORSO,
    ),
    mode_slot=ModeCut.BINARY_0_5,
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
        ("mode_slot", 11, 12),
    ]


def test_repeated_group_kinds_are_numbered() -> None:
    arm = EEActionSpace(rotation=RotationFormat.AXIS_ANGLE, gripper=GripperFormat.SIGNED)
    action = CompositeActionSpace(groups=(arm, _BASE.model_copy(update={"dof": 2}), arm))

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
        (ModeCut.BINARY_0, 0.0, False),
        (ModeCut.BINARY_0, 0.01, True),
        (ModeCut.BINARY_0_5, 0.49, False),
        (ModeCut.BINARY_0_5, 0.5, True),
        (ModeCut.BINARY_0, -1.0, False),
        (ModeCut.BINARY_0_5, -1.0, False),
        (ModeCut.BINARY_0, 1.0, True),
        (ModeCut.BINARY_0_5, 1.0, True),
    ],
)
def test_the_mode_slot_cuts(slot: ModeCut, value: float, high: bool) -> None:
    action = _RC365.model_copy(update={"mode_slot": slot})

    assert action.mode_is_high([0.0] * 11 + [value]) is high


def test_an_action_without_a_mode_slot_has_no_mode() -> None:
    action = _RC365.model_copy(update={"mode_slot": None})

    with pytest.raises(ValueError, match="declares no mode slot"):
        action.mode_is_high([0.0] * 11)


def test_the_mode_is_read_from_a_whole_action_only() -> None:
    with pytest.raises(ValueError):
        _RC365.mode_is_high([1.0])


def test_a_composite_benchmark_reports_its_arm_gripper() -> None:
    assert _benchmark(_RC365).gripper_format is GripperFormat.SIGNED_OPEN_LOW


def test_an_identical_composite_is_compatible() -> None:
    report = check_compatibility(_policy(_RC365), _benchmark(_RC365))

    assert report.status.value == "compatible", report.reasons


def test_a_composite_with_another_mode_cut_is_refused() -> None:
    kitchen = _RC365.model_copy(update={"mode_slot": ModeCut.BINARY_0})

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
        "action.mode_slot",
        "action.mode_slot.low",
    ):
        assert _check(report, name).passed, name


def test_a_joint_arm_group_is_compared_arm_by_arm() -> None:
    body = CompositeActionSpace(
        groups=(
            _BASE,
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
    assert _check(report, "action.mode_slot").passed
    assert not _check(report, "action.mode_slot.low").passed


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
        update={"groups": (arm, *_RC365.groups[1:]), "mode_slot": ModeCut.BINARY_0}
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
    two_arms = CompositeActionSpace(groups=(arm, arm), mode_slot=ModeCut.BINARY_0)

    def swap(values: list[float]) -> list[float]:
        values[first], values[second] = values[second], values[first]
        return values

    report = verify(_policy(two_arms), _benchmark(two_arms), Pipeline(action=[_Rewrite(swap)]))

    assert not report.ok


def test_a_chain_that_reshapes_the_groups_is_not_checked() -> None:
    no_mode = _RC365.model_copy(update={"mode_slot": None})
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
    raw = _RC365.model_copy(update={"mode_slot": ModeCut.BINARY_0})

    report = verify(_policy(raw), _benchmark(_RC365), Pipeline(action=[_Relabel(_RC365)]))

    assert not report.ok
    assert _check(report, "action.mode_slot").passed
    assert _check(report, "action.mode_slot.low").passed
    assert not _check(report, "action.mode_slot.middle").passed


def test_a_torso_command_change_is_not_checked() -> None:
    torso = _TORSO.model_copy(update={"command": Command.VELOCITY})
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


def test_a_mobile_base_needs_a_frame() -> None:
    with pytest.raises(ValidationError, match="needs the frame"):
        BodyActionSpace(part=BodyPart.MOBILE_BASE, dof=3, command=Command.VELOCITY)


def test_a_torso_takes_no_frame() -> None:
    with pytest.raises(ValidationError, match="takes no frame"):
        BodyActionSpace(part=BodyPart.TORSO, dof=1, command=Command.DELTA, frame=Frame.BASE)


def test_a_body_group_is_written_with_its_part() -> None:
    written = _TORSO.model_dump(mode="json")

    assert written == {"kind": "body", "part": "torso", "dof": 1, "command": "delta", "frame": None}
    assert CompositeActionSpace.model_validate({"groups": [written]}).groups == (_TORSO,)


def test_a_torso_and_a_base_of_one_width_are_not_compared_as_one() -> None:
    torso = _TORSO.model_copy(update={"dof": 3})
    policy = CompositeActionSpace(groups=(_RC365.groups[0], _BASE))
    bench = CompositeActionSpace(groups=(_RC365.groups[0], torso))

    report = verify(_policy(policy), _benchmark(bench), Pipeline(action=[_Relabel(bench)]))

    assert not any(c.name.startswith(("action.torso", "action.mobile_base")) for c in report.checks)
    assert any("no group was compared" in reason for reason in report.not_checked)
    assert resolve(_policy(policy), _benchmark(bench), _arm_pool()) is None


@pytest.mark.parametrize(
    ("model", "spec"),
    [
        (
            CompositeActionSpace,
            {"groups": [{"kind": "body", "part": "torso", "dof": 1, "command": "delta"}]},
        ),
        (BodyActionSpace, {"part": "torso", "dof": 1, "command": "delta"}),
        (
            BodyActionSpace,
            {"part": "mobile_base", "dof": 3, "command": "velocity", "frame": "base"},
        ),
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
            _BASE,
            _TORSO,
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
        _BASE,
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


def _rc365_with(**arm_update: Any) -> CompositeActionSpace:
    arm = _RC365.groups[0].model_copy(update=arm_update)
    return _RC365.model_copy(update={"groups": (arm, *_RC365.groups[1:])})


_ARM = _RC365.groups[0]


class _HoldOpen(ActionAdapter):
    """A bug: always sends an open gripper."""

    from_spec: ClassVar[type[BaseModel]] = EEActionSpace
    to_spec: ClassVar[type[BaseModel]] = EEActionSpace
    lossless: ClassVar[bool] = True

    def applies(self, source: BaseModel) -> bool:
        return isinstance(source, EEActionSpace)

    def produce(self, source: BaseModel) -> BaseModel:
        return source

    def adapt(self, values: Any, *, source: BaseModel) -> list[float]:
        return [*(float(v) for v in values[:-1]), -1.0]


class _WidenQuaternionArm(ActionAdapter):
    """Widens a quaternion arm to RoboCasa365's whole body, with the rest held still."""

    from_spec: ClassVar[type[BaseModel]] = EEActionSpace
    to_spec: ClassVar[type[BaseModel]] = CompositeActionSpace
    lossless: ClassVar[bool] = True

    def applies(self, source: BaseModel) -> bool:
        return isinstance(source, EEActionSpace) and source.rotation is RotationFormat.QUATERNION

    def produce(self, source: BaseModel) -> BaseModel:
        return _RC365.model_copy(update={"groups": (source, *_RC365.groups[1:])})

    def adapt(self, values: Any, *, source: BaseModel) -> list[float]:
        return [*(float(v) for v in values), 0.0, 0.0, 0.0, 0.0, -1.0]


def test_a_lifted_rotation_adapter_bridges_a_quaternion_arm() -> None:
    policy = _policy(_rc365_with(rotation=RotationFormat.QUATERNION))
    pipeline = Pipeline(action=[GroupAdapter(0, RotationFormatAdapter(RotationFormat.AXIS_ANGLE))])

    assert check_compatibility(policy, _benchmark(_RC365)).status.value == "incompatible"
    report = check_compatibility(policy, _benchmark(_RC365), pipeline=pipeline)
    assert report.status.value == "compatible_via_pipeline", report.reasons
    assert report.lossless is True
    checked = verify(policy, _benchmark(_RC365), pipeline)
    assert checked.ok, checked.reasons
    assert _check(checked, "action.ee.rotation").passed
    assert _check(checked, "action.mobile_base").passed


@pytest.mark.parametrize(
    ("policy_update", "bench_update", "inner"),
    [
        (
            {"rotation": RotationFormat.QUATERNION},
            {},
            RotationFormatAdapter(RotationFormat.AXIS_ANGLE),
        ),
        (
            {"gripper": GripperFormat.SIGNED},
            {},
            GripperPolarityAdapter(GripperFormat.SIGNED_OPEN_LOW),
        ),
        (
            {"gripper": GripperFormat.UNSIGNED},
            {},
            GripperThresholdAdapter(GripperFormat.SIGNED_OPEN_LOW),
        ),
        ({}, {"gripper": GripperFormat.SIGNED}, GripperThresholdAdapter(GripperFormat.SIGNED)),
        ({}, {}, _HoldOpen()),
    ],
)
def test_a_lifted_adapter_verifies_as_it_does_on_a_bare_arm(
    policy_update: dict[str, Any], bench_update: dict[str, Any], inner: ActionAdapter
) -> None:
    bare_policy = _policy(_ARM.model_copy(update=policy_update))
    bare_bench = _benchmark(_ARM.model_copy(update=bench_update))
    whole_policy, whole_bench = (
        _policy(_rc365_with(**policy_update)),
        _benchmark(_rc365_with(**bench_update)),
    )
    bare, lifted = Pipeline(action=[inner]), Pipeline(action=[GroupAdapter(0, inner)])

    bare_report = verify(bare_policy, bare_bench, bare)
    lifted_report = verify(whole_policy, whole_bench, lifted)

    for name in ("position", "rotation", "gripper", "gripper.closed"):
        expected = _check(bare_report, f"action.{name}").passed
        assert _check(lifted_report, f"action.ee.{name}").passed == expected, name
    assert (
        check_compatibility(whole_policy, whole_bench, pipeline=lifted).lossless
        == check_compatibility(bare_policy, bare_bench, pipeline=bare).lossless
    )


def _arm_pool() -> list[ActionAdapter]:
    return [
        *(RotationFormatAdapter(rotation) for rotation in RotationFormat),
        *(GripperPolarityAdapter(gripper) for gripper in GripperFormat),
        *(GripperThresholdAdapter(gripper) for gripper in GripperFormat),
    ]


@pytest.mark.parametrize(
    "policy_update",
    [
        {"rotation": RotationFormat.QUATERNION},
        {"gripper": GripperFormat.UNSIGNED},
        {"rotation": RotationFormat.EULER_XYZ, "gripper": GripperFormat.SIGNED},
    ],
)
def test_the_resolver_finds_the_bare_arm_chain_lifted(policy_update: dict[str, Any]) -> None:
    pool = _arm_pool()

    bare = resolve(_policy(_ARM.model_copy(update=policy_update)), _benchmark(_ARM), pool)
    whole = resolve(_policy(_rc365_with(**policy_update)), _benchmark(_RC365), pool)

    assert bare is not None and whole is not None
    lifted = [adapter for adapter in whole.action if isinstance(adapter, GroupAdapter)]
    assert len(lifted) == len(whole.action)
    assert [adapter.inner for adapter in lifted] == list(bare.action)
    assert verify(_policy(_rc365_with(**policy_update)), _benchmark(_RC365), whole).ok


def test_the_resolver_lifts_adapters_onto_a_whole_body_action_reached_mid_chain() -> None:
    arm = _ARM.model_copy(update={"rotation": RotationFormat.QUATERNION})
    pool = [_WidenQuaternionArm(), RotationFormatAdapter(RotationFormat.AXIS_ANGLE)]

    pipeline = resolve(_policy(arm), _benchmark(_RC365), pool)

    assert pipeline is not None
    assert [repr(adapter) for adapter in pipeline.action[1:]] == [
        "GroupAdapter(0, RotationFormatAdapter)"
    ]
    assert isinstance(pipeline.action[0], _WidenQuaternionArm)


def test_the_resolver_wraps_only_adapters_that_keep_a_groups_kind() -> None:
    action = _rc365_with(rotation=RotationFormat.QUATERNION)
    pool = [BasePinWiden(), RotationFormatAdapter(RotationFormat.AXIS_ANGLE)]

    found = resolve(_policy(action), _benchmark(_RC365), pool)

    assert found is not None
    assert [repr(adapter) for adapter in found.action] == ["GroupAdapter(0, RotationFormatAdapter)"]
    assert resolve(_policy(action), _benchmark(_RC365), [BasePinWiden()]) is None


def test_the_resolver_bridges_many_arm_groups_one_at_a_time() -> None:
    off = _ARM.model_copy(
        update={"rotation": RotationFormat.QUATERNION, "gripper": GripperFormat.UNSIGNED}
    )
    torso = _RC365.groups[2]
    still = torso.model_copy(update={"command": Command.ABSOLUTE})
    policy_action = CompositeActionSpace(groups=(off, off, off, off, torso))
    target = CompositeActionSpace(groups=(_ARM, _ARM, _ARM, _ARM, torso))
    unreachable = CompositeActionSpace(groups=(_ARM, _ARM, _ARM, _ARM, still))

    start = time.perf_counter()
    found = resolve(_policy(policy_action), _benchmark(target), _arm_pool())
    missing = resolve(_policy(target), _benchmark(unreachable), _arm_pool())

    assert time.perf_counter() - start < 5.0
    assert found is not None and len(found.action) == 8
    assert missing is None


class _ClipPositions(ActionAdapter):
    """Clips an arm's position to the unit range, as a policy's action head might."""

    from_spec: ClassVar[type[BaseModel]] = EEActionSpace
    to_spec: ClassVar[type[BaseModel]] = EEActionSpace
    lossless: ClassVar[bool] = False

    def applies(self, source: BaseModel) -> bool:
        return isinstance(source, EEActionSpace)

    def produce(self, source: BaseModel) -> BaseModel:
        return source

    def adapt(self, values: Any, *, source: BaseModel) -> list[float]:
        out = [float(v) for v in values]
        return [*(min(1.0, max(-1.0, v)) for v in out[:3]), *out[3:]]


def test_a_lifted_adapter_on_a_later_group_sees_probe_values_a_bare_arm_would() -> None:
    two_arms = CompositeActionSpace(groups=(_ARM, _ARM))

    report = verify(
        _policy(two_arms),
        _benchmark(two_arms),
        Pipeline(action=[GroupAdapter(1, _ClipPositions())]),
    )

    assert report.ok, report.reasons


def test_a_group_adapter_refuses_a_negative_index() -> None:
    with pytest.raises(ValueError, match="0 or more"):
        GroupAdapter(-1, RotationFormatAdapter(RotationFormat.AXIS_ANGLE))


def test_a_group_adapter_takes_a_composite_whose_groups_are_a_list() -> None:
    quaternion = _ARM.model_copy(update={"rotation": RotationFormat.QUATERNION})
    listed = _RC365.model_copy(update={"groups": [quaternion, *_RC365.groups[1:]]})
    pipeline = Pipeline(action=[GroupAdapter(0, RotationFormatAdapter(RotationFormat.AXIS_ANGLE))])

    assert check_compatibility(_policy(listed), _benchmark(_RC365), pipeline=pipeline).ok
    assert verify(_policy(listed), _benchmark(_RC365), pipeline).ok


def test_a_group_adapter_refuses_a_stateful_adapter() -> None:
    class Stateful(_Rewrite):
        state_key: ClassVar[str | None] = "history"

    with pytest.raises(ValueError, match="stateless"):
        GroupAdapter(0, Stateful(list))


def test_a_group_adapter_refuses_an_adapter_whose_output_no_group_holds() -> None:
    with pytest.raises(ValueError, match="keeps an arm, torso or base spec"):
        GroupAdapter(0, BasePinWiden())


def test_a_lifted_adapter_is_as_lossless_as_the_adapter_it_wraps() -> None:
    lossy = GroupAdapter(0, GripperThresholdAdapter(GripperFormat.SIGNED_OPEN_LOW))

    assert lossy.lossless is False
    assert GroupAdapter(0, RotationFormatAdapter(RotationFormat.AXIS_ANGLE)).lossless is True
    assert GroupAdapter.lossless is False


def test_a_closed_probe_names_each_arm_group_by_the_end_it_sent() -> None:
    arm = _RC365.groups[0]
    two_arms = CompositeActionSpace(groups=(arm, arm))

    report = verify(_policy(two_arms), _benchmark(two_arms), Pipeline())

    assert report.ok, report.reasons
    assert _check(report, "action.ee0.gripper.closed").passed
    assert _check(report, "action.ee1.gripper.open").passed
    assert not any(c.name == "action.ee1.gripper.closed" for c in report.checks)


def test_many_body_values_stay_inside_a_unit_clip() -> None:
    many = BodyActionSpace(part=BodyPart.TORSO, dof=40, command=Command.DELTA)
    action = CompositeActionSpace(groups=(_RC365.groups[0], _BASE, many))

    values = _probe_composite(action, openness=1.0)
    body = values[7:]

    assert all(-1.0 < v < 1.0 for v in body)
    assert len(set(values)) == len(values)
