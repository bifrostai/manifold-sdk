"""The base pose and torso a mobile robot reports: spec, compatibility, verify probe."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError
from scipy.spatial.transform import Rotation

from manifold.adapters.observation import (
    DynamicFrameRebaseAdapter,
    FrameRebaseAdapter,
    ProprioRotationAdapter,
)
from manifold.adapters.packing import PackToNativeLayout, UnpackFromNativeLayout
from manifold.core import (
    BasePoseObservationSpec,
    Benchmark,
    Camera,
    EEActionSpace,
    EEObservationSpec,
    Embodiment,
    Frame,
    GripperFormat,
    GripperObservationSpec,
    JointObservationSpec,
    ObservationAdapter,
    ObservationSpace,
    Pipeline,
    PolicySignature,
    Proprioception,
    RotationFormat,
    check_compatibility,
    verify,
)
from manifold.core.names import EmbodimentName
from manifold.core.native_layout import (
    BatchAxis,
    LayoutEntry,
    NativeLayout,
    Slice,
    SourceKind,
)
from manifold.core.values import Observation
from manifold.core.verify import probe_observation
from manifold.lib.rotation import convert, geodesic_angle
from manifold.recipes.local import _default_pool
from manifold.recipes.resolve import resolve
from manifold.sensors import CameraName

_CAMERA = Camera(name=CameraName.AGENTVIEW, shape=(8, 8, 3))
_ACTION = EEActionSpace(
    rotation=RotationFormat.AXIS_ANGLE, gripper=GripperFormat.SIGNED_OPEN_LOW, delta=True
)


def _proprio(
    rotation: RotationFormat = RotationFormat.QUATERNION, frame: Frame = Frame.WORLD
) -> Proprioception:
    return Proprioception(
        ee_pose=EEObservationSpec(
            rotation=RotationFormat.QUATERNION,
            gripper=GripperObservationSpec(dim=2),
            frame=Frame.BASE,
        ),
        base_pose=BasePoseObservationSpec(rotation=rotation, frame=frame),
    )


def _benchmark(proprio: Proprioception | None = None) -> Benchmark:
    return Benchmark(
        name="suite",
        embodiment=Embodiment(
            name=EmbodimentName.PANDA_OMRON_WHOLE_BODY,
            action=_ACTION,
            proprioception=proprio or _proprio(),
        ),
        sensors=[_CAMERA],
        instruction=False,
    )


def _policy(proprio: Proprioception | None = None) -> PolicySignature:
    return PolicySignature(
        action_space=_ACTION,
        proprioception=proprio or _proprio(),
        cameras=[_CAMERA],
        instruction=False,
    )


def _check(report: Any, name: str):  # type: ignore[no-untyped-def]
    matches = [c for c in report.checks if c.name == name]
    assert matches, f"no check named {name!r}; have {[c.name for c in report.checks]}"
    return matches[0]


class _ScalarFirstMisread(ProprioRotationAdapter):
    """A bug: reads the scalar-last quaternion as scalar-first before converting."""

    def _encode_rotation(self, rot_block: list[float], source: RotationFormat) -> list[float]:
        x, y, z, w = rot_block
        return convert([w, x, y, z], source, self.target)


class _EEPoseAsBasePose(ObservationAdapter):
    """A bug: copies the end-effector pose into base_pose."""

    from_spec = ObservationSpace
    to_spec = ObservationSpace
    lossless = True

    def applies(self, source: BaseModel) -> bool:
        return isinstance(source, ObservationSpace)

    def produce(self, source: Any) -> ObservationSpace:
        return source

    def adapt(self, observation: Any, /, *, source: Any) -> Any:
        ee = np.asarray(observation.state["ee_pose"], dtype=np.float32)
        return replace(observation, state={**observation.state, "base_pose": ee[:7].copy()})


class _BaseFromEEPose(_EEPoseAsBasePose):
    """Makes a base_pose for a benchmark that reports none, from the ee_pose."""

    def produce(self, source: Any) -> ObservationSpace:
        base = BasePoseObservationSpec(rotation=RotationFormat.QUATERNION)
        return source.with_proprioception(
            source.proprioception.model_copy(update={"base_pose": base})
        )


class _RelabelBaseFrame(ObservationAdapter):
    """Labels the base_pose as in the base frame without changing its values."""

    from_spec = ObservationSpace
    to_spec = ObservationSpace
    lossless = True

    def applies(self, source: BaseModel) -> bool:
        return isinstance(source, ObservationSpace) and source.proprioception.base_pose is not None

    def produce(self, source: Any) -> ObservationSpace:
        proprio = source.proprioception
        base = proprio.base_pose.model_copy(update={"frame": Frame.BASE})
        return source.with_proprioception(proprio.model_copy(update={"base_pose": base}))

    def adapt(self, observation: Any, /, *, source: Any) -> Any:
        return observation


def _world_ee_proprio(base_rotation: RotationFormat, base_frame: Frame) -> Proprioception:
    return Proprioception(
        ee_pose=EEObservationSpec(rotation=RotationFormat.QUATERNION, frame=Frame.WORLD),
        base_pose=BasePoseObservationSpec(rotation=base_rotation, frame=base_frame),
    )


def _rebased_ee(
    base_rotation: RotationFormat, base_rotvec: tuple[float, ...] = (0.0, 0.0, 0.7)
) -> np.ndarray:
    """Rebase a world ee_pose by a base pose given in `base_rotation`."""
    proprio = _world_ee_proprio(base_rotation, Frame.WORLD)
    space = _benchmark(proprio).observation_space
    base = [1.0, 2.0, 0.0, *convert(list(base_rotvec), RotationFormat.AXIS_ANGLE, base_rotation)]
    observation = Observation(
        state={
            "ee_pose": np.asarray([1.5, 2.0, 0.8, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "base_pose": np.asarray(base, dtype=np.float32),
        },
        sensors={},
        instruction=None,
    )
    adapter = DynamicFrameRebaseAdapter(Frame.WORLD, Frame.BASE)
    assert adapter.applies(space)
    return np.asarray(adapter.adapt(observation, source=space).state["ee_pose"])


@pytest.mark.parametrize(
    ("rotation", "width"),
    [
        (RotationFormat.AXIS_ANGLE, 6),
        (RotationFormat.EULER_XYZ, 6),
        (RotationFormat.QUATERNION, 7),
        (RotationFormat.ROTATION_6D, 9),
    ],
)
def test_a_base_pose_is_a_position_then_a_rotation(rotation: RotationFormat, width: int) -> None:
    assert BasePoseObservationSpec(rotation=rotation).expected_length() == width


def test_a_base_pose_defaults_to_the_world_frame() -> None:
    assert BasePoseObservationSpec(rotation=RotationFormat.QUATERNION).frame is Frame.WORLD


def test_an_unset_base_pose_and_torso_are_written_as_null() -> None:
    dumped = Proprioception(ee_pose=_proprio().ee_pose).model_dump(mode="json")

    assert dumped["base_pose"] is None
    assert dumped["torso_pos"] is None


def test_a_proprioception_written_without_them_still_parses() -> None:
    proprio = Proprioception(ee_pose=_proprio().ee_pose)
    written = proprio.model_dump(mode="json")
    del written["base_pose"], written["torso_pos"]

    assert Proprioception.model_validate(written) == proprio


def test_a_policy_that_needs_a_base_pose_is_refused_without_one() -> None:
    bench = _benchmark(Proprioception(ee_pose=_proprio().ee_pose))

    report = check_compatibility(_policy(), bench)

    assert report.status.value == "incompatible"
    assert any("base_pose" in reason for reason in report.reasons), report.reasons


def test_a_policy_that_needs_another_base_rotation_is_refused() -> None:
    bench = _benchmark(_proprio(RotationFormat.QUATERNION))

    report = check_compatibility(_policy(_proprio(RotationFormat.AXIS_ANGLE)), bench)

    assert report.status.value == "incompatible"


def test_the_probe_base_pose_is_known_and_unlike_the_ee_pose() -> None:
    bench = _benchmark()

    state = probe_observation(bench.observation_space).state

    base, ee = state["base_pose"], state["ee_pose"]
    assert base.shape == (7,)
    assert not np.allclose(base[:3], ee[:3])
    rotvec = convert(list(base[3:]), RotationFormat.QUATERNION, RotationFormat.AXIS_ANGLE)
    assert (
        geodesic_angle(rotvec, RotationFormat.AXIS_ANGLE, list(ee[3:7]), RotationFormat.QUATERNION)
        > 0.5
    )


def test_a_matching_base_pose_pairing_passes() -> None:
    report = verify(_policy(), _benchmark(), Pipeline())

    assert report.ok, report.reasons
    assert _check(report, "observation.base_pose.position").passed
    assert _check(report, "observation.base_pose.rotation").passed


def test_a_base_pose_rotation_re_encode_passes() -> None:
    policy = _policy(_proprio(RotationFormat.AXIS_ANGLE))
    pipeline = Pipeline(
        observation=[ProprioRotationAdapter(RotationFormat.AXIS_ANGLE, channel="base_pose")]
    )

    report = verify(policy, _benchmark(), pipeline)

    assert report.ok, report.reasons
    assert _check(report, "observation.base_pose.rotation").passed


def test_the_rotation_adapter_bridges_a_base_pose_format_mismatch() -> None:
    policy = _policy(_proprio(RotationFormat.EULER_XYZ))
    pipeline = Pipeline(
        observation=[ProprioRotationAdapter(RotationFormat.EULER_XYZ, channel="base_pose")]
    )

    report = check_compatibility(policy, _benchmark(), pipeline=pipeline)

    assert report.status.value == "compatible_via_pipeline", report.reasons


def test_the_default_pool_resolves_a_base_pose_format_mismatch() -> None:
    policy, bench = _policy(_proprio(RotationFormat.AXIS_ANGLE)), _benchmark()

    pool = _default_pool(policy, bench)
    pipeline = resolve(policy, bench, pool)

    assert len(pool) == 1
    assert isinstance(pool[0], ProprioRotationAdapter) and pool[0].channel == "base_pose"
    assert pipeline is not None
    assert verify(policy, bench, pipeline).ok


def test_the_rotation_adapter_leaves_the_base_position() -> None:
    adapter = ProprioRotationAdapter(RotationFormat.AXIS_ANGLE, channel="base_pose")
    space = _benchmark().observation_space
    observation = probe_observation(space)

    out = adapter.adapt(observation, source=space).state["base_pose"]

    assert np.allclose(out[:3], observation.state["base_pose"][:3])
    assert out.shape == (6,)


def test_verify_catches_a_base_quaternion_read_scalar_first() -> None:
    policy = _policy(_proprio(RotationFormat.AXIS_ANGLE))
    pipeline = Pipeline(
        observation=[_ScalarFirstMisread(RotationFormat.AXIS_ANGLE, channel="base_pose")]
    )

    report = verify(policy, _benchmark(), pipeline)

    assert not report.ok
    assert _check(report, "observation.base_pose.position").passed
    assert not _check(report, "observation.base_pose.rotation").passed


def test_verify_catches_the_ee_pose_fed_as_the_base_pose() -> None:
    report = verify(_policy(), _benchmark(), Pipeline(observation=[_EEPoseAsBasePose()]))

    assert not report.ok
    assert not _check(report, "observation.base_pose.position").passed
    assert not _check(report, "observation.base_pose.rotation").passed


def test_a_base_pose_frame_change_is_not_checked() -> None:
    policy = _policy(_proprio(frame=Frame.BASE))
    report = verify(policy, _benchmark(), Pipeline(observation=[_RelabelBaseFrame()]))

    assert report.ok, report.reasons
    assert not any(c.name.startswith("observation.base_pose") for c in report.checks)
    assert any("base_pose" in reason and "frame" in reason for reason in report.not_checked)


def test_a_base_pose_the_pipeline_makes_is_not_checked() -> None:
    bench = _benchmark(Proprioception(ee_pose=_proprio().ee_pose))

    report = verify(_policy(), bench, Pipeline(observation=[_BaseFromEEPose()]))

    assert report.ok, report.reasons
    assert not any(c.name.startswith("observation.base_pose") for c in report.checks)
    assert any("benchmark reports none" in reason for reason in report.not_checked)


@pytest.mark.parametrize(
    "rotation",
    [RotationFormat.AXIS_ANGLE, RotationFormat.EULER_XYZ, RotationFormat.ROTATION_6D],
)
def test_the_dynamic_rebase_reads_a_declared_base_rotation(rotation: RotationFormat) -> None:
    tilted = (0.3, -0.2, 0.7)
    expected = _rebased_ee(RotationFormat.QUATERNION, tilted)
    assert np.allclose(_rebased_ee(rotation, tilted), expected, atol=1e-5)


def test_the_dynamic_rebase_moves_the_ee_into_the_base_frame() -> None:
    rebased = _rebased_ee(RotationFormat.QUATERNION)

    # The ee sits 0.5 m along world x from a base at (1, 2) turned 0.7 rad about z.
    assert np.allclose(rebased[:3], [0.5 * np.cos(0.7), -0.5 * np.sin(0.7), 0.8], atol=1e-5)


def test_the_dynamic_rebase_needs_the_base_pose_in_its_source_frame() -> None:
    space = _benchmark(_world_ee_proprio(RotationFormat.QUATERNION, Frame.BASE)).observation_space

    assert not DynamicFrameRebaseAdapter(Frame.WORLD, Frame.BASE).applies(space)


def test_a_layout_reading_the_base_pose_is_packed_from_the_spec() -> None:
    def state(key: str, start: int, stop: int) -> LayoutEntry:
        return LayoutEntry(
            key=key,
            source=SourceKind.STATE,
            source_name="base_pose",
            ops=(Slice(start=start, stop=stop), BatchAxis(batch=True, time=True)),
        )

    layout = NativeLayout(
        entries=(state("state.base_position", 0, 3), state("state.base_rotation", 3, 7))
    )
    output = NativeLayout(
        entries=tuple(
            LayoutEntry(
                key=k, source=SourceKind.STATE, source_name=k, ops=(Slice(start=0, stop=1),)
            )
            for k in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
        )
    )
    pipeline = Pipeline(
        pack=PackToNativeLayout(layout), unpack=UnpackFromNativeLayout(output, _ACTION)
    )

    report = verify(_policy(), _benchmark(), pipeline)

    packing = _check(report, "packing.observation")
    assert packing.passed, packing.detail
    assert "2 contract-modeled native key" in packing.detail
    assert not any("base_pose" in reason for reason in report.not_checked), report.not_checked


def _with_torso(proprio: Proprioception | None = None, dof: int = 1) -> Proprioception:
    return (proprio or _proprio()).model_copy(update={"torso_pos": JointObservationSpec(dof=dof)})


class _ZeroTorso(ObservationAdapter):
    """A bug: zeroes the torso joints."""

    from_spec = ObservationSpace
    to_spec = ObservationSpace
    lossless = True

    def applies(self, source: BaseModel) -> bool:
        return isinstance(source, ObservationSpace)

    def produce(self, source: Any) -> ObservationSpace:
        return source

    def adapt(self, observation: Any, /, *, source: Any) -> Any:
        torso = np.zeros_like(observation.state["torso_pos"])
        return replace(observation, state={**observation.state, "torso_pos": torso})


def test_a_policy_that_needs_the_torso_is_refused_without_it() -> None:
    report = check_compatibility(_policy(_with_torso()), _benchmark())

    assert report.status.value == "incompatible"
    assert any("torso_pos" in reason for reason in report.reasons), report.reasons


def test_the_probe_torso_is_distinct_per_joint() -> None:
    state = probe_observation(_benchmark(_with_torso(dof=4)).observation_space).state

    torso = state["torso_pos"]
    assert torso.shape == (4,)
    assert len(set(torso.tolist())) == 4
    assert not set(torso.tolist()) & set(state["base_pose"].tolist())


def test_a_matching_torso_pairing_passes() -> None:
    proprio = _with_torso(dof=4)

    report = verify(_policy(proprio), _benchmark(proprio), Pipeline())

    assert report.ok, report.reasons
    assert _check(report, "observation.torso_pos").passed


def test_verify_catches_a_zeroed_torso() -> None:
    proprio = _with_torso()

    report = verify(_policy(proprio), _benchmark(proprio), Pipeline(observation=[_ZeroTorso()]))

    assert not report.ok
    assert "values changed" in _check(report, "observation.torso_pos").detail
    assert _check(report, "observation.base_pose.position").passed


def test_verify_catches_a_torso_shifted_in_the_arrays_it_is_handed() -> None:
    class ShiftInPlace(_ZeroTorso):
        def adapt(self, observation: Any, /, *, source: Any) -> Any:
            observation.state["torso_pos"][:] += 0.5
            return observation

    proprio = _with_torso()

    report = verify(_policy(proprio), _benchmark(proprio), Pipeline(observation=[ShiftInPlace()]))

    assert not _check(report, "observation.torso_pos").passed


def test_a_torso_the_chain_narrows_is_not_checked() -> None:
    class KeepFirstJoint(_ZeroTorso):
        def produce(self, source: Any) -> ObservationSpace:
            return source.with_proprioception(_with_torso(source.proprioception, dof=1))

        def adapt(self, observation: Any, /, *, source: Any) -> Any:
            first = observation.state["torso_pos"][:1].copy()
            return replace(observation, state={**observation.state, "torso_pos": first})

    pipeline = Pipeline(observation=[KeepFirstJoint()])

    report = verify(_policy(_with_torso(dof=1)), _benchmark(_with_torso(dof=2)), pipeline)

    assert report.ok, report.reasons
    assert any(
        "torso_pos" in reason and "no ground truth" in reason for reason in report.not_checked
    )


def test_a_torso_the_benchmark_does_not_report_is_not_checked() -> None:
    class AddTorso(_ZeroTorso):
        def produce(self, source: Any) -> ObservationSpace:
            return source.with_proprioception(_with_torso(source.proprioception))

        def adapt(self, observation: Any, /, *, source: Any) -> Any:
            torso = np.zeros(1, dtype=np.float32)
            return replace(observation, state={**observation.state, "torso_pos": torso})

    report = verify(_policy(_with_torso()), _benchmark(), Pipeline(observation=[AddTorso()]))

    assert report.ok, report.reasons
    assert not any(c.name == "observation.torso_pos" for c in report.checks)
    assert any("torso_pos" in reason for reason in report.not_checked)


@pytest.mark.parametrize(
    "torso",
    [
        JointObservationSpec(dof=1, gripper=GripperFormat.UNSIGNED),
        JointObservationSpec(dof=2, arm_count=2),
    ],
)
def test_a_torso_holds_joints_only(torso: JointObservationSpec) -> None:
    with pytest.raises(ValidationError, match="torso_pos holds the torso joints only"):
        Proprioception(torso_pos=torso)


def test_the_rotation_adapter_still_re_encodes_both_ee_arms() -> None:
    ee = EEObservationSpec(
        rotation=RotationFormat.QUATERNION, gripper=GripperObservationSpec(dim=2), arm_count=2
    )
    space = _benchmark(Proprioception(ee_pose=ee)).observation_space
    observation = probe_observation(space)

    out = ProprioRotationAdapter(RotationFormat.AXIS_ANGLE).adapt(observation, source=space)

    values = out.state["ee_pose"]
    assert values.shape == (16,)
    assert np.allclose(values[3:6], values[11:14])
    assert np.allclose(values[:3] + 1.0, values[8:11])


def test_a_declared_base_pose_of_another_length_is_an_error() -> None:
    proprio = _world_ee_proprio(RotationFormat.QUATERNION, Frame.WORLD)
    space = _benchmark(proprio).observation_space
    observation = Observation(
        state={
            "ee_pose": np.asarray([1.5, 2.0, 0.8, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "base_pose": np.eye(4, dtype=np.float32).reshape(-1),
        },
        sensors={},
        instruction=None,
    )

    with pytest.raises(ValueError, match="declared as 7 floats"):
        DynamicFrameRebaseAdapter(Frame.WORLD, Frame.BASE).adapt(observation, source=space)


def test_an_undeclared_channel_keeps_the_matrix_form() -> None:
    proprio = _world_ee_proprio(RotationFormat.AXIS_ANGLE, Frame.BASE)
    space = _benchmark(proprio).observation_space
    base = np.eye(4)
    base[:3, 3] = [1.0, 2.0, 0.0]
    observation = Observation(
        state={
            "ee_pose": np.asarray([1.5, 2.0, 0.8, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "base_matrix": base.reshape(-1).astype(np.float32),
        },
        sensors={},
        instruction=None,
    )
    adapter = DynamicFrameRebaseAdapter(Frame.WORLD, Frame.BASE, base_pose_channel="base_matrix")

    assert adapter.applies(space)
    rebased = adapter.adapt(observation, source=space).state["ee_pose"]
    assert np.allclose(rebased[:3], [0.5, 0.0, 0.8], atol=1e-6)


class _TranslateOnly(DynamicFrameRebaseAdapter):
    """A bug: moves the ee_pose by the base position but ignores the base rotation."""

    def adapt(self, observation: Any, *, source: Any) -> Any:
        ee = np.asarray(observation.state["ee_pose"], dtype=np.float32).copy()
        ee[:3] -= np.asarray(observation.state["base_pose"][:3], dtype=np.float32)
        return replace(observation, state={**observation.state, "ee_pose": ee})


def _rebase_pairing() -> tuple[PolicySignature, Benchmark]:
    bench = _benchmark(_world_ee_proprio(RotationFormat.QUATERNION, Frame.WORLD))
    policy = _policy(
        Proprioception(
            ee_pose=EEObservationSpec(rotation=RotationFormat.QUATERNION, frame=Frame.BASE)
        )
    )
    return policy, bench


def test_verify_checks_a_rebase_into_the_base_frame() -> None:
    policy, bench = _rebase_pairing()

    report = verify(
        policy, bench, Pipeline(observation=[DynamicFrameRebaseAdapter(Frame.WORLD, Frame.BASE)])
    )

    assert report.ok, report.reasons
    assert _check(report, "observation.ee_pose.position").passed
    assert _check(report, "observation.ee_pose.rotation").passed


class _RebaseInside(ObservationAdapter):
    """A rebase by the base pose that is not itself a DynamicFrameRebaseAdapter."""

    from_spec: ClassVar[type[BaseModel]] = ObservationSpace
    to_spec: ClassVar[type[BaseModel]] = ObservationSpace
    lossless: ClassVar[bool] = True

    def __init__(self) -> None:
        self.inner = DynamicFrameRebaseAdapter(Frame.WORLD, Frame.BASE)

    def applies(self, source: BaseModel) -> bool:
        return self.inner.applies(source)

    def produce(self, source: BaseModel) -> BaseModel:
        return self.inner.produce(source)

    def adapt(self, observation: Any, *, source: BaseModel) -> Any:
        return self.inner.adapt(observation, source=source)


def test_verify_checks_a_rebase_by_any_adapter_that_reads_the_base_pose() -> None:
    policy, bench = _rebase_pairing()

    report = verify(policy, bench, Pipeline(observation=[_RebaseInside()]))

    assert report.ok, report.reasons
    assert _check(report, "observation.ee_pose.position").passed
    assert _check(report, "observation.ee_pose.rotation").passed


def test_verify_catches_a_rebase_that_ignores_the_base_rotation() -> None:
    policy, bench = _rebase_pairing()

    report = verify(policy, bench, Pipeline(observation=[_TranslateOnly(Frame.WORLD, Frame.BASE)]))

    assert not report.ok
    assert not _check(report, "observation.ee_pose.position").passed
    assert not _check(report, "observation.ee_pose.rotation").passed


class _PlaceByBase(ObservationAdapter):
    """Moves a base-frame ee_pose into the world by the base pose; `translate_only` is a bug."""

    from_spec: ClassVar[type[BaseModel]] = ObservationSpace
    to_spec: ClassVar[type[BaseModel]] = ObservationSpace
    lossless: ClassVar[bool] = True

    def __init__(self, *, translate_only: bool = False) -> None:
        self.translate_only = translate_only

    def applies(self, source: BaseModel) -> bool:
        return isinstance(source, ObservationSpace)

    def produce(self, source: Any) -> ObservationSpace:
        ee = source.proprioception.ee_pose.model_copy(update={"frame": Frame.WORLD})
        return source.with_proprioception(source.proprioception.model_copy(update={"ee_pose": ee}))

    def adapt(self, observation: Any, *, source: Any) -> Any:
        ee = np.asarray(observation.state["ee_pose"], dtype=np.float64)
        base = np.asarray(observation.state["base_pose"], dtype=np.float64)
        turn = Rotation.from_quat(base[3:7])
        if self.translate_only:
            placed = [*(ee[:3] + base[:3]), *ee[3:7]]
        else:
            placed = [
                *(turn.apply(ee[:3]) + base[:3]),
                *(turn * Rotation.from_quat(ee[3:7])).as_quat(),
            ]
        return replace(observation, state={**observation.state, "ee_pose": np.asarray(placed)})


@pytest.mark.parametrize(("translate_only", "ok"), [(False, True), (True, False)])
def test_verify_checks_a_rebase_out_of_the_base_frame(translate_only: bool, ok: bool) -> None:
    base_frame = Proprioception(
        ee_pose=EEObservationSpec(rotation=RotationFormat.QUATERNION, frame=Frame.BASE),
        base_pose=BasePoseObservationSpec(rotation=RotationFormat.QUATERNION),
    )
    world = Proprioception(
        ee_pose=EEObservationSpec(rotation=RotationFormat.QUATERNION, frame=Frame.WORLD)
    )
    pipeline = Pipeline(observation=[_PlaceByBase(translate_only=translate_only)])

    report = verify(_policy(world), _benchmark(base_frame), pipeline)

    assert report.ok is ok, report.reasons
    assert _check(report, "observation.ee_pose.rotation").passed is ok


def test_the_rotation_adapter_refuses_a_channel_without_a_rotation() -> None:
    with pytest.raises(ValueError, match="not 'torso_pos'"):
        ProprioRotationAdapter(RotationFormat.AXIS_ANGLE, channel="torso_pos")


def test_the_pose_fields_are_the_fields_with_a_rotation() -> None:
    assert Proprioception.pose_fields() == ("ee_pose", "base_pose")


def test_a_static_rebase_keeps_the_static_rule_beside_a_base_pose() -> None:
    policy, bench = _rebase_pairing()
    pipeline = Pipeline(observation=[FrameRebaseAdapter(Frame.WORLD, Frame.BASE, np.eye(3))])

    report = verify(policy, bench, pipeline)

    assert report.ok, report.reasons
    assert _check(report, "observation.ee_pose.position").passed
    assert not any(c.name == "observation.ee_pose.rotation" for c in report.checks)
    assert any("ee_pose.rotation" in reason for reason in report.not_checked)


def test_a_chain_verify_cannot_reproduce_is_not_checked() -> None:
    _, bench = _rebase_pairing()
    bridge = Proprioception(
        ee_pose=EEObservationSpec(rotation=RotationFormat.QUATERNION, frame=Frame.WIDOWX_BRIDGE_EE)
    )
    pipeline = Pipeline(
        observation=[
            DynamicFrameRebaseAdapter(Frame.WORLD, Frame.BASE),
            FrameRebaseAdapter(Frame.BASE, Frame.WIDOWX_BRIDGE_EE, np.eye(3)),
        ]
    )

    report = verify(_policy(bridge), bench, pipeline)

    assert report.ok, report.reasons
    assert not any(c.name.startswith("observation.ee_pose") for c in report.checks)
    assert any("cannot reproduce" in reason for reason in report.not_checked)
