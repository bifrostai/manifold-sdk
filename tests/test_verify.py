from dataclasses import replace
from typing import Any

import numpy as np

from manifold.adapters.action import (
    GripperPolarityAdapter,
    RotationFormatAdapter,
    UnifiedSliceAdapter,
)
from manifold.adapters.observation import FrameRebaseAdapter, ProprioRotationAdapter
from manifold.core import (
    ActionAdapter,
    Benchmark,
    Camera,
    CameraCalibration,
    CameraIntrinsics,
    EEActionSpace,
    EEObservationSpec,
    Embodiment,
    Frame,
    GripperFormat,
    GripperObservationSpec,
    JointActionSpace,
    JointObservationSpec,
    ObservationAdapter,
    ObservationSpace,
    Pipeline,
    PolicySignature,
    Proprioception,
    RotationFormat,
    UnifiedActionSpace,
    verify,
)
from manifold.lib.gripper import from_openness, openness
from manifold.lib.rotation import convert


def _ee(
    rotation: RotationFormat = RotationFormat.AXIS_ANGLE,
    gripper: GripperFormat = GripperFormat.SIGNED,
) -> EEActionSpace:
    return EEActionSpace(rotation=rotation, gripper=gripper, delta=True)


def _franka_proprio(rotation: RotationFormat = RotationFormat.AXIS_ANGLE) -> Proprioception:
    return Proprioception(
        ee_pose=EEObservationSpec(rotation=rotation, gripper=GripperObservationSpec(dim=2))
    )


def _benchmark(
    rotation: RotationFormat = RotationFormat.AXIS_ANGLE,
    gripper: GripperFormat = GripperFormat.SIGNED,
) -> Benchmark:
    embodiment = Embodiment(
        name="arm",
        action=_ee(rotation, gripper),
        proprioception=_franka_proprio(rotation),
    )
    return Benchmark(
        name="suite",
        embodiment=embodiment,
        sensors=[Camera(name="agentview", shape=(8, 8, 3))],
        instruction=False,
    )


def _policy(
    action_space: Any, *, proprio: Proprioception, rotation: RotationFormat
) -> PolicySignature:
    return PolicySignature(
        action_space=action_space,
        proprioception=proprio,
        cameras=[Camera(name="agentview", shape=(8, 8, 3))],
        instruction=False,
    )


def _check(report: Any, name: str):  # type: ignore[no-untyped-def]
    matches = [c for c in report.checks if c.name == name]
    assert matches, f"no check named {name!r}; have {[c.name for c in report.checks]}"
    return matches[0]


def test_correct_rotation_format_pipeline_passes_the_rotation_check() -> None:
    # Benchmark requires QUATERNION; policy emits AXIS_ANGLE; the RotationFormatAdapter
    # bridges. The rotation meaning must survive — a passed rotation check.
    bench = _benchmark(rotation=RotationFormat.QUATERNION)
    policy = _policy(
        _ee(rotation=RotationFormat.AXIS_ANGLE),
        proprio=_franka_proprio(RotationFormat.QUATERNION),
        rotation=RotationFormat.QUATERNION,
    )
    pipeline = Pipeline(action=[RotationFormatAdapter(target=RotationFormat.QUATERNION)])

    report = verify(policy, bench, pipeline)

    assert report.ok, report.reasons
    assert _check(report, "action.rotation").passed


def test_buggy_rotation_adapter_is_caught_by_asymmetric_probe() -> None:
    # A throwaway adapter that re-encodes the rotation but mangles it (swaps two
    # quaternion elements). Zeros would round-trip; the asymmetric probe
    # plus the geodesic comparison catches the reorder.
    class ManglingRotationAdapter(ActionAdapter):
        from_spec = EEActionSpace
        to_spec = EEActionSpace
        lossless = True

        def applies(self, source: Any) -> bool:
            return isinstance(source, EEActionSpace)

        def produce(self, source: Any) -> EEActionSpace:
            return source.model_copy(update={"rotation": RotationFormat.QUATERNION})

        def adapt(self, values: Any, *, source: Any) -> list[float]:
            quat = convert(
                [values[3], values[4], values[5]], source.rotation, RotationFormat.QUATERNION
            )
            # Swap qx and qy: same length, a different (wrong) rotation.
            quat[0], quat[1] = quat[1], quat[0]
            return [values[0], values[1], values[2], *quat, values[6]]

    bench = _benchmark(rotation=RotationFormat.QUATERNION)
    policy = _policy(
        _ee(rotation=RotationFormat.AXIS_ANGLE),
        proprio=_franka_proprio(RotationFormat.QUATERNION),
        rotation=RotationFormat.QUATERNION,
    )
    pipeline = Pipeline(action=[ManglingRotationAdapter()])

    report = verify(policy, bench, pipeline)

    assert not report.ok
    assert not _check(report, "action.rotation").passed


def test_buggy_gripper_adapter_is_caught_by_asymmetric_probe() -> None:
    # An adapter that produces the correct target spec but inverts the gripper:
    # an open command comes out closed. The openness comparison catches it; a
    # zeros probe (openness ~0.5 for SIGNED) would be far less revealing.
    class InvertingGripperAdapter(ActionAdapter):
        from_spec = EEActionSpace
        to_spec = EEActionSpace
        lossless = True

        def applies(self, source: Any) -> bool:
            return isinstance(source, EEActionSpace)

        def produce(self, source: Any) -> EEActionSpace:
            return source.model_copy(update={"gripper": GripperFormat.SIGNED_OPEN_LOW})

        def adapt(self, values: Any, *, source: Any) -> list[float]:
            # Pass the raw gripper value straight through but relabel the format:
            # the value that meant "open" under SIGNED now means "closed" under
            # SIGNED_OPEN_LOW. A silent polarity inversion.
            return list(values)

    bench = _benchmark(gripper=GripperFormat.SIGNED_OPEN_LOW)
    policy = _policy(
        _ee(gripper=GripperFormat.SIGNED),
        proprio=_franka_proprio(),
        rotation=RotationFormat.AXIS_ANGLE,
    )
    pipeline = Pipeline(action=[InvertingGripperAdapter()])

    report = verify(policy, bench, pipeline)

    assert not report.ok
    assert not _check(report, "action.gripper").passed


def test_correct_gripper_polarity_pipeline_passes() -> None:
    bench = _benchmark(gripper=GripperFormat.SIGNED_OPEN_LOW)
    policy = _policy(
        _ee(gripper=GripperFormat.SIGNED),
        proprio=_franka_proprio(),
        rotation=RotationFormat.AXIS_ANGLE,
    )
    pipeline = Pipeline(action=[GripperPolarityAdapter(target=GripperFormat.SIGNED_OPEN_LOW)])

    report = verify(policy, bench, pipeline)

    assert report.ok, report.reasons
    assert _check(report, "action.gripper").passed


def test_manifest_lists_camera_normalization_and_env_under_not_checked() -> None:
    bench = _benchmark()
    policy = _policy(_ee(), proprio=_franka_proprio(), rotation=RotationFormat.AXIS_ANGLE)

    report = verify(policy, bench, Pipeline())

    joined = " | ".join(report.not_checked)
    assert "agentview" in joined  # camera orientation/channel content
    assert "normalization" in joined
    assert "env/policy behaviour" in joined
    # Shape is checked, not in not_checked.
    assert _check(report, "observation.camera.agentview.shape").passed


def test_quaternion_double_cover_still_passes() -> None:
    # An adapter that negates every quaternion element: q and -q are the same rotation,
    # so a representation-aware check must not flag it (no false negative).
    class NegatingQuaternionAdapter(ActionAdapter):
        from_spec = EEActionSpace
        to_spec = EEActionSpace
        lossless = True

        def applies(self, source: Any) -> bool:
            return isinstance(source, EEActionSpace)

        def produce(self, source: Any) -> EEActionSpace:
            return source.model_copy(update={"rotation": RotationFormat.QUATERNION})

        def adapt(self, values: Any, *, source: Any) -> list[float]:
            quat = convert(
                [values[3], values[4], values[5]], source.rotation, RotationFormat.QUATERNION
            )
            negated = [-q for q in quat]  # q and -q are the same rotation.
            return [values[0], values[1], values[2], *negated, values[6]]

    bench = _benchmark(rotation=RotationFormat.QUATERNION)
    policy = _policy(
        _ee(rotation=RotationFormat.AXIS_ANGLE),
        proprio=_franka_proprio(RotationFormat.QUATERNION),
        rotation=RotationFormat.QUATERNION,
    )
    pipeline = Pipeline(action=[NegatingQuaternionAdapter()])

    report = verify(policy, bench, pipeline)

    assert report.ok, report.reasons
    assert _check(report, "action.rotation").passed


def test_clean_catalog_match_verifies_ok() -> None:
    from manifold.benchmarks.libero import LIBERO

    obs = LIBERO.observation_space
    policy = PolicySignature(
        action_space=LIBERO.embodiment.action,
        proprioception=obs.proprioception,
        cameras=list(obs.cameras),
        instruction=obs.instruction,
    )
    report = verify(policy, LIBERO, Pipeline())
    assert report.ok, report.reasons


def test_static_failure_short_circuits_with_static_reasons() -> None:
    # The policy demands a camera the benchmark never publishes: a static
    # INCOMPATIBLE. verify must surface the static reasons as failed checks and run
    # no dynamic probes.
    bench = _benchmark()
    policy = PolicySignature(
        action_space=_ee(),
        proprioception=_franka_proprio(),
        cameras=[Camera(name="nonexistent", shape=(8, 8, 3))],
        instruction=False,
    )

    report = verify(policy, bench, Pipeline())

    assert not report.ok
    assert not report.static_passed
    assert all(c.name == "static" for c in report.checks)
    assert report.reasons  # the static mismatch reasons
    assert report.not_checked == ()  # no dynamic probes ran


def test_proprio_rotation_pipeline_passes_the_observation_rotation_check() -> None:
    # Benchmark reports QUATERNION ee_pose; policy consumes AXIS_ANGLE; the
    # ProprioRotationAdapter bridges. The observation-side rotation meaning holds.
    bench = _benchmark(rotation=RotationFormat.QUATERNION)
    policy = _policy(
        _ee(rotation=RotationFormat.QUATERNION),  # action side already matches
        proprio=_franka_proprio(RotationFormat.AXIS_ANGLE),
        rotation=RotationFormat.AXIS_ANGLE,
    )
    pipeline = Pipeline(observation=[ProprioRotationAdapter(target=RotationFormat.AXIS_ANGLE)])

    report = verify(policy, bench, pipeline)

    assert report.ok, report.reasons
    assert _check(report, "observation.ee_pose.rotation").passed


def test_frame_rebase_routes_ee_rotation_to_not_checked_but_keeps_position() -> None:
    # A FrameRebaseAdapter changes the ee_pose frame (BASE -> WIDOWX_BRIDGE_EE) by a
    # static reorientation with no coordinate-free ground truth. verify must not
    # assert the rebase as meaning-preserving: the rotation goes to not_checked, the
    # position is still checked (and passes), and the run stays ok.
    rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
    base_proprio = Proprioception(
        ee_pose=EEObservationSpec(
            rotation=RotationFormat.QUATERNION,
            gripper=GripperObservationSpec(dim=2),
            frame=Frame.BASE,
        )
    )
    rebased_proprio = Proprioception(
        ee_pose=EEObservationSpec(
            rotation=RotationFormat.QUATERNION,
            gripper=GripperObservationSpec(dim=2),
            frame=Frame.WIDOWX_BRIDGE_EE,
        )
    )
    bench = Benchmark(
        name="suite",
        embodiment=Embodiment(name="arm", action=_ee(), proprioception=base_proprio),
        sensors=[Camera(name="agentview", shape=(8, 8, 3))],
        instruction=False,
    )
    policy = _policy(_ee(), proprio=rebased_proprio, rotation=RotationFormat.QUATERNION)
    pipeline = Pipeline(observation=[FrameRebaseAdapter(Frame.BASE, Frame.WIDOWX_BRIDGE_EE, rot)])

    report = verify(policy, bench, pipeline)

    assert report.ok, report.reasons
    # The position is still checked and passes.
    assert _check(report, "observation.ee_pose.position").passed
    # The rotation is not a check — it is routed to not_checked with the rebase reason.
    assert not any(c.name == "observation.ee_pose.rotation" for c in report.checks)
    assert any(
        "ee_pose.rotation" in reason and "static rebase" in reason for reason in report.not_checked
    )


def test_unified_slice_pipeline_verifies() -> None:
    bench = _benchmark()
    policy = _policy(
        UnifiedActionSpace(width=32, payload=_ee()),
        proprio=_franka_proprio(),
        rotation=RotationFormat.AXIS_ANGLE,
    )
    pipeline = Pipeline(action=[UnifiedSliceAdapter()])

    report = verify(policy, bench, pipeline)

    assert report.ok, report.reasons


def test_gradient_probe_uses_distinct_pixels() -> None:
    # Sanity: the camera probe is not a uniform fill — distinct values exercise shape.
    from manifold.core.verify import _gradient_frame

    frame = _gradient_frame((4, 4, 3), "uint8")
    assert frame.shape == (4, 4, 3)
    assert str(frame.dtype) == "uint8"
    assert len(np.unique(frame)) > 1


def test_probe_gripper_is_unambiguously_open() -> None:
    # The first probe step's gripper must decode to fully open, so an inversion is
    # maximally visible (openness 1.0 -> 0.0), not a midpoint a flip would barely move.
    from manifold.core.verify import _probe_action_values

    spec = _ee(gripper=GripperFormat.SIGNED)
    values = _probe_action_values(spec)
    assert openness(values[-1], GripperFormat.SIGNED) == 1.0


def test_always_open_gripper_adapter_is_caught_by_closed_probe() -> None:
    # An adapter that ignores its input gripper and always emits the open value.
    # The open probe alone would pass it; the dedicated closed-gripper probe (and,
    # for chunks, the alternating step) catches it.
    class AlwaysOpenGripperAdapter(ActionAdapter):
        from_spec = EEActionSpace
        to_spec = EEActionSpace
        lossless = True

        def applies(self, source: Any) -> bool:
            return isinstance(source, EEActionSpace)

        def produce(self, source: Any) -> EEActionSpace:
            return source.model_copy(update={"gripper": GripperFormat.UNSIGNED})

        def adapt(self, values: Any, *, source: Any) -> list[float]:
            out = list(values)
            out[-1] = 1.0  # UNSIGNED open, regardless of the input.
            return out

    bench = _benchmark(gripper=GripperFormat.UNSIGNED)
    policy = _policy(
        _ee(gripper=GripperFormat.SIGNED),
        proprio=_franka_proprio(),
        rotation=RotationFormat.AXIS_ANGLE,
    )
    pipeline = Pipeline(action=[AlwaysOpenGripperAdapter()])

    report = verify(policy, bench, pipeline)

    assert not report.ok
    assert not _check(report, "action.gripper.closed").passed


def test_chunked_per_step_bug_is_caught() -> None:
    # An adapter that converts step 0 correctly but zeroes the gripper of every
    # later step. A first-step-only check would miss it; checking every step with a
    # per-step-distinct probe catches it.
    class CorruptsLaterStepsAdapter(ActionAdapter):
        from_spec = EEActionSpace
        to_spec = EEActionSpace
        lossless = True

        def applies(self, source: Any) -> bool:
            return isinstance(source, EEActionSpace)

        def produce(self, source: Any) -> EEActionSpace:
            return source.model_copy(update={"gripper": GripperFormat.SIGNED_OPEN_LOW})

        def adapt(self, values: Any, *, source: Any) -> list[float]:
            per_step = source.expected_length() // source.chunk_size
            out = list(values)
            # Correctly remap step 0's gripper, but leave later steps' gripper at
            # the raw (wrongly-labelled) value — a per-step corruption.
            from manifold.lib.gripper import remap

            out[per_step - 1] = remap(
                out[per_step - 1], source.gripper, GripperFormat.SIGNED_OPEN_LOW
            )
            return out

    chunked = EEActionSpace(
        rotation=RotationFormat.AXIS_ANGLE, gripper=GripperFormat.SIGNED, delta=True, chunk_size=3
    )
    bench_chunked = EEActionSpace(
        rotation=RotationFormat.AXIS_ANGLE,
        gripper=GripperFormat.SIGNED_OPEN_LOW,
        delta=True,
        chunk_size=3,
    )
    embodiment = Embodiment(name="arm", action=bench_chunked, proprioception=_franka_proprio())
    bench = Benchmark(
        name="suite",
        embodiment=embodiment,
        sensors=[Camera(name="agentview", shape=(8, 8, 3))],
        instruction=False,
    )
    policy = _policy(chunked, proprio=_franka_proprio(), rotation=RotationFormat.AXIS_ANGLE)
    pipeline = Pipeline(action=[CorruptsLaterStepsAdapter()])

    report = verify(policy, bench, pipeline)

    assert not report.ok
    # Step 0's gripper is fine; a later step's is wrong.
    assert _check(report, "action.gripper.step0").passed
    assert not _check(report, "action.gripper.step1").passed


class RequiresCameraPoseAdapter(ObservationAdapter):
    """Reads a calibrated camera's pose off the observation, as ADR 0008 allows."""

    from_spec = ObservationSpace
    to_spec = ObservationSpace
    lossless = True

    def applies(self, source: Any) -> bool:
        return isinstance(source, ObservationSpace)

    def produce(self, source: Any) -> ObservationSpace:
        return source

    def adapt(self, observation: Any, /, *, source: Any) -> Any:
        pose = observation.extrinsics.get("agentview")
        if pose is None:
            raise RuntimeError("no pose published for 'agentview'")
        assert np.asarray(pose).shape == (4, 4)
        return observation


def test_probe_publishes_a_pose_for_each_calibrated_camera() -> None:
    # A calibrated camera means a pose in every observation, so an adapter reading
    # one must not fail on the probe alone.
    calibrated = Camera(
        name="agentview",
        shape=(8, 8, 3),
        calibration=CameraCalibration(intrinsics=CameraIntrinsics(fx=4.0, fy=4.0, cx=4.0, cy=4.0)),
    )
    embodiment = Embodiment(name="arm", action=_ee(), proprioception=_franka_proprio())
    bench = Benchmark(name="suite", embodiment=embodiment, sensors=[calibrated], instruction=False)
    policy = _policy(_ee(), proprio=_franka_proprio(), rotation=RotationFormat.AXIS_ANGLE)
    pipeline = Pipeline(observation=[RequiresCameraPoseAdapter()])

    report = verify(policy, bench, pipeline)

    assert report.ok, report.reasons


def test_probe_poses_are_distinct_and_only_for_calibrated_cameras() -> None:
    # Sanity: an uncalibrated camera publishes no pose, and two calibrated ones do
    # not share a pose — a probe that repeats itself cannot reveal a swap.
    from manifold.core.verify import _probe_extrinsics

    intrinsics = CameraIntrinsics(fx=4.0, fy=4.0, cx=4.0, cy=4.0)
    space = ObservationSpace(
        proprioception=_franka_proprio(),
        cameras=(
            Camera(name="plain", shape=(8, 8, 3)),
            Camera(
                name="agentview",
                shape=(8, 8, 3),
                calibration=CameraCalibration(intrinsics=intrinsics),
            ),
            Camera(
                name="wrist",
                shape=(8, 8, 3),
                calibration=CameraCalibration(intrinsics=intrinsics),
            ),
        ),
        instruction=False,
    )

    poses = _probe_extrinsics(space)

    assert set(poses) == {"agentview", "wrist"}
    assert all(pose.shape == (4, 4) and pose.dtype == np.float64 for pose in poses.values())
    assert not np.array_equal(poses["agentview"], poses["wrist"])


def _bimanual_benchmark(
    rotation: RotationFormat = RotationFormat.AXIS_ANGLE,
    gripper: GripperFormat = GripperFormat.SIGNED,
) -> Benchmark:
    embodiment = Embodiment(
        name="two_arm",
        action=EEActionSpace(rotation=rotation, gripper=gripper, arm_count=2, delta=True),
        proprioception=Proprioception(
            ee_pose=EEObservationSpec(
                rotation=rotation, gripper=GripperObservationSpec(dim=2), arm_count=2
            )
        ),
    )
    return Benchmark(
        name="two_arm_suite",
        embodiment=embodiment,
        sensors=[Camera(name="agentview", shape=(8, 8, 3))],
        instruction=False,
    )


def test_bimanual_pairing_checks_both_arms() -> None:
    # An identity pipeline, so what is exercised is the probe and the per-arm
    # comparison rather than any adapter. Both arms must appear in the report: before
    # this, the probe carried one arm and the comparison read arm 0 only, so a
    # right-arm corruption could not be seen.
    bench = _bimanual_benchmark()
    policy = _policy(
        EEActionSpace(
            rotation=RotationFormat.AXIS_ANGLE,
            gripper=GripperFormat.SIGNED,
            arm_count=2,
            delta=True,
        ),
        proprio=Proprioception(
            ee_pose=EEObservationSpec(
                rotation=RotationFormat.AXIS_ANGLE,
                gripper=GripperObservationSpec(dim=2),
                arm_count=2,
            )
        ),
        rotation=RotationFormat.AXIS_ANGLE,
    )

    report = verify(policy, bench, Pipeline())

    assert report.ok, report.reasons
    for name in (
        "action.position.arm0",
        "action.position.arm1",
        "action.rotation.arm0",
        "action.rotation.arm1",
        "action.gripper.arm0",
        "action.gripper.arm1",
    ):
        assert _check(report, name).passed, name


def test_single_arm_check_names_are_unchanged() -> None:
    # The report's check names are part of its contract, so a single-arm pairing must
    # not grow a `.arm0` suffix just because the loop now exists.
    report = verify(
        _policy(_ee(), proprio=_franka_proprio(), rotation=RotationFormat.AXIS_ANGLE),
        _benchmark(),
        Pipeline(),
    )

    assert _check(report, "action.position").passed
    assert all(".arm" not in check.name for check in report.checks)


def _joint_bimanual_benchmark() -> Benchmark:
    # ALOHA's layout, as ROBOTWIN declares it: two arms of six joints and a gripper.
    embodiment = Embodiment(
        name="two_arm_joint",
        action=JointActionSpace(dof=12, gripper=GripperFormat.UNSIGNED, arm_count=2),
        proprioception=Proprioception(
            joint_pos=JointObservationSpec(dof=12, gripper=GripperFormat.UNSIGNED, arm_count=2)
        ),
    )
    return Benchmark(
        name="two_arm_joint_suite",
        embodiment=embodiment,
        sensors=[Camera(name="agentview", shape=(8, 8, 3))],
        instruction=False,
    )


def _joint_bimanual_policy() -> PolicySignature:
    return _policy(
        JointActionSpace(dof=12, gripper=GripperFormat.UNSIGNED, arm_count=2),
        proprio=Proprioception(
            joint_pos=JointObservationSpec(dof=12, gripper=GripperFormat.UNSIGNED, arm_count=2)
        ),
        rotation=RotationFormat.AXIS_ANGLE,
    )


def test_a_joint_pairing_is_compared_arm_by_arm() -> None:
    # A joint pairing used to be length-only: four checks, none touching the layout.
    # Each arm's joints and gripper now appear in the report, action and joint_pos alike.
    report = verify(_joint_bimanual_policy(), _joint_bimanual_benchmark(), Pipeline())

    assert report.ok, report.reasons
    for name in (
        "action.joints.arm0",
        "action.joints.arm1",
        "action.gripper.arm0",
        "action.gripper.arm1",
        "action.gripper.closed.arm0",
        "action.gripper.closed.arm1",
        "observation.joint_pos.arm0",
        "observation.joint_pos.arm1",
    ):
        assert _check(report, name).passed, name


def test_verify_catches_transposed_arms_on_a_joint_pairing() -> None:
    # A runner that swaps the two arms keeps the length, so a length-only check passed
    # it. The probe gives every joint of every arm its own value, so the swap shows.
    class SwapsTheArms(ActionAdapter):
        from_spec = JointActionSpace
        to_spec = JointActionSpace
        lossless = True

        def applies(self, source: Any) -> bool:
            return isinstance(source, JointActionSpace)

        def produce(self, source: Any) -> JointActionSpace:
            return source

        def adapt(self, values: Any, *, source: Any) -> list[float]:
            arm, step = source.per_arm_length(), source.per_step_length()
            out: list[float] = []
            for base in range(0, source.expected_length(), step):
                out.extend(float(v) for v in values[base + arm : base + 2 * arm])
                out.extend(float(v) for v in values[base : base + arm])
            return out

    report = verify(
        _joint_bimanual_policy(),
        _joint_bimanual_benchmark(),
        Pipeline(action=[SwapsTheArms()]),
    )

    assert not report.ok
    assert not _check(report, "action.joints.arm0").passed


def test_verify_catches_an_arm_whose_gripper_is_never_opened() -> None:
    # The probe used to key openness on (step x arms + arm), whose parity is the arm's
    # whenever arm_count is even: arm 0 was always probed open and arm 1 always closed.
    # An adapter jamming arm 1 shut therefore passed every gripper check. Keyed on the
    # step, every arm is probed open.
    class ClampsArmOneShut(ActionAdapter):
        from_spec = EEActionSpace
        to_spec = EEActionSpace
        lossless = True

        def applies(self, source: Any) -> bool:
            return isinstance(source, EEActionSpace)

        def produce(self, source: Any) -> EEActionSpace:
            return source

        def adapt(self, values: Any, *, source: Any) -> list[float]:
            out = [float(v) for v in values]
            arm, step = source.per_arm_length(), source.per_step_length()
            for base in range(0, source.expected_length(), step):
                out[base + 2 * arm - 1] = from_openness(0.0, source.gripper)
            return out

    bench = _bimanual_benchmark()
    action = EEActionSpace(
        rotation=RotationFormat.AXIS_ANGLE,
        gripper=GripperFormat.SIGNED,
        arm_count=2,
        delta=True,
        chunk_size=4,
    )
    bench = bench.model_copy(
        update={"embodiment": bench.embodiment.model_copy(update={"action": action})}
    )
    policy = _policy(
        action,
        proprio=bench.embodiment.proprioception,
        rotation=RotationFormat.AXIS_ANGLE,
    )

    report = verify(policy, bench, Pipeline(action=[ClampsArmOneShut()]))

    assert not report.ok
    assert not _check(report, "action.gripper.step0.arm1").passed


def test_verify_catches_an_ee_pose_corrupted_on_the_second_arm() -> None:
    # The ee_pose comparison read arm 0 only, so a chain that zeroed arm 1 passed both
    # the position and the rotation check. Every arm is now compared on its own slice.
    class ZeroesArmOnePose(ObservationAdapter):
        from_spec = ObservationSpace
        to_spec = ObservationSpace
        lossless = True

        def applies(self, source: Any) -> bool:
            return isinstance(source, ObservationSpace)

        def produce(self, source: Any) -> ObservationSpace:
            return source

        def adapt(self, observation: Any, /, *, source: Any) -> Any:
            arm = source.proprioception.ee_pose.per_arm_length()
            values = np.array(observation.state["ee_pose"], dtype=np.float32)
            values[arm : 2 * arm] = 0.0
            return replace(observation, state={**observation.state, "ee_pose": values})

    bench = _bimanual_benchmark()
    policy = _policy(
        bench.embodiment.action,
        proprio=bench.embodiment.proprioception,
        rotation=RotationFormat.AXIS_ANGLE,
    )

    report = verify(policy, bench, Pipeline(observation=[ZeroesArmOnePose()]))

    assert not report.ok
    assert _check(report, "observation.ee_pose.position.arm0").passed
    assert not _check(report, "observation.ee_pose.position.arm1").passed
