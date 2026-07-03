from manifold.adapters.action import BasePinWiden, GripperPolarityAdapter
from manifold.adapters.observation import Rotate180Cameras
from manifold.benchmarks.libero import LIBERO
from manifold.core import (
    Benchmark,
    Camera,
    CameraOrientation,
    Compatibility,
    EEActionSpace,
    EEObservationSpec,
    Embodiment,
    GripperFormat,
    GripperObservationSpec,
    JointActionSpace,
    Mount,
    ObservationSpace,
    Pipeline,
    PolicySignature,
    Proprioception,
    RotationFormat,
    UnifiedActionSpace,
    check_compatibility,
)
from manifold.embodiments.panda_omron import PANDA_OMRON


def _ee(gripper: GripperFormat = GripperFormat.SIGNED) -> EEActionSpace:
    return EEActionSpace(rotation=RotationFormat.AXIS_ANGLE, gripper=gripper, delta=True)


def _benchmark(*, instruction: bool = True) -> Benchmark:
    embodiment = Embodiment(
        name="arm",
        action=_ee(),
        proprioception=Proprioception(
            ee_pose=EEObservationSpec(rotation=RotationFormat.AXIS_ANGLE)
        ),
    )
    return Benchmark(
        name="suite",
        embodiment=embodiment,
        sensors=[Camera(name="agentview", shape=(224, 224, 3))],
        instruction=instruction,
    )


def _agentview(
    *,
    orientation: CameraOrientation = CameraOrientation.UPRIGHT,
    mount: Mount = Mount.SCENE,
) -> Camera:
    """The benchmark's agentview camera, optionally with overridden conventions."""
    return Camera(name="agentview", shape=(224, 224, 3), orientation=orientation, mount=mount)


def test_native_pairing_is_compatible() -> None:
    benchmark = _benchmark()
    policy = PolicySignature(
        action_space=_ee(),
        proprioception=benchmark.observation_space.proprioception,
        cameras=list(benchmark.observation_space.cameras),
        instruction=benchmark.observation_space.instruction,
    )
    assert check_compatibility(policy, benchmark).status is Compatibility.COMPATIBLE


def test_action_mismatch_without_a_pipeline_is_incompatible() -> None:
    policy = PolicySignature(
        action_space=JointActionSpace(dof=7), cameras=[_agentview()], instruction=False
    )
    report = check_compatibility(policy, _benchmark())
    assert report.status is Compatibility.INCOMPATIBLE
    assert not report.ok
    assert report.reasons


def test_action_bridged_by_a_pipeline_is_compatible_via_pipeline() -> None:
    policy = PolicySignature(
        action_space=_ee(gripper=GripperFormat.SIGNED_OPEN_LOW),
        cameras=[_agentview()],
        instruction=False,
    )
    pipeline = Pipeline(action=[GripperPolarityAdapter(target=GripperFormat.SIGNED)])
    report = check_compatibility(policy, _benchmark(), pipeline)
    assert report.status is Compatibility.COMPATIBLE_VIA_PIPELINE
    assert report.lossless is True


def test_a_required_camera_the_benchmark_lacks_is_incompatible() -> None:
    policy = PolicySignature(
        action_space=_ee(),
        cameras=[Camera(name="wrist", shape=(224, 224, 3))],
        instruction=False,
    )
    report = check_compatibility(policy, _benchmark())
    assert not report.ok
    assert any("wrist" in reason and "not published" in reason for reason in report.reasons)


def test_wanting_an_absent_instruction_is_incompatible() -> None:
    policy = PolicySignature(action_space=_ee(), cameras=[_agentview()], instruction=True)
    report = check_compatibility(policy, _benchmark(instruction=False))
    assert not report.ok
    assert any("instruction" in reason for reason in report.reasons)


def test_consuming_no_cameras_or_proprioception_is_compatible() -> None:
    # A policy consuming neither cameras nor proprioception matches any benchmark
    # that publishes some — `first_unmet` iterates only what the policy declares.
    policy = PolicySignature(action_space=_ee(), instruction=False)
    assert check_compatibility(policy, _benchmark()).status is Compatibility.COMPATIBLE


def test_consuming_ee_gripper_a_benchmark_lacks_is_incompatible() -> None:
    # The plain `_benchmark` publishes a gripper-less ee_pose; a policy whose
    # ee_pose nests a gripper must be flagged unmet (it would otherwise be
    # zero-padded and the policy could not sense its grasp). The nested gripper
    # changes the ee_pose's expected length, so it is the ee_pose that mismatches.
    policy = PolicySignature(
        action_space=_ee(),
        proprioception=Proprioception(
            ee_pose=EEObservationSpec(
                rotation=RotationFormat.AXIS_ANGLE,
                gripper=GripperObservationSpec(dim=2),
            )
        ),
        cameras=[_agentview()],
        instruction=False,
    )
    report = check_compatibility(policy, _benchmark())
    assert report.status is Compatibility.INCOMPATIBLE
    assert any("ee_pose" in reason for reason in report.reasons)


def test_camera_orientation_mismatch_without_a_flip_is_incompatible() -> None:
    # The benchmark publishes UPRIGHT frames; the policy was trained on flipped
    # ones. With no flip adapter the orientation convention does not match, and the
    # camera is a real spec axis now, so the pairing is INCOMPATIBLE.
    policy = PolicySignature(
        action_space=_ee(),
        cameras=[_agentview(orientation=CameraOrientation.ROTATED_180)],
        instruction=False,
    )
    report = check_compatibility(policy, _benchmark())
    assert report.status is Compatibility.INCOMPATIBLE
    assert any("agentview" in reason and "conventions" in reason for reason in report.reasons)


def test_camera_orientation_bridged_by_rotate180_is_compatible_via_pipeline() -> None:
    # The same orientation gap, but a Rotate180Cameras in the pipeline flips the
    # declared orientation UPRIGHT -> ROTATED_180, so the chain type-checks and the
    # pairing is COMPATIBLE_VIA_PIPELINE (losslessly).
    policy = PolicySignature(
        action_space=_ee(),
        cameras=[_agentview(orientation=CameraOrientation.ROTATED_180)],
        instruction=False,
    )
    pipeline = Pipeline(observation=[Rotate180Cameras(cameras=("agentview",))])
    report = check_compatibility(policy, _benchmark(), pipeline)
    assert report.status is Compatibility.COMPATIBLE_VIA_PIPELINE
    assert report.lossless is True


def test_camera_shape_mismatch_surfaces_via_first_unmet() -> None:
    # A shape mismatch on a consumed camera surfaces as a conventions difference
    # with a clear, camera-named reason.
    policy = PolicySignature(
        action_space=_ee(),
        cameras=[Camera(name="agentview", shape=(256, 256, 3))],
        instruction=False,
    )
    report = check_compatibility(policy, _benchmark())
    assert report.status is Compatibility.INCOMPATIBLE
    assert any("agentview" in reason and "conventions" in reason for reason in report.reasons)


def test_first_unmet_ignores_mount_provenance() -> None:
    # `mount` is provenance, not a consumed axis: a policy whose agentview is
    # declared WRIST-mounted still matches an otherwise-identical SCENE camera.
    available = _benchmark().observation_space
    wanting = ObservationSpace(cameras=(_agentview(mount=Mount.WRIST),))
    assert wanting.first_unmet(available) is None


def test_franka_ee_nests_gripper_in_ee_pose_and_a_gripper_contract_pairs_with_libero() -> None:
    # FRANKA_EE/LIBERO now publishes the parallel-jaw gripper qpos nested in its
    # ee_pose, so its EE value is 9-D — the native rotation is the (x, y, z, w)
    # QUATERNION robosuite emits (ADR-0001, decision 7: the embodiment records the env's TRUE
    # native form; the pairing pipeline bridges it to whatever the policy consumes).
    # A policy that consumes that same native quaternion with the same nested gripper
    # pairs without padding or a proprio rotation adapter.
    ee_pose = LIBERO.embodiment.proprioception.ee_pose
    assert ee_pose is not None
    assert ee_pose.rotation == RotationFormat.QUATERNION
    assert ee_pose.gripper == GripperObservationSpec(dim=2)
    assert ee_pose.expected_length() == 9  # pos(3) + quat_xyzw(4) + gripper_qpos(2)
    policy = PolicySignature(
        # LIBERO's action gripper is SIGNED_OPEN_LOW (the lerobot env opens at -1),
        # so a clean match declares the same — this test is about the nested gripper
        # qpos in proprioception, not an action-side polarity conversion.
        action_space=_ee(gripper=GripperFormat.SIGNED_OPEN_LOW),
        proprioception=Proprioception(
            ee_pose=EEObservationSpec(
                rotation=RotationFormat.QUATERNION,
                gripper=GripperObservationSpec(dim=2),
            ),
        ),
        cameras=list(LIBERO.observation_space.cameras),
        instruction=True,
    )
    assert check_compatibility(policy, LIBERO).status is Compatibility.COMPATIBLE


def test_frame_history_is_the_spec_edge_for_a_clip_consuming_policy() -> None:
    from manifold.adapters.observation import StackFrameHistory

    # The benchmark publishes a single (256, 256, 3) frame; the policy consumes a
    # (4, 256, 256, 3) clip. The shapes differ, so without a pipeline the pairing
    # is INCOMPATIBLE — only a StackFrameHistory lengthens the camera to the clip.
    benchmark = Benchmark(
        name="suite",
        embodiment=Embodiment(
            name="arm",
            action=_ee(),
            proprioception=Proprioception(
                ee_pose=EEObservationSpec(rotation=RotationFormat.AXIS_ANGLE)
            ),
        ),
        sensors=[Camera(name="agentview", shape=(256, 256, 3))],
        instruction=True,
    )
    policy = PolicySignature(
        action_space=_ee(),
        proprioception=benchmark.embodiment.proprioception,
        cameras=[Camera(name="agentview", shape=(4, 256, 256, 3))],
        instruction=True,
    )

    without = check_compatibility(policy, benchmark)
    assert without.status is Compatibility.INCOMPATIBLE

    pipeline = Pipeline(observation=[StackFrameHistory(cameras=("agentview",))])
    via = check_compatibility(policy, benchmark, pipeline)
    assert via.status is Compatibility.COMPATIBLE_VIA_PIPELINE


def _panda_omron_arm() -> EEActionSpace:
    # The 7-D arm payload PANDA_OMRON's 12-D whole-body action carries, declared
    # standalone — the action a 7-D arm-only policy (Cosmos RoboCasa) emits.
    action = PANDA_OMRON.action
    assert isinstance(action, UnifiedActionSpace)
    arm = action.payload
    assert isinstance(arm, EEActionSpace)
    return arm


def _robocasa_benchmark() -> Benchmark:
    # A PANDA_OMRON benchmark whose action is the 12-D whole-body UnifiedActionSpace.
    return Benchmark(
        name="robocasa",
        embodiment=PANDA_OMRON,
        sensors=[Camera(name="agentview_left", shape=(224, 224, 3))],
        instruction=True,
    )


def test_base_pin_widen_bridges_a_7d_arm_action_to_the_12d_whole_body_space() -> None:
    # A 7-D arm-only policy (the Cosmos RoboCasa case) emits the PANDA_OMRON arm
    # payload; BasePinWiden widens it to the benchmark's 12-D UnifiedActionSpace by
    # appending the base-pin tail, so the action chain reaches the benchmark's space.
    policy = PolicySignature(
        action_space=_panda_omron_arm(),
        proprioception=PANDA_OMRON.proprioception,
        cameras=[Camera(name="agentview_left", shape=(224, 224, 3))],
        instruction=True,
    )
    pipeline = Pipeline(action=[BasePinWiden(width=12)])
    report = check_compatibility(policy, _robocasa_benchmark(), pipeline)
    assert report.status is Compatibility.COMPATIBLE_VIA_PIPELINE
    # The widening invents the base tail, so it is lossy (a caller opts in).
    assert report.lossless is False


def test_base_pin_widen_produces_the_exact_benchmark_action_space() -> None:
    # produce() must equal the benchmark's declared 12-D action so the chain gates.
    adapter = BasePinWiden(width=12)
    arm = _panda_omron_arm()
    assert adapter.applies(arm)
    assert adapter.produce(arm) == PANDA_OMRON.action


def test_base_pin_widen_appends_the_pin_tail_per_step() -> None:
    # A 2-step chunk of the 7-D arm action widens to 2 x 12, each arm block followed
    # by the [0, 0, 0, 0, -1] base pin.
    arm = _panda_omron_arm().model_copy(update={"chunk_size": 2})
    adapter = BasePinWiden(width=12)
    step_a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    step_b = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
    out = adapter.adapt([*step_a, *step_b], source=arm)
    assert out == [*step_a, 0.0, 0.0, 0.0, 0.0, -1.0, *step_b, 0.0, 0.0, 0.0, 0.0, -1.0]


def test_base_pin_widen_does_not_apply_to_a_full_width_action() -> None:
    # An arm action already as wide as the target has no room for the pin tail,
    # so the adapter does not fire (the termination invariant).
    wide_arm = EEActionSpace(
        rotation=RotationFormat.ROTATION_6D,  # pos3 + rot6 + gripper1 = 10
        gripper=GripperFormat.SIGNED,
        delta=True,
    )
    assert not BasePinWiden(width=7).applies(wide_arm)
