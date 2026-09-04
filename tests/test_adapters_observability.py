"""Tests for the camera-convention adapters and the observability pillar.

Covers PART A (SwapChannelOrder, ResizeCameras) and PART B (tap) of the
adapter-layer workstreams.
"""

import numpy as np
import pytest

from manifold.adapters import ActionTap, ObservationTap
from manifold.adapters.action import GripperThresholdAdapter
from manifold.adapters.observation import (
    FlipHorizontalCameras,
    FlipVerticalCameras,
    FrameRebaseAdapter,
    ResizeCameras,
    Rotate180Cameras,
    SwapChannelOrder,
)
from manifold.core import (
    Action,
    Benchmark,
    Camera,
    CameraAxes,
    CameraCalibration,
    CameraIntrinsics,
    CameraOrientation,
    ChannelOrder,
    Compatibility,
    EEActionSpace,
    EEObservationSpec,
    Embodiment,
    Frame,
    GripperFormat,
    GripperObservationSpec,
    Modality,
    Observation,
    ObservationSpace,
    Pipeline,
    PolicySignature,
    Proprioception,
    RotationFormat,
    check_compatibility,
)


def _ee() -> EEActionSpace:
    return EEActionSpace(
        rotation=RotationFormat.AXIS_ANGLE, gripper=GripperFormat.SIGNED, delta=True
    )


def _benchmark(camera: Camera) -> Benchmark:
    embodiment = Embodiment(
        name="arm",
        action=_ee(),
        proprioception=Proprioception(
            ee_pose=EEObservationSpec(rotation=RotationFormat.AXIS_ANGLE)
        ),
    )
    return Benchmark(name="suite", embodiment=embodiment, sensors=[camera], instruction=False)


# --- PART A: SwapChannelOrder -------------------------------------------------


def test_swap_channel_order_adapt_reverses_the_channel_axis() -> None:
    source = ObservationSpace(
        cameras=(Camera(name="agentview", shape=(1, 1, 3), channel_order=ChannelOrder.BGR),)
    )
    # One pixel with distinct channel values so the reversal is observable.
    frame = np.array([[[10, 20, 30]]], dtype=np.uint8)
    extra = np.array([[[1, 2, 3]]], dtype=np.uint8)
    observation = Observation(sensors={"agentview": frame, "extra": extra})

    adapter = SwapChannelOrder(target=ChannelOrder.RGB, cameras=("agentview",))
    result = adapter.adapt(observation, source=source)

    assert np.array_equal(result.sensors["agentview"], [[[30, 20, 10]]])
    assert result.sensors["agentview"].dtype == np.uint8
    assert result.sensors["agentview"].flags["C_CONTIGUOUS"]
    # The unnamed sensor passes through untouched.
    assert np.array_equal(result.sensors["extra"], extra)


def test_swap_channel_order_produce_flips_the_declared_channel_order() -> None:
    source = ObservationSpace(
        cameras=(
            Camera(name="agentview", shape=(1, 1, 3), channel_order=ChannelOrder.BGR),
            Camera(name="extra", shape=(1, 1, 3), channel_order=ChannelOrder.BGR),
        )
    )
    adapter = SwapChannelOrder(target=ChannelOrder.RGB, cameras=("agentview",))

    assert adapter.applies(source)
    produced = adapter.produce(source)

    assert type(produced) is ObservationSpace
    agentview = produced.camera("agentview")
    assert agentview is not None and agentview.channel_order is ChannelOrder.RGB
    # The unnamed camera is untouched.
    extra = produced.camera("extra")
    assert extra is not None and extra.channel_order is ChannelOrder.BGR


@pytest.mark.parametrize(
    "adapter,source",
    [
        (
            SwapChannelOrder(target=ChannelOrder.RGB, cameras=("agentview",)),
            ObservationSpace(
                cameras=(Camera(name="agentview", shape=(1, 1, 3), channel_order=ChannelOrder.BGR),)
            ),
        ),
        (
            ResizeCameras(cameras=("agentview",), shape=(4, 4)),
            ObservationSpace(cameras=(Camera(name="agentview", shape=(8, 8, 3)),)),
        ),
    ],
    ids=["SwapChannelOrder", "ResizeCameras"],
)
def test_adapter_applies_is_false_once_at_target(adapter, source) -> None:
    # The termination invariant: once at the target, the edge no longer fires.
    assert adapter.applies(source)
    assert not adapter.applies(adapter.produce(source))


def test_swap_channel_order_bridges_an_rgb_vs_bgr_mismatch() -> None:
    # The benchmark publishes BGR; the policy consumes RGB. Without the swap the
    # channel-order convention does not match (INCOMPATIBLE); with it the chain
    # type-checks and the pairing is COMPATIBLE_VIA_PIPELINE, losslessly.
    benchmark = _benchmark(
        Camera(name="agentview", shape=(8, 8, 3), channel_order=ChannelOrder.BGR)
    )
    policy = PolicySignature(
        action_space=_ee(),
        cameras=[Camera(name="agentview", shape=(8, 8, 3), channel_order=ChannelOrder.RGB)],
        instruction=False,
    )

    without = check_compatibility(policy, benchmark)
    assert without.status is Compatibility.INCOMPATIBLE

    pipeline = Pipeline(
        observation=[SwapChannelOrder(target=ChannelOrder.RGB, cameras=("agentview",))]
    )
    via = check_compatibility(policy, benchmark, pipeline)
    assert via.status is Compatibility.COMPATIBLE_VIA_PIPELINE
    assert via.lossless is True


# --- PART A: ResizeCameras ----------------------------------------------------


def test_resize_cameras_adapt_produces_the_target_hw() -> None:
    source = ObservationSpace(cameras=(Camera(name="agentview", shape=(8, 8, 3)),))
    frame = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    observation = Observation(sensors={"agentview": frame})

    adapter = ResizeCameras(cameras=("agentview",), shape=(4, 4))
    result = adapter.adapt(observation, source=source)

    # Right shape (H, W preserved channel count), right dtype.
    assert result.sensors["agentview"].shape == (4, 4, 3)
    assert result.sensors["agentview"].dtype == np.uint8


def test_resize_cameras_produce_updates_the_spec_shape() -> None:
    source = ObservationSpace(cameras=(Camera(name="agentview", shape=(8, 8, 3)),))
    adapter = ResizeCameras(targets={"agentview": (224, 224, 3)})

    assert adapter.applies(source)
    produced = adapter.produce(source)

    camera = produced.camera("agentview")
    assert camera is not None
    assert camera.shape == (224, 224, 3)


def test_resize_cameras_flags_lossy() -> None:
    # Resolution is the lossy axis: a resize is never a hard incompatibility but
    # flags the report as lossy (report.lossless is False).
    benchmark = _benchmark(Camera(name="agentview", shape=(8, 8, 3)))
    policy = PolicySignature(
        action_space=_ee(),
        cameras=[Camera(name="agentview", shape=(4, 4, 3))],
        instruction=False,
    )
    pipeline = Pipeline(observation=[ResizeCameras(cameras=("agentview",), shape=(4, 4))])

    report = check_compatibility(policy, benchmark, pipeline)
    assert report.status is Compatibility.COMPATIBLE_VIA_PIPELINE
    assert report.lossless is False


def test_resize_cameras_resizes_the_image_plane_of_a_rank4_clip() -> None:
    # A clip camera (T, H, W, C) — e.g. produced by StackFrameHistory — must resize
    # its trailing H, W and leave the leading time axis and the channel axis intact.
    source = ObservationSpace(cameras=(Camera(name="agentview", shape=(4, 8, 8, 3)),))
    clip = np.arange(4 * 8 * 8 * 3, dtype=np.uint8).reshape(4, 8, 8, 3)
    observation = Observation(sensors={"agentview": clip})

    adapter = ResizeCameras(cameras=("agentview",), shape=(4, 4))

    # produce rewrites only H, W: (4, 8, 8, 3) -> (4, 4, 4, 3).
    produced = adapter.produce(source)
    camera = produced.camera("agentview")
    assert camera is not None and camera.shape == (4, 4, 4, 3)

    # adapt resizes the image plane, preserving the time and channel axes.
    result = adapter.adapt(observation, source=source)
    assert result.sensors["agentview"].shape == (4, 4, 4, 3)
    assert result.sensors["agentview"].dtype == np.uint8


def _depth_space(shape: tuple[int, ...]) -> ObservationSpace:
    return ObservationSpace(
        cameras=(
            Camera(name="agentview_depth", shape=shape, dtype="float32", modality=Modality.DEPTH),
        )
    )


def test_resize_cameras_resamples_a_depth_frame_without_interpolating() -> None:
    # A depth camera resamples nearest-neighbour whatever the adapter's order says. A
    # combining kernel would return distances between two surfaces, and would evaluate
    # `inf - inf` through the spline filter and hand back NaN — replacing the no-hit
    # marker the SDK uses with one it does not (ADR 0007).
    depth = np.array(
        [
            [1.0, 1.0, 5.0, 5.0],
            [1.0, 1.0, 5.0, 5.0],
            [1.0, 1.0, np.inf, np.inf],
            [1.0, 1.0, np.inf, np.inf],
        ],
        dtype=np.float32,
    ).reshape(4, 4, 1)
    observation = Observation(sensors={"agentview_depth": depth})

    adapter = ResizeCameras(cameras=("agentview_depth",), shape=(2, 2))
    out = adapter.adapt(observation, source=_depth_space((4, 4, 1))).sensors["agentview_depth"]

    assert out.shape == (2, 2, 1)
    assert out.dtype == np.float32
    assert not np.isnan(out).any()
    assert np.isinf(out).sum() == 1
    # Every reading is one that was in the input: nothing between the two surfaces.
    finite = out[np.isfinite(out)]
    assert np.all((finite == 1.0) | (finite == 5.0))


def test_resize_cameras_pads_a_depth_frame_with_inf() -> None:
    # Aspect-preserving pad on depth fills with `inf`, not zero: a zero pad is a
    # surface at the lens, which is a distance, where `inf` says there is no reading.
    depth = np.full((4, 2, 1), 2.0, dtype=np.float32)
    observation = Observation(sensors={"agentview_depth": depth})

    adapter = ResizeCameras(cameras=("agentview_depth",), shape=(4, 4), pad=True)
    out = adapter.adapt(observation, source=_depth_space((4, 2, 1))).sensors["agentview_depth"]

    assert out.shape == (4, 4, 1)
    assert np.isinf(out[:, 0, 0]).all()
    assert not (out == 0.0).any()


# --- GripperThresholdAdapter --------------------------------------------------


def _unsigned_ee() -> EEActionSpace:
    # An open-high continuous gripper head (the form the WidowX/LIBERO policies emit
    # before the threshold), single step.
    return EEActionSpace(
        rotation=RotationFormat.AXIS_ANGLE, gripper=GripperFormat.UNSIGNED, delta=True
    )


def _threshold(target: GripperFormat, head: float) -> float:
    adapter = GripperThresholdAdapter(target=target)
    source = _unsigned_ee()
    out = adapter.adapt([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, head], source=source)
    return out[-1]


def test_gripper_threshold_target_polarity_decides_the_open_sign() -> None:
    # The same open-high head above the cutoff is "open"; the target format decides
    # the emitted sign: SIGNED opens at +1, SIGNED_OPEN_LOW opens at -1. One adapter,
    # two declared outcomes — this is the whole point of binarizing into the target.
    assert _threshold(GripperFormat.SIGNED, 0.9) == 1.0
    assert _threshold(GripperFormat.SIGNED_OPEN_LOW, 0.9) == -1.0
    # And below the cutoff (closed) the signs invert accordingly.
    assert _threshold(GripperFormat.SIGNED, 0.1) == -1.0
    assert _threshold(GripperFormat.SIGNED_OPEN_LOW, 0.1) == 1.0


def test_gripper_threshold_is_lossy_and_terminates() -> None:
    adapter = GripperThresholdAdapter(target=GripperFormat.SIGNED)
    assert adapter.lossless is False
    source = _unsigned_ee()
    assert adapter.applies(source)
    # produce sets the target gripper, so applies is False after — fires once.
    produced = adapter.produce(source)
    assert isinstance(produced, EEActionSpace)
    assert produced.gripper is GripperFormat.SIGNED
    assert not adapter.applies(produced)


def test_gripper_threshold_thresholds_every_chunk_step() -> None:
    # A 2-step chunk with one open head and one closed head: each step's gripper is
    # thresholded independently into the target.
    adapter = GripperThresholdAdapter(target=GripperFormat.SIGNED_OPEN_LOW)
    source = EEActionSpace(
        rotation=RotationFormat.AXIS_ANGLE,
        gripper=GripperFormat.UNSIGNED,
        delta=True,
        chunk_size=2,
    )
    step_open = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]
    step_closed = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]
    out = adapter.adapt([*step_open, *step_closed], source=source)
    assert out[6] == -1.0  # open -> -1 under SIGNED_OPEN_LOW
    assert out[13] == 1.0  # closed -> +1


# --- FrameRebaseAdapter -------------------------------------------------------


def _quat_proprio(frame: Frame = Frame.BASE) -> ObservationSpace:
    return ObservationSpace(
        proprioception=Proprioception(
            ee_pose=EEObservationSpec(
                rotation=RotationFormat.QUATERNION,
                gripper=GripperObservationSpec(dim=1),
                frame=frame,
            )
        )
    )


def test_frame_rebase_produce_advances_the_frame_and_terminates() -> None:
    rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
    adapter = FrameRebaseAdapter(Frame.BASE, Frame.WIDOWX_BRIDGE_EE, rot)
    source = _quat_proprio(Frame.BASE)

    assert adapter.lossless is True
    assert adapter.applies(source)
    produced = adapter.produce(source)
    assert produced.proprioception.ee_pose is not None
    assert produced.proprioception.ee_pose.frame is Frame.WIDOWX_BRIDGE_EE
    # The rotation format is untouched (only the frame changes); ProprioRotation
    # handles the format re-encode downstream.
    assert produced.proprioception.ee_pose.rotation is RotationFormat.QUATERNION
    # Termination: once at the target frame the edge no longer fires.
    assert not adapter.applies(produced)


def test_frame_rebase_is_a_noop_when_the_frame_already_differs() -> None:
    rot = np.eye(3)
    adapter = FrameRebaseAdapter(Frame.BASE, Frame.WIDOWX_BRIDGE_EE, rot)
    # Source already in the target frame -> does not apply.
    assert not adapter.applies(_quat_proprio(Frame.WIDOWX_BRIDGE_EE))


# --- PART B: tap --------------------------------------------------------------


def test_observation_tap_returns_identical_data_and_calls_the_sink() -> None:
    source = ObservationSpace(
        proprioception=Proprioception(
            ee_pose=EEObservationSpec(rotation=RotationFormat.AXIS_ANGLE)
        ),
        cameras=(Camera(name="agentview", shape=(2, 2, 3)),),
    )
    state = np.arange(6, dtype=np.float32)
    frame = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    observation = Observation(
        state={"ee_pose": state}, sensors={"agentview": frame}, instruction="pick the cube"
    )
    readings = []

    tap = ObservationTap(label="seam", sink=readings.append)
    result = tap.adapt(observation, source=source)

    # Byte-identical data passed through.
    assert result is observation
    assert np.array_equal(result.state["ee_pose"], state)
    assert np.array_equal(result.sensors["agentview"], frame)
    # The sink was called with a per-field summary.
    assert len(readings) == 1
    assert readings[0]["label"] == "seam"
    assert readings[0]["fields"]["state.ee_pose"]["shape"] == (6,)
    assert readings[0]["fields"]["sensors.agentview"]["dtype"] == "uint8"


def test_action_tap_returns_identical_values_and_calls_the_sink() -> None:
    source = _ee()
    values = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 1.0], dtype=np.float32)
    readings = []

    tap = ActionTap(label="action-seam", sink=readings.append)
    result = tap.adapt(values, source=source)

    assert result is values
    assert len(readings) == 1
    assert readings[0]["label"] == "action-seam"
    assert readings[0]["fields"]["values"]["shape"] == (7,)


def test_a_tap_is_transparent_in_a_pipeline() -> None:
    # A pipeline with a tap behaves identically to one without it: the tap observes
    # but never changes the data flowing through, on both sides.
    obs_source = ObservationSpace(
        proprioception=Proprioception(
            ee_pose=EEObservationSpec(rotation=RotationFormat.AXIS_ANGLE)
        ),
        cameras=(Camera(name="agentview", shape=(2, 2, 3)),),
    )
    observation = Observation(
        state={"ee_pose": np.arange(6, dtype=np.float32)},
        sensors={"agentview": np.arange(12, dtype=np.uint8).reshape(2, 2, 3)},
    )
    action_source = _ee()
    action = Action.from_array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 1.0])

    plain = Pipeline()
    tapped = Pipeline(observation=[ObservationTap()], action=[ActionTap()])

    plain_obs = plain.apply_observation(observation, source=obs_source)
    tapped_obs = tapped.apply_observation(observation, source=obs_source)
    assert np.array_equal(plain_obs.state["ee_pose"], tapped_obs.state["ee_pose"])
    assert np.array_equal(plain_obs.sensors["agentview"], tapped_obs.sensors["agentview"])

    plain_act = plain.apply_action(action, source=action_source)
    tapped_act = tapped.apply_action(action, source=action_source)
    assert np.array_equal(plain_act.values, tapped_act.values)


# --- calibration through the geometry adapters --------------------------------


def _calibrated(shape: tuple[int, ...], fovy: float = 45.0) -> Camera:
    return Camera(
        name="agentview",
        shape=shape,
        calibration=CameraCalibration(
            intrinsics=CameraIntrinsics.from_fov(
                fovy_degrees=fovy, height=shape[-3], width=shape[-2]
            ),
            axes=CameraAxes.OPENCV,
            frame=Frame.WORLD,
        ),
    )


def test_resize_scales_the_intrinsics_with_the_image() -> None:
    # Resampling moves the pixel grid the pinhole is measured in, so all four values
    # scale. Leaving them behind would keep claiming the old field of view (ADR 0008).
    source = ObservationSpace(cameras=(_calibrated((256, 256, 3)),))
    produced = ResizeCameras(cameras=("agentview",), shape=(128, 128)).produce(source)

    camera = produced.camera("agentview")
    assert camera is not None and camera.calibration is not None
    k = camera.calibration.intrinsics
    assert k.fx == pytest.approx(309.019336 / 2, rel=1e-6)
    assert k.cx == pytest.approx(64.0) and k.cy == pytest.approx(64.0)


def test_padded_resize_shifts_the_principal_point_by_the_centring_offset() -> None:
    # Aspect-preserving pad scales by one factor and then centres, so the principal
    # point moves by the pad rather than staying at the middle of the old image.
    source = ObservationSpace(cameras=(_calibrated((128, 64, 3)),))
    produced = ResizeCameras(cameras=("agentview",), shape=(128, 128), pad=True).produce(source)

    camera = produced.camera("agentview")
    assert camera is not None and camera.calibration is not None
    # scale = min(128/128, 128/64) = 1.0; the 64-wide image is centred in 128 -> +32.
    assert camera.calibration.intrinsics.cx == pytest.approx(32.0 + 32.0)
    assert camera.calibration.intrinsics.cy == pytest.approx(64.0)


def test_rotate180_leaves_the_calibration_alone() -> None:
    # Rearranging an array re-labels which row is row 0; it does not move the camera. So
    # the intrinsics and the pose pass through untouched and only `orientation` advances.
    # "Correcting" either one would put the scene underground (ADR 0008).
    source = ObservationSpace(cameras=(_calibrated((256, 256, 3)),))
    adapter = Rotate180Cameras(cameras=("agentview",))

    produced = adapter.produce(source).camera("agentview")
    assert produced is not None and produced.calibration is not None
    assert produced.orientation is CameraOrientation.ROTATED_180
    assert produced.calibration == _calibrated((256, 256, 3)).calibration

    pose = np.eye(4)
    pose[:3, 3] = (1.0, 2.0, 3.0)
    result = adapter.adapt(
        Observation(
            sensors={"agentview": np.zeros((256, 256, 3), dtype=np.uint8)},
            extrinsics={"agentview": pose},
        ),
        source=source,
    )
    np.testing.assert_array_equal(result.extrinsics["agentview"], pose)


def test_a_pose_passes_through_an_adapter_that_does_not_touch_geometry() -> None:
    # The hazard a fourth aggregate introduces: an adapter that rebuilds the
    # observation must not drop it. Channel order is not geometry, so it passes through.
    source = ObservationSpace(cameras=(_calibrated((4, 4, 3)),))
    pose = np.eye(4)
    result = SwapChannelOrder(ChannelOrder.BGR, cameras=("agentview",)).adapt(
        Observation(
            sensors={"agentview": np.zeros((4, 4, 3), dtype=np.uint8)},
            extrinsics={"agentview": pose},
        ),
        source=source,
    )
    np.testing.assert_array_equal(result.extrinsics["agentview"], pose)


def test_a_vertical_flip_keeps_the_calibration() -> None:
    # The flip is what MAKES an OpenCV calibration valid for a bottom-up renderer: the
    # intrinsics assume row 0 is the top, and this is the adapter that makes that true.
    # So it carries the calibration rather than dropping it.
    source = ObservationSpace(cameras=(_calibrated((256, 256, 3)),))
    adapter = FlipVerticalCameras(cameras=("agentview",))

    assert adapter.applies(source) is True
    produced = adapter.produce(source).camera("agentview")
    assert produced is not None
    assert produced.orientation is CameraOrientation.FLIPPED_VERTICAL
    assert produced.calibration == _calibrated((256, 256, 3)).calibration


def test_a_vertical_flip_keeps_the_pose_too() -> None:
    source = ObservationSpace(cameras=(_calibrated((4, 4, 3)),))
    pose = np.eye(4)
    result = FlipVerticalCameras(cameras=("agentview",)).adapt(
        Observation(
            sensors={"agentview": np.zeros((4, 4, 3), dtype=np.uint8)},
            extrinsics={"agentview": pose},
        ),
        source=source,
    )
    np.testing.assert_array_equal(result.extrinsics["agentview"], pose)


# --- PART C: camera orientation as a closed vocabulary ------------------------

UP = CameraOrientation.UPRIGHT
R180 = CameraOrientation.ROTATED_180
FV = CameraOrientation.FLIPPED_VERTICAL
FH = CameraOrientation.FLIPPED_HORIZONTAL


def _oriented(
    orientation: CameraOrientation, shape: tuple[int, ...] = (2, 3, 3)
) -> ObservationSpace:
    return ObservationSpace(
        cameras=(Camera(name="agentview", shape=shape, orientation=orientation),)
    )


def _frame(observation: Observation) -> np.ndarray:
    return np.asarray(observation.sensors["agentview"])


def _observation(frame: np.ndarray) -> Observation:
    return Observation(state={}, sensors={"agentview": frame}, instruction=None)


def test_the_orientations_are_closed_under_composition() -> None:
    # The four are the two independent mirrors under composition, so composing any
    # pair lands on a member (closure), doing it twice returns the original (every
    # member is its own inverse), and UPRIGHT is the identity. A vocabulary missing
    # one of the four would break closure, which is what the fourth member fixes.
    for a in CameraOrientation:
        assert a.flipped(UP) is a
        for b in CameraOrientation:
            assert a.flipped(b) in set(CameraOrientation)
            assert a.flipped(b).flipped(b) is a


@pytest.mark.parametrize(
    "adapter,source,target",
    [
        # The two edges that existed before this vocabulary closed: unchanged.
        (Rotate180Cameras(cameras=("agentview",)), UP, R180),
        (FlipVerticalCameras(cameras=("agentview",)), UP, FV),
        # ... and the ones only composition can express.
        (Rotate180Cameras(cameras=("agentview",)), FV, FH),
        (Rotate180Cameras(cameras=("agentview",)), R180, UP),
        (Rotate180Cameras(cameras=("agentview",)), FH, FV),
        (FlipVerticalCameras(cameras=("agentview",)), FV, UP),
        (FlipVerticalCameras(cameras=("agentview",)), R180, FH),
        (FlipVerticalCameras(cameras=("agentview",)), FH, R180),
        (FlipHorizontalCameras(cameras=("agentview",)), UP, FH),
        (FlipHorizontalCameras(cameras=("agentview",)), FV, R180),
        (FlipHorizontalCameras(cameras=("agentview",)), R180, FV),
        (FlipHorizontalCameras(cameras=("agentview",)), FH, UP),
    ],
)
def test_an_orientation_edge_composes_onto_whatever_the_source_declares(
    adapter, source, target
) -> None:
    # The whole table, so a change to the composition cannot pass by only covering
    # the UPRIGHT row. The first two rows are the pre-existing edges: this is also
    # the regression guard that generalising did not move them.
    space = _oriented(source)
    assert adapter.applies(space)
    assert adapter.produce(space).camera("agentview").orientation is target


def test_a_flip_bridges_a_benchmark_that_publishes_bottom_up_frames() -> None:
    # The case the old `is UPRIGHT` precondition could not express, and the reason
    # this change exists: a benchmark that ships FLIPPED_VERTICAL frames against a
    # policy trained on upright ones. Before, no edge left FLIPPED_VERTICAL at all.
    benchmark = _benchmark(Camera(name="agentview", shape=(2, 3, 3), orientation=FV))
    signature = PolicySignature(
        action_space=_ee(),
        proprioception=benchmark.embodiment.proprioception,
        cameras=[Camera(name="agentview", shape=(2, 3, 3), orientation=UP)],
        instruction=False,
    )
    pipeline = Pipeline(observation=[FlipVerticalCameras(cameras=("agentview",))])
    report = check_compatibility(signature, benchmark, pipeline)
    assert report.status is Compatibility.COMPATIBLE_VIA_PIPELINE
    assert report.lossless


def test_each_flip_reverses_only_its_own_axis() -> None:
    # Distinctness: the rotation is the composition of the two mirrors, so using one
    # where another is wanted adds a spurious swap. Pinned on an asymmetric frame.
    frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    source = _oriented(UP)
    vertical = _frame(
        FlipVerticalCameras(cameras=("agentview",)).adapt(_observation(frame), source=source)
    )
    horizontal = _frame(
        FlipHorizontalCameras(cameras=("agentview",)).adapt(_observation(frame), source=source)
    )
    rotated = _frame(
        Rotate180Cameras(cameras=("agentview",)).adapt(_observation(frame), source=source)
    )
    np.testing.assert_array_equal(vertical, frame[::-1, :, :])
    np.testing.assert_array_equal(horizontal, frame[:, ::-1, :])
    np.testing.assert_array_equal(rotated, frame[::-1, ::-1, :])
    # The rotation IS the two mirrors composed.
    np.testing.assert_array_equal(rotated, vertical[:, ::-1, :])


@pytest.mark.parametrize(
    "adapter,expected",
    [
        (FlipVerticalCameras(cameras=("agentview",)), lambda a: a[:, ::-1, :, :]),
        (FlipHorizontalCameras(cameras=("agentview",)), lambda a: a[:, :, ::-1, :]),
        (Rotate180Cameras(cameras=("agentview",)), lambda a: a[:, ::-1, ::-1, :]),
    ],
    ids=["FlipVertical", "FlipHorizontal", "Rotate180"],
)
def test_a_flip_reorients_the_image_plane_of_a_rank4_clip(adapter, expected) -> None:
    # `StackFrameHistory` makes rank-4 camera tensors and RLDX consumes them, so a
    # flip written against rank 3 would reverse the TIME axis of a clip instead of
    # its rows - silently feeding the model its frames backwards. The axes are
    # addressed from the end of the shape for exactly this reason.
    clip = np.arange(2 * 2 * 3 * 3, dtype=np.uint8).reshape(2, 2, 3, 3)
    out = _frame(adapter.adapt(_observation(clip), source=_oriented(UP, shape=(2, 2, 3, 3))))
    assert out.shape == clip.shape
    np.testing.assert_array_equal(out, expected(clip))
