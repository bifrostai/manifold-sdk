"""Tests for the camera-convention adapters and the observability pillar.

Covers PART A (SwapChannelOrder, ResizeCameras) and PART B (tap) of the
adapter-layer workstreams.
"""

import numpy as np
import pytest

from manifold.adapters import ActionTap, ObservationTap
from manifold.adapters.action import GripperThresholdAdapter
from manifold.adapters.observation import FrameRebaseAdapter, ResizeCameras, SwapChannelOrder
from manifold.core import (
    Action,
    Benchmark,
    Camera,
    ChannelOrder,
    Compatibility,
    EEActionSpace,
    EEObservationSpec,
    Embodiment,
    Frame,
    GripperFormat,
    GripperObservationSpec,
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
