import numpy as np
import pytest

from manifold.adapters.action import GripperPolarityAdapter, RotationFormatAdapter
from manifold.adapters.observation import (
    ProprioRotationAdapter,
    Rotate180Cameras,
    StackFrameHistory,
)
from manifold.core import (
    Action,
    Camera,
    EEActionSpace,
    EEObservationSpec,
    GripperFormat,
    Observation,
    ObservationSpace,
    Pipeline,
    Proprioception,
    RotationFormat,
)
from manifold.core.pipeline import StatefulStateMissing
from manifold.core.state import DEFAULT_LANE, PipelineState


def _ee_action_space(rotation: RotationFormat, gripper: GripperFormat) -> EEActionSpace:
    return EEActionSpace(rotation=rotation, gripper=gripper, delta=True)


def test_apply_observation_empty_chain_is_a_noop() -> None:
    source = ObservationSpace(
        proprioception=Proprioception(ee_pose=EEObservationSpec(rotation=RotationFormat.AXIS_ANGLE))
    )
    observation = Observation(
        state={"ee_pose": np.arange(6, dtype=np.float32)},
        sensors={"agentview": np.zeros((4, 4, 3), dtype=np.uint8)},
        instruction="pick up the cube",
    )

    result = Pipeline().apply_observation(observation, source=source)

    # values, instruction, and sensors are all preserved unchanged.
    assert np.array_equal(result.state["ee_pose"], observation.state["ee_pose"])
    assert result.instruction == "pick up the cube"
    assert np.array_equal(result.sensors["agentview"], observation.sensors["agentview"])


def test_apply_action_single_adapter_remaps_the_gripper() -> None:
    # SIGNED opens at +1; SIGNED_OPEN_LOW opens at -1, so the polarity flips sign.
    source = _ee_action_space(RotationFormat.AXIS_ANGLE, GripperFormat.SIGNED)
    pipeline = Pipeline(action=[GripperPolarityAdapter(target=GripperFormat.SIGNED_OPEN_LOW)])
    # pos(3) + axis-angle(3) + gripper(1): the gripper is the trailing element.
    action = Action.from_array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 1.0])

    result = pipeline.apply_action(action, source=source)

    # The pose passes through untouched; only the gripper is re-encoded.
    assert np.allclose(result.values[:6], action.values[:6])
    assert result.values[6] == -1.0


def test_apply_action_single_adapter_walks_the_spec_for_rotation() -> None:
    # Axis-angle (3 floats) -> quaternion (4 floats): the produced spec is wider,
    # so a correct spec walk yields a 7 -> 8 length change.
    source = _ee_action_space(RotationFormat.AXIS_ANGLE, GripperFormat.SIGNED)
    pipeline = Pipeline(action=[RotationFormatAdapter(target=RotationFormat.QUATERNION)])
    action = Action.from_array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0])

    result = pipeline.apply_action(action, source=source)

    produced = source.model_copy(update={"rotation": RotationFormat.QUATERNION})
    assert len(result.values) == produced.expected_length() == 8
    # Position and gripper survive at their new offsets; a zero rotation vector
    # encodes to the identity quaternion (0, 0, 0, 1).
    assert np.allclose(result.values[:3], [0.1, -0.2, 0.3])
    assert np.allclose(result.values[3:7], [0.0, 0.0, 0.0, 1.0])
    assert result.values[7] == 1.0


def test_apply_observation_single_adapter_reencodes_the_pose() -> None:
    source = ObservationSpace(
        proprioception=Proprioception(ee_pose=EEObservationSpec(rotation=RotationFormat.AXIS_ANGLE))
    )
    pipeline = Pipeline(observation=[ProprioRotationAdapter(target=RotationFormat.QUATERNION)])
    observation = Observation(
        state={"ee_pose": np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0], dtype=np.float32)},
        instruction="stack the blocks",
    )

    result = pipeline.apply_observation(observation, source=source)

    # axis-angle(3) -> quaternion(4) widens the pose from 6 to 7 floats.
    assert len(result.state["ee_pose"]) == 7
    assert np.allclose(result.state["ee_pose"][:3], [0.1, -0.2, 0.3])
    assert np.allclose(result.state["ee_pose"][3:7], [0.0, 0.0, 0.0, 1.0])
    assert result.instruction == "stack the blocks"


def test_rotate180_cameras_flips_named_cameras_and_passes_through_the_rest() -> None:
    # An asymmetric 2x2x3 frame so a 180-degree rotation is observable: the corners
    # swap diagonally. agentview is rotated; the unnamed "extra" sensor and the
    # state and instruction all pass through untouched.
    source = ObservationSpace(
        proprioception=Proprioception(
            ee_pose=EEObservationSpec(rotation=RotationFormat.AXIS_ANGLE)
        ),
        cameras=(Camera(name="agentview", shape=(2, 2, 3)),),
    )
    agentview = np.array([[[1, 0, 0], [2, 0, 0]], [[3, 0, 0], [4, 0, 0]]], dtype=np.uint8)
    extra = np.array([[[9, 9, 9]]], dtype=np.uint8)
    observation = Observation(
        state={"ee_pose": np.arange(6, dtype=np.float32)},
        sensors={"agentview": agentview, "extra": extra},
        instruction="pick up the cube",
    )

    pipeline = Pipeline(observation=[Rotate180Cameras(cameras=("agentview",))])
    result = pipeline.apply_observation(observation, source=source)

    # 180-degree rotation: [[1,2],[3,4]] -> [[4,3],[2,1]] on the channel-0 plane.
    assert np.array_equal(result.sensors["agentview"][..., 0], [[4, 3], [2, 1]])
    assert result.sensors["agentview"].dtype == np.uint8
    assert result.sensors["agentview"].flags["C_CONTIGUOUS"]
    # The unnamed sensor, state, and instruction are untouched.
    assert np.array_equal(result.sensors["extra"], extra)
    assert np.array_equal(result.state["ee_pose"], observation.state["ee_pose"])
    assert result.instruction == "pick up the cube"


def test_rotate180_cameras_is_self_inverse() -> None:
    source = ObservationSpace(
        proprioception=Proprioception(
            ee_pose=EEObservationSpec(rotation=RotationFormat.AXIS_ANGLE)
        ),
        cameras=(Camera(name="wrist", shape=(2, 3, 3)),),
    )
    frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    observation = Observation(sensors={"wrist": frame})

    adapter = Rotate180Cameras(cameras=("wrist",))
    once = adapter.adapt(observation, source=source)
    twice = adapter.adapt(once, source=source)

    # Applying the rotation twice returns the original frame.
    assert not np.array_equal(once.sensors["wrist"], frame)
    assert np.array_equal(twice.sensors["wrist"], frame)


def test_rotate180_cameras_skips_an_absent_camera() -> None:
    source = ObservationSpace(
        proprioception=Proprioception(
            ee_pose=EEObservationSpec(rotation=RotationFormat.AXIS_ANGLE)
        ),
        cameras=(Camera(name="agentview", shape=(2, 2, 3)),),
    )
    agentview = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    observation = Observation(sensors={"agentview": agentview})

    # "wrist" is named but absent; it must be skipped rather than crash.
    adapter = Rotate180Cameras(cameras=("agentview", "wrist"))
    result = adapter.adapt(observation, source=source)

    assert "wrist" not in result.sensors
    assert np.array_equal(result.sensors["agentview"], agentview[::-1, ::-1])


def test_apply_action_threads_produce_between_two_adapters() -> None:
    # Two steps: first re-encode the rotation, then flip the gripper polarity. The
    # second adapter must see the spec the first produced (quaternion), not the
    # original (axis-angle), or it would slice the gripper at the wrong offset.
    source = _ee_action_space(RotationFormat.AXIS_ANGLE, GripperFormat.SIGNED)
    pipeline = Pipeline(
        action=[
            RotationFormatAdapter(target=RotationFormat.QUATERNION),
            GripperPolarityAdapter(target=GripperFormat.SIGNED_OPEN_LOW),
        ]
    )
    action = Action.from_array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0])

    result = pipeline.apply_action(action, source=source)

    # quaternion pose widens to 8 floats; the gripper (now at index 7) flips sign.
    assert len(result.values) == 8
    assert np.allclose(result.values[:3], [0.1, -0.2, 0.3])
    assert np.allclose(result.values[3:7], [0.0, 0.0, 0.0, 1.0])
    assert result.values[7] == -1.0


def _single_frame_spec(
    name: str = "agentview", shape: tuple[int, ...] = (1, 1, 3)
) -> ObservationSpace:
    return ObservationSpace(cameras=(Camera(name=name, shape=shape),))


def _frame(value: int, shape: tuple[int, ...] = (1, 1, 3)) -> np.ndarray:
    """A constant-valued frame, so a stacked clip is identifiable by its values."""
    return np.full(shape, value, dtype=np.uint8)


def test_frame_history_stacks_frames_across_steps() -> None:
    # n_frames=4, stride=2 -> buffer depth 7, sampled at offsets (-7, -5, -3, -1)
    # from the newest. Feed frames 0..6 (the buffer then holds exactly 0..6), so
    # the strided sample is frames 0, 2, 4, 6 (newest last).
    source = _single_frame_spec()
    pipeline = Pipeline(observation=[StackFrameHistory(cameras=("agentview",))])
    state = PipelineState()

    last = None
    for value in range(7):
        observation = Observation(sensors={"agentview": _frame(value)})
        last = pipeline.apply_observation(observation, source=source, state=state)

    assert last is not None
    clip = last.sensors["agentview"]
    # A new leading time axis of length n_frames; the inference batch dim stays out.
    assert clip.shape == (4, 1, 1, 3)
    sampled = [int(clip[i, 0, 0, 0]) for i in range(4)]
    assert sampled == [0, 2, 4, 6]


def test_frame_history_first_frame_repeat_pads() -> None:
    # The very first frame of an episode has no history, so the buffer is
    # repeat-padded with it: every sampled offset is that one frame.
    source = _single_frame_spec()
    adapter = StackFrameHistory(cameras=("agentview",))
    pipeline = Pipeline(observation=[adapter])
    state = PipelineState()

    observation = Observation(sensors={"agentview": _frame(5)})
    result = pipeline.apply_observation(observation, source=source, state=state)

    clip = result.sensors["agentview"]
    assert clip.shape == (4, 1, 1, 3)
    assert np.all(clip == 5)


def test_frame_history_produce_lengthens_camera_to_a_clip() -> None:
    adapter = StackFrameHistory(cameras=("agentview",), n_frames=4)
    source = _single_frame_spec(shape=(256, 256, 3))

    produced = adapter.produce(source)

    camera = produced.camera("agentview")
    assert camera is not None
    assert camera.shape == (4, 256, 256, 3)
    # The edge fires at most once per camera: once rank-4, applies is False.
    assert adapter.applies(source)
    assert not adapter.applies(produced)


def test_frame_history_reset_lane_clears_so_next_first_frame_repeat_pads() -> None:
    source = _single_frame_spec()
    pipeline = Pipeline(observation=[StackFrameHistory(cameras=("agentview",))])
    state = PipelineState()

    # Build up some history within episode 1.
    for value in range(4):
        observation = Observation(sensors={"agentview": _frame(value)})
        pipeline.apply_observation(observation, source=source, state=state)

    # Episode boundary: reset clears the buffer for this lane.
    state.reset_lane(DEFAULT_LANE)

    # The first frame of episode 2 must repeat-pad afresh — no tail from episode 1.
    observation = Observation(sensors={"agentview": _frame(9)})
    result = pipeline.apply_observation(observation, source=source, state=state)
    assert np.all(result.sensors["agentview"] == 9)


def test_stateful_adapter_without_state_raises() -> None:
    source = _single_frame_spec()
    pipeline = Pipeline(observation=[StackFrameHistory(cameras=("agentview",))])
    observation = Observation(sensors={"agentview": _frame(0)})

    with pytest.raises(StatefulStateMissing):
        pipeline.apply_observation(observation, source=source)


def test_stateless_pipeline_still_works_with_no_state_arg() -> None:
    # A stateless chain does not need a state argument — the existing call site is unchanged.
    source = ObservationSpace(
        proprioception=Proprioception(
            ee_pose=EEObservationSpec(rotation=RotationFormat.AXIS_ANGLE)
        ),
        cameras=(Camera(name="agentview", shape=(2, 2, 3)),),
    )
    observation = Observation(
        state={"ee_pose": np.arange(6, dtype=np.float32)},
        sensors={"agentview": np.zeros((2, 2, 3), dtype=np.uint8)},
    )
    pipeline = Pipeline(observation=[Rotate180Cameras(cameras=("agentview",))])

    result = pipeline.apply_observation(observation, source=source)

    assert result.sensors["agentview"].shape == (2, 2, 3)
