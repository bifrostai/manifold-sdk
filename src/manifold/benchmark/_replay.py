"""The replay that the Client writes for each episode, the same way for every benchmark.

Each frame holds the config's cameras, the instruction, the events panel, the
scalars and the events, and, when the config has a simulator, the scene's poses,
the camera poses, the simulator's other cameras and its scalars.

The scalars that every benchmark has come from the action and the observation:

- `Action`: the command that the policy sent. The Rerun viewer shows this page
  beside the scene, with the gripper on the top row and the arm below it.
- `Robot`: the end effector's pose, velocity and angular velocity, the gripper's
  fingers, and the command against where the arm went.
- `Task`: success, whether the episode has succeeded so far, whether the gripper
  grasps an object, and each fact event.

The events are the hooks' events and the ones that this module derives:
`EpisodeStartedEvent`, `EpisodeEndedEvent`, `GripperCommandedEvent` when the commanded gripper
changes direction, and `GraspedEvent` while the gripper touches an object and is
commanded shut. The hooks return each `FactEvent` at every step, and this module
records it only when its `is_active` changes. It records any other event as it comes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from manifold.core.conventions import RotationFormat, ee_step_layout
from manifold.core.embodiment.action_space import (
    EEActionSpace,
    JointActionSpace,
    UnifiedActionSpace,
)
from manifold.lib.gripper import is_gripper_shut
from manifold.lib.rotation import convert
from manifold.replay import (
    RERUN_ARM_ROW,
    RERUN_GRIPPER_ROW,
    RERUN_ROTATION_ROW,
    EpisodeEndedEvent,
    EpisodeStartedEvent,
    FactEvent,
    GraspedEvent,
    GripperAction,
    GripperCommandedEvent,
    GripperTouchingEvent,
    ReplayCameraPinhole,
    ReplayChannel,
    ReplayChannelKind,
    ReplayLogWriter,
    RerunScalar,
    RerunTab,
    replay_log_path,
    rerun_panel_line,
    rerun_scalars,
)

if TYPE_CHECKING:
    from pathlib import Path

    from manifold.benchmark.config import BenchmarkConfig
    from manifold.benchmark.simulators import Simulator
    from manifold.core.values import Observation
    from manifold.replay import ManifoldEvent


_logger = logging.getLogger(__name__)


def _event_key(event: FactEvent) -> tuple[Any, ...]:
    """The fact event without its `is_active`, the same at every step."""
    return (type(event), *event.model_dump(exclude={"is_active"}).values())


def _arm_action(action: Any) -> EEActionSpace | JointActionSpace | None:
    """The arm's part of an action space, which starts at the action's first value.

    An end effector or joint action is all arm. A unified action's arm is its payload,
    which fills its leading slots. Any other action has no arm that the replay reads,
    so it gives None.
    """
    if isinstance(action, (EEActionSpace, JointActionSpace)):
        return action
    if isinstance(action, UnifiedActionSpace):
        return action.payload
    return None


def _plot_name(event: FactEvent) -> str:
    """The fact event's plot name, such as `bowl_1_on_plate_1`."""
    fields = event.model_dump(mode="json", exclude={"is_active", "order"})
    kind = fields.pop("event_type")
    if "objects" in fields:
        return "_".join([kind, *fields["objects"]])
    if "reference" in fields:
        return f"{fields['object']}_{kind}_{fields['reference']}"
    return f"{fields['object']}_{kind}"


class _ReplayRecorder:
    """Records the replay of one env's episodes, and writes a log for each episode."""

    def __init__(self, config: BenchmarkConfig, simulator: Simulator | None, env_id: int) -> None:
        self._config = config
        self._simulator = simulator
        self._env_id = env_id
        self._action = config.spec.embodiment.action
        self._arm = _arm_action(self._action)
        self._gripper = self._arm.gripper if self._arm is not None else None
        self._log: ReplayLogWriter | None = None
        self._step = 0
        self._latest: dict[tuple[Any, ...], FactEvent] = {}
        # The facts that the first frame charted. The log's header lists its channels at
        # the first frame, so a fact that a hook first returns later has no chart.
        self._charted: frozenset[tuple[Any, ...]] | None = None
        self._warned_uncharted = False
        self._grasped: dict[str, bool] = {}
        self._commanded_shut: bool | None = None
        self._succeeded = False
        self._previous: tuple[np.ndarray, np.ndarray] | None = None

    def begin(
        self,
        output_dir: Path,
        episode_idx: int,
        observation: Observation,
        attributes: dict[str, str],
        events: tuple[ManifoldEvent, ...],
    ) -> None:
        """Open the episode's log, and write its scene and its first frame.

        The first frame decides the log's channels: the scalars that it holds are the
        ones that every later frame holds.
        """
        self._step = 0
        self._latest = {}
        self._charted = None
        self._warned_uncharted = False
        self._grasped = {}
        self._commanded_shut = None
        self._succeeded = False
        self._previous = None
        bodies = self._simulator.replay_bodies(self._env_id) if self._simulator is not None else ()
        recorded = [
            EpisodeStartedEvent(instruction=observation.instruction or "", attributes=attributes),
            *self._events_to_write(events, None),
        ]
        scalars = self._frame_scalars(observation, None, success=False)
        self._charted = frozenset(self._latest)
        images = self._images(observation)
        channels = [ReplayChannel(name=name, kind=ReplayChannelKind.IMAGE) for name in images]
        channels += [
            ReplayChannel(name="instruction", kind=ReplayChannelKind.TEXT),
            ReplayChannel(name="events", kind=ReplayChannelKind.TEXT),
        ]
        channels += [
            ReplayChannel(
                name=scalar.name, kind=ReplayChannelKind.SCALAR, group=scalar.tab, row=scalar.row
            )
            for scalar in scalars
        ]
        self._log = ReplayLogWriter(
            replay_log_path(output_dir, episode_idx),
            episode_idx=episode_idx,
            channels=channels,
            # The viewer shows the Action page beside the scene, if the frame plots one.
            overview_group=RerunTab.ACTION
            if any(scalar.tab is RerunTab.ACTION for scalar in scalars)
            else None,
            cameras=tuple(
                ReplayCameraPinhole.from_camera(sensor)
                for sensor in self._config.spec.sensors
                if sensor.calibration is not None
            ),
        )
        if bodies:
            self._log.write_scene(bodies)
        self._write_step(observation, images, scalars, recorded)

    def record(
        self,
        action: Any,
        observation: Observation,
        events: tuple[ManifoldEvent, ...],
        *,
        success: bool,
        done: bool,
    ) -> None:
        """Write the frame that `action` produced, and the end of the episode on its last."""
        self._step += 1
        self._succeeded = self._succeeded or success
        recorded = [*self._gripper_commanded_events(action), *self._events_to_write(events, action)]
        if done:
            recorded.append(EpisodeEndedEvent(success=success))
        scalars = self._frame_scalars(observation, action, success=success)
        self._write_step(observation, self._images(observation), scalars, recorded)

    def end(self) -> None:
        """Close the log."""
        if self._log is not None:
            self._log.close()
        self._log = None

    def _write_step(
        self,
        observation: Observation,
        images: dict[str, np.ndarray],
        scalars: list[RerunScalar],
        events: list[ManifoldEvent],
    ) -> None:
        if self._log is None:
            return
        lines = [
            line for event in events if (line := rerun_panel_line(event, self._step)) is not None
        ]
        text = {"instruction": observation.instruction or ""}
        if lines:
            text["events"] = " / ".join(lines)
        simulator = self._simulator
        self._log.write_step(
            simulator.replay_poses(self._env_id) if simulator is not None else {},
            extrinsics=simulator.replay_extrinsics(self._env_id) if simulator is not None else {},
            images=images,
            scalars={scalar.name: scalar.value for scalar in scalars},
            text=text,
            events=events,
        )

    def _images(self, observation: Observation) -> dict[str, np.ndarray]:
        """The config's cameras, which the Client has turned upright, and the simulator's."""
        images = {
            name: np.asarray(observation.sensors[name]) for name in self._config.replay.cameras
        }
        if self._simulator is not None:
            images.update(self._simulator.replay_images(self._env_id))
        return images

    def _gripper_commanded_events(self, action: Any) -> list[ManifoldEvent]:
        """`GripperCommandedEvent` when the commanded gripper changes direction.

        The first command of an episode sets the direction and records no event. A
        command that `is_gripper_shut` calls neither keeps the previous direction.
        """
        value = self._gripper_command(action)
        if value is None or self._gripper is None:
            return []
        shut = is_gripper_shut(value, self._gripper)
        if shut is None:
            return []
        previous, self._commanded_shut = self._commanded_shut, shut
        if previous is None or previous == shut:
            return []
        return [GripperCommandedEvent(action=GripperAction.CLOSE if shut else GripperAction.OPEN)]

    def _events_to_write(
        self, events: tuple[ManifoldEvent, ...], action: Any
    ) -> list[ManifoldEvent]:
        """The fact events that changed, the other events, and `GraspedEvent`.

        A `FactEvent` is recorded on the first frame when it is active, and later
        when its `is_active` changes. `GraspedEvent` is derived from
        `GripperTouchingEvent` while the gripper is commanded shut.
        """
        recorded: list[ManifoldEvent] = []
        for event in events:
            if not isinstance(event, FactEvent):
                recorded.append(event)
                continue
            key = _event_key(event)
            last = self._latest.get(key)
            if (last.is_active if last is not None else False) != event.is_active:
                recorded.append(event)
            self._latest[key] = event
        shut = self._commanded_shut is True
        for latest in self._latest.values():
            if not isinstance(latest, GripperTouchingEvent):
                continue
            grasped = latest.is_active and shut
            if self._grasped.get(latest.object, False) != grasped:
                recorded.append(GraspedEvent(object=latest.object, is_active=grasped))
            self._grasped[latest.object] = grasped
        return recorded

    def _gripper_command(self, action: Any) -> float | None:
        """The first arm's commanded gripper, or None for an action without one."""
        if action is None or self._gripper is None:
            return None
        values = np.asarray(action.values, dtype=np.float64).reshape(-1)
        if isinstance(self._arm, EEActionSpace):
            position, rotation, _ = ee_step_layout(self._arm.rotation, self._arm.gripper)
            return float(values[position + rotation])
        if isinstance(self._arm, JointActionSpace):
            return float(values[self._arm.dof // self._arm.arm_count])
        return None

    def _frame_scalars(
        self, observation: Observation, action: Any, *, success: bool
    ) -> list[RerunScalar]:
        """The frame's scalars: the shared ones, then the simulator's."""
        ee = self._ee_pose(observation)
        scalars = [
            *self._action_scalars(action),
            *self._robot_scalars(observation, action, ee),
            *self._task_scalars(success),
        ]
        if self._simulator is not None:
            scalars += self._simulator.replay_scalars(
                self._env_id, ee[0] if ee is not None else None
            )
        if ee is not None:
            self._previous = ee
        return scalars

    def _action_scalars(self, action: Any) -> list[RerunScalar]:
        """The command that the policy sent, zero on the first frame.

        An end effector command plots `ee_pos_x` to `ee_pos_z` and the rotation as
        `ee_rot_x` to `ee_rot_z`, or `ee_rot/<i>` for a rotation that is not three
        values. A joint command plots `joint_cmd/<i>`.
        """
        values = np.zeros(self._action.expected_length())
        if action is not None:
            given = np.asarray(action.values, dtype=np.float64).reshape(-1)[: len(values)]
            values[: len(given)] = given
        space = self._arm
        scalars = []
        if self._gripper is not None:
            command = self._gripper_command(action)
            scalars.append(
                RerunScalar(
                    "gripper",
                    RerunTab.ACTION,
                    command if command is not None else 0.0,
                    RERUN_GRIPPER_ROW,
                )
            )
        if isinstance(space, EEActionSpace):
            position, rotation, _ = ee_step_layout(space.rotation, space.gripper)
            scalars += [
                RerunScalar(f"ee_pos_{axis}", RerunTab.ACTION, float(values[index]), RERUN_ARM_ROW)
                for index, axis in enumerate("xyz")
            ]
            names = (
                [f"ee_rot_{axis}" for axis in "xyz"]
                if rotation == 3
                else [f"ee_rot/{index}" for index in range(rotation)]
            )
            scalars += [
                RerunScalar(
                    name, RerunTab.ACTION, float(values[position + index]), RERUN_ROTATION_ROW
                )
                for index, name in enumerate(names)
            ]
        elif isinstance(space, JointActionSpace):
            scalars += [
                RerunScalar(
                    f"joint_cmd/{index}", RerunTab.ACTION, float(values[index]), RERUN_ARM_ROW
                )
                for index in range(space.dof // space.arm_count)
            ]
        return scalars

    def _ee_pose(self, observation: Observation) -> tuple[np.ndarray, np.ndarray] | None:
        """The first arm's end effector position and `xyzw` quaternion, if observed."""
        spec = self._config.spec.embodiment.proprioception.ee_pose
        if spec is None or "ee_pose" not in observation.state:
            return None
        values = np.asarray(observation.state["ee_pose"], dtype=np.float64).reshape(-1)
        position, rotation, _ = ee_step_layout(spec.rotation, None)
        quaternion = np.asarray(
            convert(
                values[position : position + rotation].tolist(),
                spec.rotation,
                RotationFormat.QUATERNION,
            )
        )
        return values[:position].copy(), quaternion

    def _robot_scalars(
        self, observation: Observation, action: Any, ee: tuple[np.ndarray, np.ndarray] | None
    ) -> list[RerunScalar]:
        """The end effector, the gripper's fingers, and the command against the arm."""
        scalars: list[RerunScalar] = []
        if ee is not None:
            position, quaternion = ee
            scalars += rerun_scalars("ee_pos", RerunTab.ROBOT, position)
            scalars += rerun_scalars("ee_quat", RerunTab.ROBOT, quaternion)
            scalars += rerun_scalars("ee_vel", RerunTab.ROBOT, self._velocity(position))
            scalars += rerun_scalars(
                "ee_angvel", RerunTab.ROBOT, self._angular_velocity(quaternion)
            )
        scalars += rerun_scalars("gripper_qpos", RerunTab.ROBOT, self._fingers(observation))
        scalars += self._tracking_scalars(observation, action, ee)
        return scalars

    def _fingers(self, observation: Observation) -> np.ndarray:
        """The first arm's gripper values in the observation, or none."""
        proprioception = self._config.spec.embodiment.proprioception
        ee_spec = proprioception.ee_pose
        if ee_spec is not None and ee_spec.gripper is not None and "ee_pose" in observation.state:
            values = np.asarray(observation.state["ee_pose"], dtype=np.float64).reshape(-1)
            position, rotation, _ = ee_step_layout(ee_spec.rotation, None)
            return values[position + rotation : position + rotation + ee_spec.gripper.dim]
        joint_spec = proprioception.joint_pos
        if (
            joint_spec is not None
            and joint_spec.gripper is not None
            and "joint_pos" in observation.state
        ):
            values = np.asarray(observation.state["joint_pos"], dtype=np.float64).reshape(-1)
            arm = joint_spec.dof // joint_spec.arm_count
            return values[arm : arm + 1]
        return np.zeros(0)

    def _tracking_scalars(
        self, observation: Observation, action: Any, ee: tuple[np.ndarray, np.ndarray] | None
    ) -> list[RerunScalar]:
        """Where the command asked the arm to be, against where it is.

        An end effector command asks for a position, or for the observed position
        plus the command when it is a delta. A joint command asks for joint angles,
        compared with the observed `joint_pos`.
        """
        space = self._arm
        scalars: list[RerunScalar] = []
        values = (
            np.asarray(action.values, dtype=np.float64).reshape(-1)
            if action is not None
            else np.zeros(self._action.expected_length())
        )
        if isinstance(space, EEActionSpace) and ee is not None:
            wanted = ee[0] + values[:3] if space.delta else values[:3]
            for index, axis in enumerate("xyz"):
                scalars.append(
                    RerunScalar(
                        f"track_{axis}/commanded", RerunTab.ROBOT, float(wanted[index]), None
                    )
                )
                scalars.append(
                    RerunScalar(f"track_{axis}/observed", RerunTab.ROBOT, float(ee[0][index]), None)
                )
        elif isinstance(space, JointActionSpace) and "joint_pos" in observation.state:
            arm = space.dof // space.arm_count
            angles = np.asarray(observation.state["joint_pos"], dtype=np.float64).reshape(-1)[:arm]
            wanted = angles + values[:arm] if space.delta else values[:arm]
            for index in range(arm):
                scalars.append(
                    RerunScalar(
                        f"track_j{index}/commanded", RerunTab.ROBOT, float(wanted[index]), None
                    )
                )
                scalars.append(
                    RerunScalar(
                        f"track_j{index}/observed", RerunTab.ROBOT, float(angles[index]), None
                    )
                )
        return scalars

    def _task_scalars(self, success: bool) -> list[RerunScalar]:
        """Success, success so far, whether the gripper grasps anything, and each fact event.

        A fact that a hook first returns after the first frame is recorded as an event
        but has no chart, because the log's header lists its charts at the first frame.
        """
        scalars = [
            RerunScalar("success", RerunTab.TASK, float(success), None),
            RerunScalar("episode_success", RerunTab.TASK, float(self._succeeded or success), None),
            RerunScalar("grasp", RerunTab.TASK, float(any(self._grasped.values())), None),
        ]
        scalars += [
            RerunScalar(
                f"facts/{_plot_name(event)}",
                RerunTab.TASK,
                float(event.is_active),
                None,
            )
            for key, event in self._latest.items()
            if self._charted is None or key in self._charted
        ]
        uncharted = (
            []
            if self._charted is None
            else [
                _plot_name(event) for key, event in self._latest.items() if key not in self._charted
            ]
        )
        if uncharted and not self._warned_uncharted:
            self._warned_uncharted = True
            _logger.warning(
                "the hooks first returned the facts %s after the reset, so they have no chart; "
                "return every fact from the reset on",
                uncharted,
            )
        return scalars

    def _velocity(self, position: np.ndarray) -> np.ndarray:
        """The end effector's velocity, from its position on the last frame."""
        if self._previous is None:
            return np.zeros(3)
        return (position - self._previous[0]) / self._config.run.step_seconds

    def _angular_velocity(self, quaternion: np.ndarray) -> np.ndarray:
        """The end effector's angular velocity, from its rotation since the last frame.

        For two unit quaternions a step apart, the vector part of
        `q_now * conj(q_last)` is half the rotation vector.
        """
        if self._previous is None:
            return np.zeros(3)
        x0, y0, z0, w0 = (float(v) for v in self._previous[1][:4])
        x1, y1, z1, w1 = (float(v) for v in quaternion[:4])
        delta = np.array(
            [
                w1 * -x0 + x1 * w0 + y1 * -z0 - z1 * -y0,
                w1 * -y0 - x1 * -z0 + y1 * w0 + z1 * -x0,
                w1 * -z0 + x1 * -y0 - y1 * -x0 + z1 * w0,
            ]
        )
        return 2.0 * delta / self._config.run.step_seconds
