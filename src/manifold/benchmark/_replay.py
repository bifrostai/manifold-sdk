"""The replay that the Client writes for each episode of a benchmark.

Each frame contains the config's cameras, the instruction, the events panel, the
scalars and the events. If the config has a simulator, a frame also contains the
scene's poses, the camera poses, the simulator's extra cameras and its scalars.

The shared scalars come from the action and the observation, on three pages:

- `Action`: the command that the policy sent.
- `Robot`: the end effector's pose, velocity and angular velocity, the gripper's
  fingers, and the commanded position next to the observed position.
- `Task`: the step's success, the episode's success so far, whether the gripper
  grasps an object, and each fact event.

This module writes the hooks' events and the events it derives from each step.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from manifold.core.conventions import RotationFormat, ee_step_layout
from manifold.core.embodiment.action_space import (
    EEActionSpace,
    JointActionSpace,
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
    ReplayFingerPad,
    ReplayLogWriter,
    RerunScalar,
    RerunTab,
    TouchingEvent,
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
    """The key of a fact: the event's type and its fields other than `is_active`."""
    return (type(event), *event.model_dump(exclude={"is_active"}).values())


def _arm_action(action: Any) -> EEActionSpace | JointActionSpace | None:
    """Return `action` if it is an end effector or joint action space, or None.

    For any other action, such as a unified action, the replay plots no command.
    """
    if isinstance(action, (EEActionSpace, JointActionSpace)):
        return action
    return None


def _fitted(values: Any, length: int) -> np.ndarray:
    """Return `values` as a flat float array of `length` values.

    Missing values at the end are NaN, and values past `length` are dropped. If
    `values` contains a value that is not a number, all values are NaN.
    """
    fitted = np.full(length, np.nan)
    try:
        given = np.asarray(values, dtype=np.float64).reshape(-1)[:length]
    except (TypeError, ValueError):
        return fitted
    fitted[: len(given)] = given
    return fitted


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
    """The replay recorder of an env. It writes a replay log for each episode."""

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
        # The keys of the facts that the first frame charts. The log's header lists the
        # channels at the first frame, so a fact that a hook first returns later has no chart.
        self._charted: frozenset[tuple[Any, ...]] | None = None
        self._warned_uncharted = False
        self._warned_unreadable = False
        self._grasped: dict[str, bool] = {}
        self._commanded_shut: bool | None = None
        self._succeeded = False
        self._previous: tuple[np.ndarray, np.ndarray] | None = None
        self._touching: frozenset[tuple[str, str]] = frozenset()

    def begin(
        self,
        output_dir: Path,
        episode_idx: int,
        observation: Observation,
        attributes: dict[str, str],
        events: tuple[ManifoldEvent, ...],
    ) -> None:
        """Open the episode's log, and write its scene and its first frame.

        The scalars of the first frame become the log's channels. Later frames write
        the same scalars.
        """
        self._step = 0
        self._latest = {}
        self._charted = None
        self._warned_uncharted = False
        self._warned_unreadable = False
        self._grasped = {}
        self._commanded_shut = None
        self._succeeded = False
        self._previous = None
        self._touching = frozenset()
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
            # If the first frame plots `Action` scalars, the viewer shows that page beside
            # the scene.
            overview_group=RerunTab.ACTION
            if any(scalar.tab is RerunTab.ACTION for scalar in scalars)
            else None,
            cameras=tuple(
                ReplayCameraPinhole.from_camera(sensor)
                for sensor in self._config.spec.sensors
                if sensor.calibration is not None
            ),
            finger_pads=tuple(
                ReplayFingerPad(body=pad.body, offset=np.asarray(pad.offset, dtype=np.float32))
                for pad in self._config.replay.finger_pads
            ),
            step_seconds=self._config.run.step_seconds,
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
        """Write the frame that `action` produced, with `EpisodeEndedEvent` on the last frame."""
        self._step += 1
        self._succeeded = self._succeeded or success
        recorded = [
            *self._gripper_commanded_events(action),
            *self._events_to_write(events, action),
            *self._touch_events(),
        ]
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

    def _touch_events(self) -> list[ManifoldEvent]:
        """`TouchingEvent` for each robot part that started or stopped touching a body."""
        if self._simulator is None:
            return []
        touching = self._simulator.replay_contacts(self._env_id)
        events: list[ManifoldEvent] = [
            *(
                TouchingEvent(object=part, reference=thing, is_active=True)
                for part, thing in sorted(touching - self._touching)
            ),
            *(
                TouchingEvent(object=part, reference=thing, is_active=False)
                for part, thing in sorted(self._touching - touching)
            ),
        ]
        self._touching = touching
        return events

    def _images(self, observation: Observation) -> dict[str, np.ndarray]:
        """The frames of the config's replay cameras and of the simulator's extra cameras.

        The Client turns the frames of the config's cameras upright before this method
        reads them.
        """
        images = {
            name: np.asarray(observation.sensors[name]) for name in self._config.replay.cameras
        }
        if self._simulator is not None:
            images.update(self._simulator.replay_images(self._env_id))
        return images

    def _gripper_commanded_events(self, action: Any) -> list[ManifoldEvent]:
        """`GripperCommandedEvent` if the commanded gripper changes direction.

        The first command of an episode sets the direction and produces no event. If
        `is_gripper_shut` returns None for a command, the direction stays the same.
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

        This method writes a `FactEvent` on the first frame if it is active, and later
        when its `is_active` changes. An object is grasped while its
        `GripperTouchingEvent` is active and the policy commands the gripper shut. This
        method writes a `GraspedEvent` when that state changes.
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
        """The gripper command of the first arm, or None if the action has no gripper."""
        if action is None or self._gripper is None:
            return None
        values = self._action_values(action)
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
        """The scalars of the command that the policy sent, all zero on the first frame.

        An end effector command plots `ee_pos_x` to `ee_pos_z` and the rotation as
        `ee_rot_x` to `ee_rot_z`, or as `ee_rot/<i>` if the rotation has other than
        three values. A joint command plots `joint_cmd/<i>`.
        """
        values = self._action_values(action)
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
        """Return the first arm's end effector position and `xyzw` quaternion.

        This method returns None if the observation has no end effector pose. A
        rotation of all zeros has no orientation. For it, this method returns a
        quaternion of zeros and logs no warning.
        """
        spec = self._config.spec.embodiment.proprioception.ee_pose
        if spec is None:
            return None
        position, rotation, _ = ee_step_layout(spec.rotation, None)
        values = self._state_values(observation, "ee_pose", position + rotation)
        rotation_values = values[position:]
        if not rotation_values.any():
            return values[:position], np.zeros(4)
        try:
            quaternion = np.asarray(
                convert(rotation_values.tolist(), spec.rotation, RotationFormat.QUATERNION)
            )
        except ValueError:
            # A rotation with NaN, or with a norm too small to normalise, has no
            # orientation.
            self._warn_unreadable("ee_pose")
            quaternion = np.full(4, np.nan)
        return values[:position], quaternion

    def _robot_scalars(
        self, observation: Observation, action: Any, ee: tuple[np.ndarray, np.ndarray] | None
    ) -> list[RerunScalar]:
        """The scalars of the end effector, the gripper's fingers and the arm's tracking."""
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
        """The gripper values of the first arm in the observation, or an empty array."""
        proprioception = self._config.spec.embodiment.proprioception
        ee_spec = proprioception.ee_pose
        if ee_spec is not None and ee_spec.gripper is not None:
            position, rotation, _ = ee_step_layout(ee_spec.rotation, None)
            start = position + rotation
            values = self._state_values(observation, "ee_pose", start + ee_spec.gripper.dim)
            return values[start:]
        joint_spec = proprioception.joint_pos
        if joint_spec is not None and joint_spec.gripper is not None:
            arm = joint_spec.dof // joint_spec.arm_count
            return self._state_values(observation, "joint_pos", arm + 1)[arm:]
        return np.zeros(0)

    def _action_values(self, action: Any) -> np.ndarray:
        """The action's values at the length of the action space, or zeros if `action` is None."""
        length = self._action.expected_length()
        if action is None:
            return np.zeros(length)
        values = _fitted(action.values, length)
        if np.asarray(action.values).size != length or not np.isfinite(values).all():
            self._warn_unreadable("the action")
        return values

    def _state_values(self, observation: Observation, name: str, length: int) -> np.ndarray:
        """The observation's state `name` at `length` values, with NaN for missing values."""
        given = observation.state.get(name)
        values = _fitted(given if given is not None else (), length)
        if given is None or np.asarray(given).size < length or not np.isfinite(values).all():
            self._warn_unreadable(name)
        return values

    def _warn_unreadable(self, name: str) -> None:
        """Log a warning the first time that a value cannot be charted in this episode."""
        if self._warned_unreadable:
            return
        self._warned_unreadable = True
        _logger.warning(
            "%s is missing, too short or not finite, so the replay charts NaN for it", name
        )

    def _tracking_scalars(
        self, observation: Observation, action: Any, ee: tuple[np.ndarray, np.ndarray] | None
    ) -> list[RerunScalar]:
        """The commanded position of the arm next to its observed position.

        For an end effector command, the commanded position is the command. If the
        command is a delta, the commanded position is the observed position plus the
        command. A joint command follows the same rule for joint angles, against the
        observed `joint_pos`.
        """
        space = self._arm
        scalars: list[RerunScalar] = []
        values = self._action_values(action)
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
        elif (
            isinstance(space, JointActionSpace)
            and self._config.spec.embodiment.proprioception.joint_pos is not None
        ):
            arm = space.dof // space.arm_count
            angles = self._state_values(observation, "joint_pos", arm)
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
        """The step's success, the episode's success so far, the grasp, and each fact.

        If a hook first returns a fact after the first frame, the replay writes the
        fact as an event but has no chart for it. The log's header lists the charts at
        the first frame.
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
