"""The recorder that writes a replay for each episode of a benchmark.

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
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from manifold.benchmark.simulators import HAND, bounding_box_part
from manifold.core.conventions import RotationFormat, ee_step_layout
from manifold.core.embodiment.action_space import (
    CompositeActionSpace,
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
    NavigatingEvent,
    ReplayCameraPinhole,
    ReplayChannel,
    ReplayChannelKind,
    ReplayFingerPad,
    ReplayLogWriter,
    ReplayMesh,
    ReplaySceneBody,
    ReplaySceneBodyRole,
    ReplayScenePart,
    RerunScalar,
    RerunTab,
    TouchingEvent,
    replay_log_path,
    rerun_panel_line,
    rerun_scalars,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from manifold.benchmark.config import BenchmarkConfig
    from manifold.benchmark.simulators import Simulator
    from manifold.core.conventions import GripperFormat
    from manifold.core.values import Observation
    from manifold.replay import ManifoldEvent, ReplayPose


_logger = logging.getLogger(__name__)


# The height in metres that an object rises above its last rest before a hand grasps it.
_LIFT_METRES = 0.02
# The number of steps after a grasp ends in which a new hold grasps the object again
# without a new lift.
_TOUCH_GAP_STEPS = 2
# The speed in metres a second below which an object counts as still.
_REST_SPEED = 0.05
# The number of steps an object stays still before it is at rest.
_REST_STEPS = 3


def _boxed(body: ReplaySceneBody) -> ReplaySceneBody:
    """`body` with its parts replaced by one box that bounds them, in the body's frame.

    Grading takes each body's size from its parts, and the box keeps that size.
    """
    if not body.parts:
        return body
    posed = [
        ReplayScenePart(
            mesh=ReplayMesh(
                vertices=_rotate(part.orientation, np.asarray(part.mesh.vertices, dtype=np.float64))
                + np.asarray(part.position, dtype=np.float64),
                faces=part.mesh.faces,
            ),
            color=part.color,
        )
        for part in body.parts
    ]
    return replace(body, parts=(bounding_box_part(posed),))


def _rotate(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """`vectors` rotated by an `xyzw` quaternion."""
    x, y, z, w = (float(value) for value in quaternion)
    u = np.array([x, y, z])
    return vectors + 2.0 * np.cross(u, np.cross(u, vectors) + w * vectors)


def _event_key(event: FactEvent) -> tuple[Any, ...]:
    """The key of a fact: the event's type and its fields other than `is_active`."""
    return (type(event), *event.model_dump(exclude={"is_active"}).values())


@dataclass(frozen=True)
class _Gripper:
    """One gripper of the action: its name, its command's place in an action, and its encoding."""

    name: str
    index: int
    encoding: GripperFormat


def _grippers(action: Any) -> tuple[_Gripper, ...]:
    """Each arm's gripper in `action`, named by the arm's position among its arms, from `"0"`.

    An end effector or joint action space holds `arm_count` arms, and each arm's values
    end with its gripper's command. A whole-body action holds the arms of each of its end
    effector and joint groups, in the groups' order.
    """
    if isinstance(action, (EEActionSpace, JointActionSpace)):
        arms = [(action, 0)]
    elif isinstance(action, CompositeActionSpace):
        arms = [
            (group, start)
            for group, (_, start, _) in zip(action.groups, action.group_slices(), strict=True)
            if isinstance(group, (EEActionSpace, JointActionSpace))
        ]
    else:
        arms = []
    grippers = []
    position = 0
    for space, start in arms:
        width = space.per_arm_length()
        for arm in range(space.arm_count):
            if space.gripper is not None:
                index = start + (arm + 1) * width - 1
                grippers.append(_Gripper(str(position), index, space.gripper))
            position += 1
    return tuple(grippers)


def _gripper_pads(
    pads: Sequence[str], grippers: tuple[_Gripper, ...]
) -> dict[str, tuple[str, str]]:
    """The two finger pads of each gripper, by the gripper's name.

    `pads` lists two pads for each gripper, in the grippers' order. One pair serves every
    gripper.

    Raises:
        ValueError: If `pads` lists more than one pair, and not one pair for each gripper.
    """
    pairs = [(pads[index], pads[index + 1]) for index in range(0, len(pads), 2)]
    if len(pairs) == 1:
        return {gripper.name: pairs[0] for gripper in grippers}
    if pairs and len(pairs) != len(grippers):
        raise ValueError(
            f"the config lists {len(pairs)} pairs of finger pads for {len(grippers)} grippers; "
            "list one pair that serves every gripper, or a pair for each gripper"
        )
    return {gripper.name: pair for gripper, pair in zip(grippers, pairs, strict=False)}


def _arm_action(action: Any) -> tuple[EEActionSpace | JointActionSpace, slice] | None:
    """Return the arm part of `action` and where its values sit in an action, or None.

    An end effector or joint action space is all arm. A whole-body action's arm is its first
    end effector or joint group, such as a mobile robot's arm before its base and torso. For
    any other action, such as a unified action, the replay plots no command.
    """
    if isinstance(action, (EEActionSpace, JointActionSpace)):
        return action, slice(0, action.expected_length())
    if isinstance(action, CompositeActionSpace):
        for group, (_, start, stop) in zip(action.groups, action.group_slices(), strict=True):
            if isinstance(group, (EEActionSpace, JointActionSpace)):
                return group, slice(start, stop)
    return None


# The mobile base navigates while it travels faster than this, in metres per second, or
# turns faster than `_NAVIGATING_TURN`, in degrees per second. Measured on RoboCasa365's
# PandaOmron: a parked base moves at most 2.3 mm/s and 0.2 degrees/s when its arm pushes
# on something, a normal drive travels at 0.25 m/s, and the slowest turn measured is
# 1.4 degrees/s.
_NAVIGATING_SPEED = 0.01
_NAVIGATING_TURN = 1.0


def _yaw(quaternion: np.ndarray) -> float:
    """The heading of an `xyzw` quaternion about the vertical axis, in radians."""
    x, y, z, w = quaternion
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


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


def _plot_name(event: FactEvent, end_effector_types: Mapping[str, str]) -> str:
    """The fact event's plot name, such as `bowl_1_on_plate_1` or `cup_hand_touching_left`.

    A gripper event of a declared hand, such as `left`, reads `hand` in place of `gripper`.
    """
    fields = event.model_dump(mode="json", exclude={"is_active", "order", "end_effector"})
    kind = fields.pop("event_type")
    if "objects" in fields:
        return "_".join([kind, *fields["objects"]])
    if "reference" in fields:
        return f"{fields['object']}_{kind}_{fields['reference']}"
    if "gripper" in fields:
        kind = kind.replace("gripper", end_effector_types.get(fields["gripper"], "gripper"))
        return f"{fields['object']}_{kind}_{fields['gripper']}"
    return f"{fields['object']}_{kind}"


class _ReplayRecorder:
    """The replay recorder of an env. It writes a replay log for each episode."""

    def __init__(self, config: BenchmarkConfig, simulator: Simulator | None, env_id: int) -> None:
        self._config = config
        self._simulator = simulator
        self._env_id = env_id
        self._action = config.spec.embodiment.action
        # The type of each declared end effector, such as `hand`, by its name.
        self._end_effector_types = {
            end_effector.name: end_effector.end_effector_type.value
            for end_effector in config.spec.end_effectors or ()
        }
        arm = _arm_action(self._action)
        self._arm = arm[0] if arm is not None else None
        self._arm_slice = arm[1] if arm is not None else slice(0)
        self._grippers = _grippers(self._action)
        self._log: ReplayLogWriter | None = None
        self._step = 0
        self._latest: dict[tuple[Any, ...], FactEvent] = {}
        # The keys of the facts that the first frame charts. The log's header lists the
        # channels at the first frame, so a fact that a hook first returns later has no chart.
        self._charted: frozenset[tuple[Any, ...]] | None = None
        self._warned_uncharted = False
        self._warned_unreadable = False
        # Whether each gripper holds each object, by gripper name and object.
        self._grasped: dict[tuple[str, str], bool] = {}
        # Whether each gripper was last commanded shut, by gripper name.
        self._commanded_shut: dict[str, bool] = {}
        # The base's last position and heading, and whether it is navigating.
        self._base: tuple[np.ndarray, float] | None = None
        self._navigating = False
        self._succeeded = False
        self._previous: tuple[np.ndarray, np.ndarray] | None = None
        self._touching: frozenset[tuple[str, str, str]] = frozenset()
        touch_links = getattr(config.simulator, "touch_links", None) or {}
        # The hand and the thumb flag of each hand link. When the config has hand links,
        # the recorder records each hand's touches and grasps of the task's objects as that
        # hand's, named by the hand, and the hooks record none of the grippers' touches or
        # grasps.
        self._hand_links = {
            link: (part.hand, part.thumb)
            for link, part in touch_links.items()
            if not isinstance(part, str)
        }
        self._hands = sorted({hand for hand, _ in self._hand_links.values()})
        # The finger pads, if the simulator reports their contacts. With pads, a gripper
        # grasps an object only when both of its pads touch it. Without them, it grasps
        # whatever its `GripperTouchingEvent` says it touches.
        self._pads = frozenset(simulator.replay_finger_pads() if simulator is not None else ())
        self._gripper_pads = _gripper_pads(
            simulator.replay_finger_pads() if simulator is not None else (), self._grippers
        )
        # The task's object for each object body in the episode's scene, set by `begin`.
        self._object_bodies: dict[str, str] = {}
        self._objects: frozenset[str] = frozenset()
        # Each object's last position, the number of steps it has been still, and its
        # height at its last rest, which `_hand_grasp_events` uses.
        self._last_position: dict[str, np.ndarray] = {}
        self._still_steps: dict[str, int] = {}
        self._rest_height: dict[str, float] = {}
        # The number of steps since the grasp ended, for each object that a hand let go of.
        self._since_release: dict[str, int] = {}

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
        self._commanded_shut = {}
        self._base = self._base_pose(observation)
        self._navigating = False
        self._succeeded = False
        self._previous = None
        self._touching = frozenset()
        bodies = self._simulator.replay_bodies(self._env_id) if self._simulator is not None else ()
        bodies = tuple(_boxed(body) for body in bodies)
        self._object_bodies = {
            body.name: body.object
            for body in bodies
            if body.role == ReplaySceneBodyRole.OBJECT and body.object is not None
        }
        soft = (
            self._simulator.replay_soft_objects(self._env_id) if self._simulator is not None else ()
        )
        self._objects = frozenset(self._object_bodies.values()) | frozenset(soft)
        self._last_position = {}
        self._still_steps = {}
        self._rest_height = {}
        self._since_release = {}
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
            failure_types=self._config.replay.failure_types,
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
        contacts = self._contacts()
        pad_touches = frozenset((part, thing) for part, thing in contacts if part in self._pads)
        contacts -= pad_touches
        poses = self._simulator.replay_poses(self._env_id) if self._simulator is not None else {}
        recorded = [
            *self._gripper_commanded_events(action),
            *self._navigating_events(observation),
            *self._events_to_write(
                (*events, *self._hand_facts(contacts)), action, self._pinched(pad_touches)
            ),
            *self._hand_grasp_events(contacts, poses),
            *self._touch_events(contacts),
        ]
        if done:
            recorded.append(EpisodeEndedEvent(success=success))
        scalars = self._frame_scalars(observation, action, success=success)
        self._write_step(observation, self._images(observation), scalars, recorded, poses)

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
        poses: dict[str, ReplayPose] | None = None,
    ) -> None:
        if self._log is None:
            return
        lines = [
            line
            for event in events
            if (line := rerun_panel_line(event, self._step, self._end_effector_types)) is not None
        ]
        text = {"instruction": observation.instruction or ""}
        if lines:
            text["events"] = " / ".join(lines)
        simulator = self._simulator
        if poses is None:
            poses = simulator.replay_poses(self._env_id) if simulator is not None else {}
        self._log.write_step(
            poses,
            extrinsics=simulator.replay_extrinsics(self._env_id) if simulator is not None else {},
            images=images,
            scalars={scalar.name: scalar.value for scalar in scalars},
            text=text,
            events=events,
        )

    def _contacts(self) -> frozenset[tuple[str, str]]:
        """Each robot part that touched a body during the last action, with the body."""
        if self._simulator is None:
            return frozenset()
        return frozenset(self._simulator.replay_contacts(self._env_id))

    def _hand_facts(self, contacts: frozenset[tuple[str, str]]) -> tuple[ManifoldEvent, ...]:
        """`GripperTouchingEvent` for each hand and each task object, active while it touches."""
        touched = {
            (self._hand_links[part][0], thing)
            for part, thing in contacts
            if part in self._hand_links and thing in self._objects
        }
        return tuple(
            GripperTouchingEvent(gripper=hand, object=thing, is_active=(hand, thing) in touched)
            for hand in self._hands
            for thing in sorted(self._objects)
        )

    def _hand_grasp_events(
        self, contacts: frozenset[tuple[str, str]], poses: dict[str, ReplayPose]
    ) -> list[ManifoldEvent]:
        """`GraspedEvent` for each hand that started or stopped grasping one of the task's objects.

        The hands grasp an object once the object has risen `_LIFT_METRES` above where it
        last came to rest, while the thumb and another link of one hand touch it, or
        while both hands touch it. The grasp ends on the first step that no hand touches
        the object. When the hands hold the object again within `_TOUCH_GAP_STEPS` steps,
        the object is grasped again without a new lift, and grading merges the two grasps
        into one. While the object is grasped, each hand that touches it grasps it.
        """
        if not self._hands:
            return []
        links: dict[str, set[str]] = {}
        for link, thing in contacts:
            if link in self._hand_links and thing in self._objects:
                links.setdefault(thing, set()).add(link)
        events: list[ManifoldEvent] = []
        for body, thing in self._object_bodies.items():
            pose = poses.get(body)
            if pose is None:
                continue
            touching = links.get(thing, set())
            held = any(self._grasped.get((hand, thing), False) for hand in self._hands)
            height = self._rested_height(
                thing,
                np.asarray(pose.position, dtype=np.float64),
                still_free=not touching and not held,
            )
            # A grasp that ended within the gap starts again at the next hold.
            regrasp = (
                self._holds(touching) and 0 < self._since_release.get(thing, 0) <= _TOUCH_GAP_STEPS
            )
            grasped = (
                bool(touching)
                if held
                else bool(regrasp)
                or (self._holds(touching) and height - self._rest_height[thing] >= _LIFT_METRES)
            )
            if held and not grasped:
                self._since_release[thing] = 1
            elif grasped:
                self._since_release.pop(thing, None)
            elif thing in self._since_release:
                self._since_release[thing] += 1
            holders = {self._hand_links[link][0] for link in touching} if grasped else set()
            for hand in self._hands:
                events += self._grasp_change(hand, thing, grasped=hand in holders)
        return events

    def _rested_height(self, thing: str, position: np.ndarray, *, still_free: bool) -> float:
        """The object's height at `position`, after setting its rest height when it is at rest.

        An object is at rest once its body has moved slower than `_REST_SPEED` for
        `_REST_STEPS` steps and, on the current step, no hand touches or grasps it
        (`still_free`). The first height read for an object is also its first rest height.
        """
        last = self._last_position.get(thing)
        speed = (
            0.0
            if last is None
            else float(np.linalg.norm(position - last)) / self._config.run.step_seconds
        )
        self._last_position[thing] = position
        still = self._still_steps.get(thing, 0) + 1 if speed <= _REST_SPEED else 0
        self._still_steps[thing] = still
        if last is None or (still >= _REST_STEPS and still_free):
            self._rest_height[thing] = float(position[2])
        return float(position[2])

    def _holds(self, links: set[str]) -> bool:
        """Whether `links` hold an object.

        A hold takes the thumb and another link of one hand, or links of both hands.
        """
        hands: dict[str, set[bool]] = {}
        for link in links:
            hand, thumb = self._hand_links[link]
            hands.setdefault(hand, set()).add(thumb)
        return len(hands) >= 2 or any(kinds == {True, False} for kinds in hands.values())

    def _touch_events(self, contacts: frozenset[tuple[str, str]]) -> list[ManifoldEvent]:
        """`TouchingEvent` for each robot part that started or stopped touching a body.

        The result leaves out a hand's touches of the task's objects, because `_hand_facts`
        records them as that hand touching the object. The result records a hand's other
        touches as touches by `HAND`, with the hand's name.
        """
        touching = frozenset(
            (HAND, self._hand_links[part][0], thing)
            if part in self._hand_links
            else (part, "", thing)
            for part, thing in contacts
            if not (part in self._hand_links and thing in self._objects)
        )
        events: list[ManifoldEvent] = [
            *(
                TouchingEvent(
                    object=part, reference=thing, end_effector=hand or None, is_active=True
                )
                for part, hand, thing in sorted(touching - self._touching)
            ),
            *(
                TouchingEvent(
                    object=part, reference=thing, end_effector=hand or None, is_active=False
                )
                for part, hand, thing in sorted(self._touching - touching)
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
        """`GripperCommandedEvent` for each gripper whose command changes direction.

        A gripper's first command of an episode sets its direction and produces no event.
        If `is_gripper_shut` returns None for a command, the direction stays the same.
        """
        if action is None:
            return []
        values = self._action_values(action)
        events: list[ManifoldEvent] = []
        for gripper in self._grippers:
            shut = is_gripper_shut(float(values[gripper.index]), gripper.encoding)
            if shut is None:
                continue
            previous = self._commanded_shut.get(gripper.name)
            self._commanded_shut[gripper.name] = shut
            if previous is not None and previous != shut:
                action_taken = GripperAction.CLOSE if shut else GripperAction.OPEN
                events.append(GripperCommandedEvent(gripper=gripper.name, action=action_taken))
        return events

    def _navigating_events(self, observation: Observation) -> list[NavigatingEvent]:
        """`NavigatingEvent` when the mobile base starts or stops navigating.

        The base navigates while it travels or turns faster than the thresholds above,
        measured between this step's base pose and the last one's. A robot that reports
        no base pose produces no event.
        """
        base, previous = self._base_pose(observation), self._base
        self._base = base
        if base is None or previous is None:
            return []
        seconds = self._config.run.step_seconds
        speed = float(np.linalg.norm(base[0] - previous[0])) / seconds
        turn = (base[1] - previous[1] + np.pi) % (2 * np.pi) - np.pi
        turning = float(abs(np.degrees(turn))) / seconds
        navigating = speed > _NAVIGATING_SPEED or turning > _NAVIGATING_TURN
        if navigating == self._navigating:
            return []
        self._navigating = navigating
        return [NavigatingEvent(is_active=navigating)]

    def _base_pose(self, observation: Observation) -> tuple[np.ndarray, float] | None:
        """The mobile base's position and heading, or None if the robot reports no base pose."""
        spec = self._config.spec.embodiment.proprioception.base_pose
        if spec is None:
            return None
        position, rotation, _ = ee_step_layout(spec.rotation, None)
        values = self._state_values(observation, "base_pose", position + rotation)
        try:
            quaternion = np.asarray(
                convert(values[position:].tolist(), spec.rotation, RotationFormat.QUATERNION)
            )
        except ValueError:
            return values[:position], float("nan")
        return values[:position], _yaw(quaternion)

    def _pinched(self, pad_touches: frozenset[tuple[str, str]]) -> dict[str, frozenset[str]]:
        """The names of the things that both finger pads of each gripper touch, by gripper.

        The result is empty when the simulator reports no finger pads.
        """
        return {
            gripper: frozenset.intersection(
                *(frozenset(thing for part, thing in pad_touches if part == pad) for pad in pads)
            )
            for gripper, pads in self._gripper_pads.items()
        }

    def _events_to_write(
        self,
        events: tuple[ManifoldEvent, ...],
        action: Any,
        pinched: Mapping[str, frozenset[str]] | None = None,
    ) -> list[ManifoldEvent]:
        """The fact events that changed, the other events, and `GraspedEvent`.

        This method writes a `FactEvent` on the first frame if it is active, and later
        when its `is_active` changes. A gripper grasps an object while the policy tells
        it to close and both of its finger pads touch the object, as `pinched` lists by
        gripper. When the simulator reports no finger pads, a gripper grasps an object
        while the policy tells it to close and its `GripperTouchingEvent` for the object
        is active. This method writes a `GraspedEvent` each time a gripper starts or
        stops grasping an object.

        Raises:
            ValueError: If a gripper event names a gripper that the action and the hands
                do not have.
        """
        recorded: list[ManifoldEvent] = []
        for event in events:
            if not isinstance(event, FactEvent):
                recorded.append(event)
                continue
            if isinstance(event, (GraspedEvent, GripperTouchingEvent)):
                self._check_gripper(event.gripper)
            key = _event_key(event)
            last = self._latest.get(key)
            if (last.is_active if last is not None else False) != event.is_active:
                recorded.append(event)
            self._latest[key] = event
        if self._hands:
            # `_hand_grasp_events` records a hand's grasps, because a hand has no gripper
            # command to derive them from.
            return recorded
        for latest in self._latest.values():
            if not isinstance(latest, GripperTouchingEvent):
                continue
            held = (
                latest.object in (pinched or {}).get(latest.gripper, frozenset())
                if self._pads
                else latest.is_active
            )
            grasped = held and self._commanded_shut.get(latest.gripper, False)
            recorded += self._grasp_change(latest.gripper, latest.object, grasped=grasped)
        return recorded

    def _grasp_change(self, gripper: str, thing: str, *, grasped: bool) -> list[ManifoldEvent]:
        """Set whether `gripper` grasps `thing`, and return `GraspedEvent` if that changed."""
        changed = self._grasped.get((gripper, thing), False) != grasped
        self._grasped[(gripper, thing)] = grasped
        return [GraspedEvent(gripper=gripper, object=thing, is_active=grasped)] if changed else []

    def _check_gripper(self, name: str) -> None:
        """Raise if no gripper of the action and no hand has the name `name`."""
        names = [gripper.name for gripper in self._grippers] + self._hands
        if name not in names:
            raise ValueError(
                f"a gripper event names gripper {name!r}, but this benchmark's grippers are "
                f"{names}: each arm's gripper is named by the arm's position in the action, "
                "from '0', and each hand by its name in `simulator.touch_links`"
            )

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
        three values. A joint command plots `joint_cmd/<i>`. Each gripper's command is
        `gripper/<name>`.
        """
        every_value = self._action_values(action)
        values = every_value[self._arm_slice]
        space = self._arm
        scalars = [
            RerunScalar(
                f"gripper/{gripper.name}",
                RerunTab.ACTION,
                float(every_value[gripper.index]),
                RERUN_GRIPPER_ROW,
            )
            for gripper in self._grippers
        ]
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
        observed `joint_pos`. On the high side of a whole-body action's mode slot, a delta
        adds to the arm's previous target rather than its observed position, so the
        commanded line is off on those steps.
        """
        space = self._arm
        scalars: list[RerunScalar] = []
        values = self._action_values(action)[self._arm_slice]
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
                f"facts/{_plot_name(event, self._end_effector_types)}",
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
                _plot_name(event, self._end_effector_types)
                for key, event in self._latest.items()
                if key not in self._charted
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
