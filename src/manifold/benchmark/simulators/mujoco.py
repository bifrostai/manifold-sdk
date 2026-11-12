"""The MuJoCo simulator, which reads the scene, poses and scalars of a MuJoCo model.

The replay scene contains the geoms in the groups that the config's `geom_groups`
lists, grouped by body. The arm's links and the gripper's fingers are bodies too.
The group of a geom determines whether the cameras draw it. Its collision flags do
not determine this, because some drawn geoms also collide. robosuite draws groups 1
and 2 and puts its collision geoms in group 0, so a robosuite benchmark lists
`[1, 2]`.

The config's prefixes sort the model's joints and bodies into the arm, the gripper,
the rest of the robot, and the objects. robosuite uses the prefixes `robot0_`,
`gripper0_` and `mount0_`.

The simulator reads the model through a handle, such as robosuite's `MjSim`. A
handle is an object with `model` and `data`. This module imports `mujoco` inside
the functions that need it, so the SDK installs and imports without it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from operator import attrgetter
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from manifold.benchmark._task_names import TaskBody, TaskSite
from manifold.benchmark.simulators import (
    ARM,
    GRIPPER,
    TABLE,
    Simulator,
)
from manifold.core.conventions import CameraAxes
from manifold.lib.compat import assert_never
from manifold.lib.rotation import convert_camera_axes, wxyz_to_xyzw, xyzw_from_rotation_matrix
from manifold.replay import (
    ReplayMesh,
    ReplayPose,
    ReplaySceneBody,
    ReplaySceneBodyRole,
    ReplayScenePart,
    RerunScalar,
    RerunTab,
    rerun_scalars,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from manifold.benchmark._task_names import TaskTarget


_logger = logging.getLogger(__name__)


# The smallest force, in newtons, that counts as a touch.
_TOUCH_NEWTONS = 0.1


@dataclass(frozen=True)
class _MujocoEpisodeCache:
    """Values that `replay_bodies` reads from the model at the start of an episode.

    The model stays the same during an episode, so the other replay methods use these
    values at each step instead of searching the model again.

    `body_ids` contains the ids of the bodies with a drawn geom, in the order of the
    scene's bodies.

    The joint fields are integer arrays of indices into MuJoCo's flat state arrays.
    Such an array selects a part's values in a single read, such as
    `data.qpos[cache.arm_qpos]` for the arm's joint angles:

    - `arm_qpos`, `gripper_qpos` and `object_qpos` index `data.qpos`, which contains
      the joints' positions.
    - `arm_dof`, `gripper_dof` and `object_dof` index `data.qvel`, `data.qacc`
      and `data.qfrc_actuator`, which contain the joints' velocities, accelerations
      and forces. MuJoCo calls each of these values a degree of freedom, or dof.

    `objects` maps the name of each object's body to the body's id. An object is a
    body with a joint outside the robot. `contacts` maps the name of each body whose
    contact force the replay plots to the body's id.

    `parts` maps each body of the arm and of the gripper to `ARM` or `GRIPPER`.
    `things` maps each body that the task names to the name and to whether the body is
    an object, for the bodies that a robot part can touch. `sites` maps the scene name
    of each site that the task names to the id of the body the site moves with.
    """

    body_ids: tuple[int, ...]
    arm_qpos: np.ndarray
    arm_dof: np.ndarray
    gripper_qpos: np.ndarray
    gripper_dof: np.ndarray
    object_qpos: np.ndarray
    object_dof: np.ndarray
    objects: dict[str, int]
    contacts: dict[str, int]
    parts: dict[int, str]
    things: dict[int, tuple[str, bool]]
    sites: dict[str, int]


class BenchmarkMujocoSimulatorConfig(BaseModel):
    """The MuJoCo simulator's values from the benchmark's config.

    `path` is the attribute path from the env that `on_load` returns to the MuJoCo
    handle, such as `sim` for robosuite. `cameras` maps each spec camera that the
    replay poses to the camera's name in the model. `geom_groups` lists the geom
    groups that the cameras draw. `arm_prefix` and `gripper_prefix` select the arm's
    and the gripper's joints. `robot_prefixes` selects the bodies of the robot, and
    `contact_prefixes` selects the bodies whose contact force the replay plots.
    `object_body_suffix` is what the model adds to an object's name to name its root
    body, such as `_main` for robosuite.
    """

    # A config states every value, so an unknown key is an error.
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["mujoco"]
    path: str
    cameras: dict[str, str]
    geom_groups: tuple[int, ...]
    arm_prefix: str
    gripper_prefix: str
    robot_prefixes: tuple[str, ...]
    contact_prefixes: tuple[str, ...]
    object_body_suffix: str = Field(min_length=1)


class MujocoSimulator(Simulator):
    """Simulator that reads the scene, poses and scalars from a MuJoCo handle for each env."""

    def __init__(self, config: BenchmarkMujocoSimulatorConfig) -> None:
        self._config = config
        # Map each env id to the env that `on_load` returned.
        self._envs: dict[int, Any] = {}
        self._caches: dict[int, _MujocoEpisodeCache] = {}

    @property
    def supported_versions(self) -> dict[str, str]:
        """The version ranges of MuJoCo and robosuite that this simulator reads."""
        return {"mujoco": ">=3.2,<3.9", "robosuite": ">=1.4,<1.6"}

    def start(self) -> None:
        """Do nothing, because MuJoCo runs in the benchmark's process."""

    def stop(self) -> None:
        """Do nothing, because MuJoCo runs in the benchmark's process."""

    def attach(self, env: Any, env_ids: Sequence[int]) -> None:
        """Store `env` as env `env_ids[0]`.

        Each read looks up the handle at the config's path in `env`, because a
        framework can replace the handle. robosuite builds a new `MjSim` on each reset.

        Raises:
            ValueError: If `env_ids` contains more than one id, because a MuJoCo
                handle contains one env.
        """
        if len(env_ids) != 1:
            raise ValueError(f"a MuJoCo handle holds one env, not {len(env_ids)}")
        self._envs[env_ids[0]] = env

    def post_step(self) -> None:
        """Keep nothing, because each replay method reads the env when it is called."""

    def replay_bodies(
        self, env_id: int, task_names: Mapping[str, TaskTarget]
    ) -> tuple[ReplaySceneBody, ...]:
        """Each body with a drawn geom, and each site that `task_names` refers to.

        A task can lay out its objects again on each reset. This method therefore
        also reads the joints, the objects and the contact bodies again, and caches
        them for the episode. A body that `task_names` refers to takes the task's
        name, and so do the parts of a named fixture: the bodies under it in the
        model's body tree.

        Raises:
            ValueError: If `task_names` refers to a body or a site that the model
                lacks, to the world or the robot, or to a site of a shape that the
                scene does not draw, or if it gives one body two names or gives a body
                a name that another body has.
        """
        sim = self._sim(env_id)
        model = sim.model
        drawn = frozenset(self._config.geom_groups)
        parts: dict[int, list[ReplayScenePart]] = {}
        skipped: dict[int, int] = {}
        means: dict[int, tuple[float, float, float] | None] = {}
        for geom_id in range(model.ngeom):
            if int(model.geom_group[geom_id]) not in drawn:
                continue
            mesh = self._geom_mesh(model, geom_id)
            if mesh is None:
                kind = int(model.geom_type[geom_id])
                skipped[kind] = skipped.get(kind, 0) + 1
                continue
            parts.setdefault(int(model.geom_bodyid[geom_id]), []).append(
                ReplayScenePart(
                    mesh=mesh,
                    position=np.asarray(model.geom_pos[geom_id], dtype=np.float32),
                    orientation=wxyz_to_xyzw(model.geom_quat[geom_id]).astype(np.float32),
                    color=self._geom_colour(model, geom_id, means),
                )
            )
        self._log_skipped_geoms(skipped)
        qpos, dof = self._joint_indices(model)
        object_ids = self._object_bodies(model)
        objects = frozenset(object_ids.values())
        names = [self._body_name(model, body_id) for body_id in range(int(model.nbody))]
        owners = self._owners(names)
        labels = {
            body_id: (
                self._body_role(name, body_id, objects),
                self._object_name(name, body_id, objects, owners),
            )
            for body_id, name in enumerate(names)
        }
        labels, named_by, parents = self._task_labels(model, names, labels, task_names)
        sites = self._task_sites(model, names, labels, named_by, task_names)
        self._check_task_names_distinct(labels, named_by, sites)
        cache = _MujocoEpisodeCache(
            body_ids=tuple(sorted(parts)),
            arm_qpos=qpos["arm"],
            arm_dof=dof["arm"],
            gripper_qpos=qpos["gripper"],
            gripper_dof=dof["gripper"],
            object_qpos=qpos["objects"],
            object_dof=dof["objects"],
            objects=object_ids,
            contacts={
                name: body_id
                for body_id in range(int(model.nbody))
                if (name := self._body_name(model, body_id)).startswith(
                    self._config.contact_prefixes
                )
            },
            parts=self._robot_parts(model),
            things={
                body_id: (name, role is ReplaySceneBodyRole.OBJECT)
                for body_id, (role, name) in labels.items()
                if name is not None
            },
            sites={body.name: parent_id for body, parent_id in sites},
        )
        self._caches[env_id] = cache
        return (
            *(
                ReplaySceneBody(
                    name=self._body_name(model, body_id),
                    parts=tuple(parts[body_id]),
                    role=labels[body_id][0],
                    object=labels[body_id][1],
                    parent=parents.get(body_id),
                )
                for body_id in cache.body_ids
            ),
            *(body for body, _ in sites),
        )

    def replay_poses(self, env_id: int) -> dict[str, ReplayPose]:
        """Each scene body's world pose, from `data.xpos` and `data.xquat`.

        MuJoCo's `MjData`, dm_control's physics and robosuite's wrapper all have these
        two attributes.
        """
        sim = self._sim(env_id)
        data = sim.data
        cache = self._caches[env_id]
        poses = {
            self._body_name(sim.model, body_id): ReplayPose(
                position=np.asarray(data.xpos[body_id], dtype=np.float32),
                orientation=wxyz_to_xyzw(data.xquat[body_id]).astype(np.float32),
            )
            for body_id in cache.body_ids
        }
        # A site's part is posed in its body's frame, so the site moves with the body.
        for site_name, body_id in cache.sites.items():
            poses[site_name] = ReplayPose(
                position=np.asarray(data.xpos[body_id], dtype=np.float32),
                orientation=wxyz_to_xyzw(data.xquat[body_id]).astype(np.float32),
            )
        return poses

    def replay_extrinsics(self, env_id: int) -> dict[str, ReplayPose]:
        """Each camera that the config maps, as its camera-to-world pose in OpenCV axes.

        `cam_xpos` and `cam_xmat` give the camera-to-world pose in OpenGL axes.
        """
        sim = self._sim(env_id)
        model, data = sim.model, sim.data
        posed = {}
        for name, camera in self._config.cameras.items():
            camera_id = int(model.camera(camera).id)
            rotation = convert_camera_axes(
                np.asarray(data.cam_xmat[camera_id], dtype=float),
                CameraAxes.OPENGL,
                CameraAxes.OPENCV,
            )
            posed[name] = ReplayPose(
                position=np.asarray(data.cam_xpos[camera_id], dtype=np.float32),
                orientation=xyzw_from_rotation_matrix(rotation),
            )
        return posed

    def replay_images(self, env_id: int) -> dict[str, np.ndarray]:
        """An empty dict, because a MuJoCo benchmark sends the policy each camera of its replay."""
        return {}

    def replay_contacts(self, env_id: int) -> frozenset[tuple[str, str]]:
        """Each robot part that touches a named body at the last physics step, with the body.

        A touch that starts and ends between two actions is missed.
        """
        sim = self._sim(env_id)
        cache = self._caches[env_id]
        model = getattr(sim.model, "_model", sim.model)
        data = getattr(sim.data, "_data", sim.data)
        geom_body = np.asarray(model.geom_bodyid, dtype=np.int64)
        touching = set()
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            first, second = int(geom_body[contact.geom1]), int(geom_body[contact.geom2])
            for mine, other in ((first, second), (second, first)):
                part = cache.parts.get(mine)
                thing = cache.things.get(other)
                if part is None or thing is None or (part == GRIPPER and thing[1]):
                    continue
                if self._normal_force(model, data, index) >= _TOUCH_NEWTONS:
                    touching.add((part, thing[0]))
        return frozenset(touching)

    def replay_scalars(self, env_id: int, ee_position: np.ndarray | None) -> list[RerunScalar]:
        """The arm's and the gripper's joints, the distance to each object, and the forces.

        The arm's and the gripper's torques are `qfrc_actuator` at their joints' dof
        indices. `ee_force` and `ee_torque` are the first six values of `sensordata`,
        which robosuite fills from the wrist's force-torque sensor.
        """
        data = self._sim(env_id).data
        cache = self._caches[env_id]
        qpos = np.asarray(data.qpos, dtype=np.float64)
        qvel = np.asarray(data.qvel, dtype=np.float64)
        qacc = np.asarray(data.qacc, dtype=np.float64)
        actuator = np.asarray(data.qfrc_actuator, dtype=np.float64)
        angles = qpos[cache.arm_qpos]
        scalars = [
            *rerun_scalars("joints", RerunTab.ROBOT, angles),
            *rerun_scalars("joints_vel", RerunTab.ROBOT, qvel[cache.arm_dof]),
            *rerun_scalars("joints_acc", RerunTab.ROBOT, qacc[cache.arm_dof]),
            *rerun_scalars("joint_pos_cos", RerunTab.ROBOT, np.cos(angles)),
            *rerun_scalars("joint_pos_sin", RerunTab.ROBOT, np.sin(angles)),
            *rerun_scalars("gripper_qvel", RerunTab.ROBOT, qvel[cache.gripper_dof]),
        ]
        positions = data.xpos
        for name, body_id in cache.objects.items():
            distance = 0.0
            if ee_position is not None:
                offset = np.asarray(positions[body_id], dtype=np.float64) - ee_position[:3]
                distance = float(np.linalg.norm(offset))
            scalars.append(RerunScalar(f"object_dist/{name}", RerunTab.TASK, distance, None))
        external = np.asarray(data.cfrc_ext, dtype=np.float64)
        for name, body_id in cache.contacts.items():
            scalars.append(
                RerunScalar(
                    f"contacts/{name}",
                    RerunTab.PHYSICS,
                    float(np.linalg.norm(external[body_id])),
                    None,
                )
            )
        sensors = np.asarray(getattr(data, "sensordata", ()), dtype=np.float64).reshape(-1)
        scalars += [
            *rerun_scalars("ee_force", RerunTab.PHYSICS, sensors[:3]),
            *rerun_scalars("ee_torque", RerunTab.PHYSICS, sensors[3:6]),
            *rerun_scalars("joint_torques", RerunTab.PHYSICS, actuator[cache.arm_dof]),
            *rerun_scalars("gripper_torques", RerunTab.PHYSICS, actuator[cache.gripper_dof]),
            *rerun_scalars(
                "actuator_force", RerunTab.PHYSICS, np.asarray(data.actuator_force).reshape(-1)
            ),
            *rerun_scalars("ctrl", RerunTab.PHYSICS, np.asarray(data.ctrl).reshape(-1)),
            *rerun_scalars("qpos_robot", RerunTab.PHYSICS, angles),
            *rerun_scalars("qvel_robot", RerunTab.PHYSICS, qvel[cache.arm_dof]),
            *rerun_scalars("qpos_gripper", RerunTab.PHYSICS, qpos[cache.gripper_qpos]),
            *rerun_scalars("qvel_gripper", RerunTab.PHYSICS, qvel[cache.gripper_dof]),
            *rerun_scalars("qpos_objects", RerunTab.PHYSICS, qpos[cache.object_qpos]),
            *rerun_scalars("qvel_objects", RerunTab.PHYSICS, qvel[cache.object_dof]),
        ]
        return scalars

    def _sim(self, env_id: int) -> Any:
        """The current MuJoCo handle of env `env_id`, at the config's path from its env."""
        return attrgetter(self._config.path)(self._envs[env_id])

    def _joint_indices(self, model: Any) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """The indices of the joint values of the arm, the gripper and the objects.

        Both dicts have the keys `arm`, `gripper` and `objects`. The first dict
        contains indices into `qpos`, and the second contains indices into the dof
        arrays. MuJoCo stores the joints' values one after another, so a joint's
        values run from its address to the next joint's address. The values of the
        last joint run to the end.
        """
        count = int(model.njnt)
        qpos = [*(int(address) for address in model.jnt_qposadr[:count]), int(model.nq)]
        dof = [*(int(address) for address in model.jnt_dofadr[:count]), int(model.nv)]
        found: dict[str, tuple[list[int], list[int]]] = {
            "arm": ([], []),
            "gripper": ([], []),
            "objects": ([], []),
        }
        for joint_id in range(count):
            name = self._joint_name(model, joint_id)
            if name.startswith(self._config.arm_prefix):
                part = "arm"
            elif name.startswith(self._config.gripper_prefix):
                part = "gripper"
            else:
                part = "objects"
            found[part][0].extend(range(qpos[joint_id], qpos[joint_id + 1]))
            found[part][1].extend(range(dof[joint_id], dof[joint_id + 1]))
        return (
            {part: np.array(indices[0], dtype=np.int64) for part, indices in found.items()},
            {part: np.array(indices[1], dtype=np.int64) for part, indices in found.items()},
        )

    def _body_role(self, name: str, body_id: int, objects: frozenset[int]) -> ReplaySceneBodyRole:
        """The body's role in the task.

        A body under a robot prefix is the robot, a body with a joint outside the robot
        is an object, and any other body is a fixture.
        """
        if name.startswith(self._config.robot_prefixes):
            return ReplaySceneBodyRole.ROBOT
        if body_id in objects:
            return ReplaySceneBodyRole.OBJECT
        return ReplaySceneBodyRole.FIXTURE

    def _object_name(
        self, name: str, body_id: int, objects: frozenset[int], owners: tuple[str, ...]
    ) -> str | None:
        """The task's name for the body, or None for the world, the robot, or an unnamed object.

        robosuite gives an object's root body the object's name, with a suffix such as
        `_main`. A fixture's other bodies start with the fixture's name, such as
        `flat_stove_1_burner`, and are named for the fixture. Any other fixture, such as
        the table or a wall, is `TABLE`.
        """
        # Body 0 is MuJoCo's world, whose geoms span the whole room.
        if body_id == 0 or name.startswith(self._config.robot_prefixes):
            return None
        suffix = self._config.object_body_suffix
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
        if body_id in objects:
            return None
        return next((owner for owner in owners if name.startswith(f"{owner}_")), TABLE)

    def _owners(self, names: list[str]) -> tuple[str, ...]:
        """The names of the model's root bodies, the longest first."""
        suffix = self._config.object_body_suffix
        found = {name[: -len(suffix)] for name in names if name.endswith(suffix) and name != suffix}
        return tuple(sorted(found, key=len, reverse=True))

    def _task_labels(
        self,
        model: Any,
        names: list[str],
        labels: dict[int, tuple[ReplaySceneBodyRole, str | None]],
        task_names: Mapping[str, TaskTarget],
    ) -> tuple[dict[int, tuple[ReplaySceneBodyRole, str | None]], dict[int, str], dict[int, str]]:
        """`labels` with each body that `task_names` refers to named for the task.

        Also return the task name that names each such body, and the fixture's task name
        for each part of a named fixture. A named object's body takes the task name. A
        named fixture's parts are the bodies under it in the model's body tree, such as
        its doors, shelves and knobs, and they take the fixture's role. A part with a name
        of its own takes the task name with the rest of that name, so
        `stove_1_knob_front_left` becomes `stove_knob_front_left`. A part that the default
        labels name after the fixture, or leave unnamed with a name that does not start
        with the fixture's, takes the task name.
        """
        renamed = dict(labels)
        named_by: dict[int, str] = {}
        parents: dict[int, str] = {}
        parent_ids = np.asarray(model.body_parentid, dtype=np.int64)
        for task_name, target in task_names.items():
            if isinstance(target, TaskBody):
                target_id = self._task_body_id(names, labels, task_name, target)
                role = labels[target_id][0]
                members = [target_id]
                if role is ReplaySceneBodyRole.FIXTURE:
                    members += self._descendants(parent_ids, target_id)
                owner = self._root_name(target.body)
                for body_id in members:
                    if body_id in named_by:
                        raise ValueError(
                            f"task names {named_by[body_id]!r} and {task_name!r} both name "
                            f"body {names[body_id]!r}"
                        )
                    named_by[body_id] = task_name
                    if body_id == target_id:
                        renamed[body_id] = (role, task_name)
                        continue
                    part_name = self._task_part_name(
                        task_name, owner, names[body_id], labels[body_id][1]
                    )
                    renamed[body_id] = (role, part_name)
                    parents[body_id] = task_name
            elif isinstance(target, TaskSite):
                continue
            else:
                assert_never(target)
        return renamed, named_by, parents

    @staticmethod
    def _task_body_id(
        names: list[str],
        labels: dict[int, tuple[ReplaySceneBodyRole, str | None]],
        task_name: str,
        target: TaskBody,
    ) -> int:
        """The id of the body that `target` refers to, which must be outside the robot.

        Body 0 is MuJoCo's world, whose geoms span the whole room.
        """
        if target.body not in names:
            raise ValueError(
                f"task name {task_name!r} refers to body {target.body!r}, which the model lacks"
            )
        body_id = names.index(target.body)
        if body_id == 0 or labels[body_id][0] is ReplaySceneBodyRole.ROBOT:
            raise ValueError(
                f"task name {task_name!r} refers to body {target.body!r}, which is the world "
                "or the robot"
            )
        return body_id

    @staticmethod
    def _descendants(parent_ids: np.ndarray, root_id: int) -> list[int]:
        """The ids of the bodies under body `root_id` in the model's body tree."""
        found = []
        for body_id in range(1, len(parent_ids)):
            ancestor = int(parent_ids[body_id])
            while ancestor not in (0, root_id):
                ancestor = int(parent_ids[ancestor])
            if body_id != root_id and ancestor == root_id:
                found.append(body_id)
        return found

    def _root_name(self, body_name: str) -> str:
        """`body_name` without the suffix that names an object's or a fixture's root body."""
        suffix = self._config.object_body_suffix
        return body_name[: -len(suffix)] if body_name.endswith(suffix) else body_name

    def _task_part_name(
        self, task_name: str, owner: str, body_name: str, body_object: str | None
    ) -> str:
        """The task's name for a part of the fixture whose root body is named `owner`."""
        prefix = f"{owner}_"
        if body_object is not None and body_object.startswith(prefix):
            return task_name + body_object[len(owner) :]
        if body_object is None and body_name.startswith(prefix):
            return task_name + self._root_name(body_name)[len(owner) :]
        return task_name

    def _task_sites(
        self,
        model: Any,
        names: list[str],
        labels: dict[int, tuple[ReplaySceneBodyRole, str | None]],
        named_by: dict[int, str],
        task_names: Mapping[str, TaskTarget],
    ) -> list[tuple[ReplaySceneBody, int]]:
        """A scene body for each site that `task_names` refers to, with its body's id.

        The body is named after the site, takes the task name and the role of the body
        that the site moves with, and holds one part: the site's shape, posed in that
        body's frame. When a task name names that body too, the site is its part.
        """
        sites = []
        named_sites: dict[str, str] = {}
        for task_name, target in task_names.items():
            if isinstance(target, TaskSite):
                if target.site in named_sites:
                    raise ValueError(
                        f"task names {named_sites[target.site]!r} and {task_name!r} both name "
                        f"site {target.site!r}"
                    )
                named_sites[target.site] = task_name
                sites.append(self._task_site(model, names, labels, named_by, task_name, target))
            elif isinstance(target, TaskBody):
                continue
            else:
                assert_never(target)
        return sites

    def _task_site(
        self,
        model: Any,
        names: list[str],
        labels: dict[int, tuple[ReplaySceneBodyRole, str | None]],
        named_by: dict[int, str],
        task_name: str,
        target: TaskSite,
    ) -> tuple[ReplaySceneBody, int]:
        if target.site in names:
            raise ValueError(f"site {target.site!r} has the name of a body")
        try:
            site_id = int(model.site(target.site).id)
        except KeyError as error:
            raise ValueError(
                f"task name {task_name!r} refers to site {target.site!r}, which the model lacks"
            ) from error
        mesh = self._primitive_mesh(
            int(model.site_type[site_id]), np.asarray(model.site_size[site_id], dtype=np.float32)
        )
        if mesh is None:
            raise ValueError(f"site {target.site!r} has a shape that the scene does not draw")
        body_id = int(model.site_bodyid[site_id])
        role = labels[body_id][0]
        if body_id == 0 or role is ReplaySceneBodyRole.ROBOT:
            raise ValueError(f"site {target.site!r} moves with the world or the robot")
        red, green, blue, alpha = (float(value) for value in model.site_rgba[site_id])
        part = ReplayScenePart(
            mesh=mesh,
            position=np.asarray(model.site_pos[site_id], dtype=np.float32),
            orientation=wxyz_to_xyzw(model.site_quat[site_id]).astype(np.float32),
            color=(red, green, blue, alpha),
        )
        body = ReplaySceneBody(
            name=target.site,
            parts=(part,),
            role=role,
            object=task_name,
            parent=named_by.get(body_id),
        )
        return body, body_id

    @staticmethod
    def _check_task_names_distinct(
        labels: dict[int, tuple[ReplaySceneBodyRole, str | None]],
        named_by: dict[int, str],
        sites: list[tuple[ReplaySceneBody, int]],
    ) -> None:
        """Raise if a task name gives a body or a site a name that another body has.

        The grader matches a fact to the scene by name, so a name that two unrelated
        bodies share would link a fact to the wrong body.
        """
        sources: dict[str, set[str | None]] = {}
        for body_id, (role, name) in labels.items():
            if name is not None and role is not ReplaySceneBodyRole.ROBOT:
                sources.setdefault(name, set()).add(named_by.get(body_id))
        for body, _ in sites:
            if body.object is not None:
                sources.setdefault(body.object, set()).add(body.object)
        clashes = sorted(name for name, by in sources.items() if len(by) > 1 and by - {None})
        if clashes:
            raise ValueError(f"task names give names that other bodies have: {clashes}")

    def _robot_parts(self, model: Any) -> dict[int, str]:
        """Each body of the gripper as `GRIPPER`, and each body of the arm as `ARM`."""
        parts = {}
        for body_id in range(int(model.nbody)):
            name = self._body_name(model, body_id)
            if name.startswith(self._config.gripper_prefix):
                parts[body_id] = GRIPPER
            elif name.startswith(self._config.arm_prefix):
                parts[body_id] = ARM
        return parts

    def _object_bodies(self, model: Any) -> dict[str, int]:
        """Each object's body name and id, for the bodies with a joint outside the robot."""
        counts = np.asarray(model.body_jntnum, dtype=np.int64).reshape(-1)
        return {
            name: body_id
            for body_id in range(int(model.nbody))
            if int(counts[body_id]) > 0
            and not (name := self._body_name(model, body_id)).startswith(
                self._config.robot_prefixes
            )
        }

    @staticmethod
    def _log_skipped_geoms(skipped: dict[int, int]) -> None:
        """Log the drawn geoms that the scene leaves out, so a missing object shows in the log."""
        # The SDK does not depend on `mujoco`, so this import is local.
        import mujoco  # ty: ignore[unresolved-import]

        if len(skipped) == 0:
            return
        described = ", ".join(
            f"{count} {mujoco.mjtGeom(kind).name}" for kind, count in sorted(skipped.items())
        )
        _logger.info("replay scene omits untessellated rendered geoms: %s", described)

    @staticmethod
    def _geom_mesh(model: Any, geom_id: int) -> ReplayMesh | None:
        """The geom's triangles, or None for a geom type that the scene leaves out.

        The scene draws boxes, cylinders and meshes. It leaves out the other types,
        such as planes, height fields and spheres.
        """
        # The SDK does not depend on `mujoco`, so this import is local.
        import mujoco  # ty: ignore[unresolved-import]

        kind = int(model.geom_type[geom_id])
        if kind != mujoco.mjtGeom.mjGEOM_MESH:
            return MujocoSimulator._primitive_mesh(
                kind, np.asarray(model.geom_size[geom_id], dtype=np.float32)
            )
        mesh_id = int(model.geom_dataid[geom_id])
        vertex_start = int(model.mesh_vertadr[mesh_id])
        face_start = int(model.mesh_faceadr[mesh_id])
        vertices = np.asarray(
            model.mesh_vert[vertex_start : vertex_start + int(model.mesh_vertnum[mesh_id])]
        ).reshape(-1, 3)
        faces = np.asarray(
            model.mesh_face[face_start : face_start + int(model.mesh_facenum[mesh_id])]
        ).reshape(-1, 3)
        return ReplayMesh(vertices=vertices, faces=faces.astype(np.uint32))

    @staticmethod
    def _primitive_mesh(kind: int, size: np.ndarray) -> ReplayMesh | None:
        """The triangles of a box or a cylinder of MuJoCo type `kind`, or None for another type.

        A geom and a site of the same type store their size in the same way.
        """
        # The SDK does not depend on `mujoco`, so this import is local.
        import mujoco  # ty: ignore[unresolved-import]

        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            return ReplayMesh.box([float(value) for value in size[:3]])
        if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return ReplayMesh.cylinder(radius=float(size[0]), half_length=float(size[1]))
        return None

    @staticmethod
    def _texture_mean(model: Any, texture_id: int) -> tuple[float, float, float] | None:
        """The texture's mean RGB colour, or None if its pixels are not available.

        MuJoCo 3.2.3 and later store each texture's pixels in `tex_data`, from
        `tex_adr`. Older versions have no `tex_data`, so `_geom_colour` uses the
        material's colour instead.
        """
        data = getattr(model, "tex_data", None)
        if data is None:
            return None
        channels = int(model.tex_nchannel[texture_id])
        count = int(model.tex_width[texture_id]) * int(model.tex_height[texture_id]) * channels
        start = int(model.tex_adr[texture_id])
        pixels = np.asarray(data).reshape(-1)[start : start + count]
        if count == 0 or pixels.size < count:
            return None
        mean = pixels.reshape(-1, channels)[:, :3].astype(np.float32).mean(axis=0) / 255.0
        return float(mean[0]), float(mean[1]), float(mean[2])

    @staticmethod
    def _geom_colour(
        model: Any, geom_id: int, means: dict[int, tuple[float, float, float] | None]
    ) -> tuple[float, float, float, float]:
        """The geom's colour: its texture's mean colour, else its material's, else its own.

        The geom's own alpha applies in each case. `means` caches the mean colour of
        each texture after the first read.
        """
        rgba = np.asarray(model.geom_rgba[geom_id], dtype=np.float32).reshape(-1)
        alpha = float(rgba[3])
        material_id = int(model.geom_matid[geom_id])
        if material_id < 0:
            return float(rgba[0]), float(rgba[1]), float(rgba[2]), alpha
        texture_id = int(np.asarray(model.mat_texid[material_id]).reshape(-1).max())
        if texture_id >= 0:
            if texture_id not in means:
                means[texture_id] = MujocoSimulator._texture_mean(model, texture_id)
            mean = means[texture_id]
            if mean is not None:
                return mean[0], mean[1], mean[2], alpha
        material = np.asarray(model.mat_rgba[material_id], dtype=np.float32).reshape(-1)
        return float(material[0]), float(material[1]), float(material[2]), alpha

    @staticmethod
    def _normal_force(model: Any, data: Any, index: int) -> float:
        """The normal force of contact `index`, in newtons."""
        # The SDK does not depend on `mujoco`, so this import is local.
        import mujoco  # ty: ignore[unresolved-import]

        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, wrench)
        return abs(float(wrench[0]))

    @staticmethod
    def _body_name(model: Any, body_id: int) -> str:
        """The model's name for the body, or `body<id>` for an unnamed body.

        The replay stores poses by body name, so a second body with the same name
        would replace the first. An unnamed body therefore gets its id in its name.

        This method uses `model.body(id)`, because MuJoCo's `MjModel` has it from 2.3.7
        and robosuite's model wrapper passes it through. robosuite's `body_id2name` is
        absent from MuJoCo and dm_control.
        """
        return model.body(body_id).name or f"body{body_id}"

    @staticmethod
    def _joint_name(model: Any, joint_id: int) -> str:
        """The model's name for a joint, or an empty string for an unnamed joint."""
        return model.joint(joint_id).name


__all__ = [
    "BenchmarkMujocoSimulatorConfig",
    "MujocoSimulator",
]
