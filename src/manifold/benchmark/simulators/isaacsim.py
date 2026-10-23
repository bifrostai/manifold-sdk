"""The simulator for Isaac Sim: the scene, poses and scalars of an Isaac Lab env.

`start` launches Isaac Sim through Isaac Lab's `AppLauncher`, which has to happen
before anything imports `isaaclab` or `pxr`. So this module imports neither at
its top, and each function that needs one imports it.

Isaac Lab's `AppLauncher` installs a Python handler for SIGSEGV. Python runs that
handler only between bytecodes. A segfault inside a C call never reaches a bytecode,
so the process faults again at the same instruction, keeps a CPU core busy, and does
not end. So `start` gives SIGSEGV to `faulthandler`. On a segfault, `faulthandler`
prints the Python stack of each thread and ends the process with exit code 139, and
the runner can then retry the task. Kit loads its extensions while an env is built,
and an extension can install its own SIGSEGV handler. So `attach` gives SIGSEGV to
`faulthandler` again.

`IsaacSimSimulator.apply_settings` sets the config's physics and scene settings on
an Isaac Lab env's config. The hooks call it while they build the env, because the
simulator gets the env only after it is built.

The scene is read from the USD stage under the env's own prim, such as
`/World/envs/env_3`. Each drawn prim is grouped under its nearest ancestor with
`UsdPhysics.RigidBodyAPI`, which is the prim that PhysX moves. A prim without such
an ancestor is fixed, such as a table, and is posed once from the stage. Isaac Lab
tracks the links of each articulation and the root of each rigid object. Each
step reads their poses from Isaac Lab's tensors.

Isaac Lab lays a batched env out as a grid and reports world positions. This
module subtracts the env's origin in `scene.env_origins`, so the positions that
it returns are in the env's frame.

Isaac Lab's `isaaclab` package orders quaternions `wxyz` before version 1.0.0 and
`xyzw` from 1.0.0 on. `attach` reads the version, and the quaternions that this
module returns are `xyzw`.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import re
import resource
import signal
import sys
import traceback
from dataclasses import dataclass
from operator import attrgetter
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from manifold.benchmark.simulators import (
    BOX_VERTICES,
    GRIPPER,
    TABLE,
    Simulator,
    bounding_box_part,
)
from manifold.core.embodiment import JointActionSpace
from manifold.lib.rotation import rotation_matrix_from_xyzw, wxyz_to_xyzw, xyzw_from_rotation_matrix
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
    from collections.abc import Sequence

    from manifold.benchmark.config import BenchmarkConfig

_logger = logging.getLogger(__name__)

# The soft limit on open files that `start` asks for. A host with more than two
# GPUs needs more file descriptors than a container's default for resources such
# as Vulkan fences, and Kit raises the limit only when its own request is higher.
_FILE_LIMIT = 16384

# The near plane that `IsaacSimSimulator.apply_settings` gives a camera without a clipping
# range. A higher near plane would cut the gripper out of a wrist camera.
_NEAR_CLIP_METRES = 0.01

# The colour of a part with no material colour and no display colour.
_GREY = (0.7, 0.7, 0.7, 1.0)

# The shader inputs for a material's constant colour. A UsdPreviewSurface
# names it `diffuseColor`, and Omniverse's MDL shaders use the other names.
_COLOUR_INPUTS = ("diffuseColor", "diffuse_color_constant", "base_color_constant", "diffuse_tint")

# The factor that a camera's frame is box-averaged by before the replay stores it.
_IMAGE_DIVISOR = 2

# A contact force of at least this many newtons counts as the gripper touching an object.
_TOUCH_NEWTONS = 0.5

# How near the end effector an object's origin counts as near the gripper, when
# no contact sensor reports the gripper's forces.
_GRIPPER_NEAR_METRES = 0.08

# A robot part presses on a body this hard or harder, in newtons, when it touches it.
_PART_TOUCH_NEWTONS = 0.1

# The start of the scene name of each contact sensor that `apply_settings` adds.
_TOUCH_SENSOR_PREFIX = "manifold_touch__"

# How near two objects' origins count as near each other.
_OBJECT_NEAR_METRES = 0.10

# What Isaac Lab tracks for a body: "link" or "root", the asset's name in the
# scene, and the link's index in the asset's `body_names`, or 0 for a root.
_TrackedBody = tuple[str, str, int]


@dataclass(frozen=True)
class _IsaacSimEpisodeCache:
    """What `replay_bodies` found in the env at the start of an episode, for each later step.

    `replay_bodies` walks the USD stage to find these values, and a walk is too slow
    to repeat at each step.

    `bodies` maps each scene body to what Isaac Lab tracks for it, or to the pose
    in the env's frame of a fixed body. `sensors` contains each contact sensor that
    the scalars plot, with the names of the objects that it filters on.
    """

    bodies: dict[str, _TrackedBody | ReplayPose]
    joints: int
    arm: int
    links: tuple[str, ...]
    objects: tuple[str, ...]
    terms: tuple[str, ...]
    sensors: tuple[tuple[str, tuple[str, ...]], ...]
    cameras: tuple[str, ...]


class BenchmarkIsaacSimSimulatorConfig(BaseModel):
    """Where an Isaac Lab env is, and how Isaac Sim is launched.

    `path` is the attribute path from the env that `on_load` returns to Isaac Lab's
    env, such as `env.unwrapped` for a gym env. `app` contains the keyword arguments of Isaac Lab's
    `AppLauncher`, such as `{"headless": true, "enable_cameras": true}`. `robot` is
    the scene's name for the robot's articulation.

    `IsaacSimSimulator.apply_settings` sets `max_position_iterations`, `env_spacing`
    and `camera_far_clip` on an env's config. A null value keeps the env's own value.
    `max_position_iterations` caps PhysX's position solver iterations in a step, from
    1 to 255. `env_spacing` is the distance in metres
    between neighbouring envs. `camera_far_clip` is the distance in metres past which
    the cameras draw nothing.

    `touch_links` maps each robot link whose contacts the replay records to `arm` or
    `gripper`, by the link's prim path below the robot's prim, such as `panda_link1`.
    `apply_settings` adds a contact sensor to each link, filtered to the scene's other
    assets. Null adds none, and the replay then records no contacts.
    """

    # A config lists all of its values, so an unknown key raises.
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["isaacsim"]
    path: str
    app: dict[str, Any]
    robot: str
    max_position_iterations: int | None = Field(ge=1, le=255)
    env_spacing: float | None = Field(gt=0)
    camera_far_clip: float | None = Field(gt=0)
    touch_links: dict[str, Literal["arm", "gripper"]] | None


def _take_sigsegv() -> None:
    """Give SIGSEGV to `faulthandler`, which ends the process on a segfault.

    Each call sets the handler again, so it replaces a handler that an extension
    installed since the last call.
    """
    faulthandler.enable(all_threads=True)


def _camera_cfgs(scene_cfg: Any) -> list[Any]:
    """Return the camera configs on `scene_cfg`.

    A task can name its cameras anything, so this function checks the type of each
    attribute instead of looking up names.
    """
    if scene_cfg is None:
        return []
    from isaaclab.sensors import CameraCfg, TiledCameraCfg  # ty: ignore[unresolved-import]

    cameras = []
    for name in dir(scene_cfg):
        if name.startswith("_"):
            continue
        try:
            value = getattr(scene_cfg, name)
        except Exception:  # A property that needs a built scene raises.
            continue
        if isinstance(value, (CameraCfg, TiledCameraCfg)):
            cameras.append(value)
    return cameras


class IsaacSimSimulator(Simulator):
    """Read the scene, poses and scalars of Isaac Lab envs, and start and stop Isaac Sim."""

    def __init__(
        self, config: BenchmarkIsaacSimSimulatorConfig, benchmark: BenchmarkConfig
    ) -> None:
        self._config = config
        self._arm = self._arm_joints(benchmark)
        self._spec_cameras = tuple(str(camera.name) for camera in benchmark.spec.sensors)
        self._app: Any = None
        self._envs: dict[int, tuple[Any, int]] = {}
        self._caches: dict[int, _IsaacSimEpisodeCache] = {}
        self._scalar_last = False
        # The touch links already warned about, so each warns once.
        self._unnamed_touches: set[str] = set()

    @staticmethod
    def apply_settings(env_cfg: Any, config: BenchmarkIsaacSimSimulatorConfig) -> None:
        """Set the config's solver cap, env spacing, camera far clip and touch sensors on `env_cfg`.

        `env_cfg` is an Isaac Lab env's config before the env is built. A setting that is
        null keeps the env's value. A setting without its target in `env_cfg` raises,
        because a setting that did not apply looks the same as one that did.

        Raises:
            ValueError: If `env_cfg` has no target for a setting that the config gives.
        """
        if config.max_position_iterations is not None:
            physx = getattr(getattr(env_cfg, "sim", None), "physx", None)
            if physx is None or not hasattr(physx, "max_position_iteration_count"):
                raise ValueError("the env's config has no `sim.physx.max_position_iteration_count`")
            physx.max_position_iteration_count = config.max_position_iterations
        scene_cfg = getattr(env_cfg, "scene", None)
        if config.env_spacing is not None:
            if scene_cfg is None or not hasattr(scene_cfg, "env_spacing"):
                raise ValueError("the env's config has no `scene.env_spacing`")
            scene_cfg.env_spacing = config.env_spacing
        if config.camera_far_clip is not None:
            cameras = _camera_cfgs(scene_cfg)
            if not cameras:
                raise ValueError("the env's scene config has no camera to set the far clip of")
            for camera in cameras:
                spawn = getattr(camera, "spawn", None)
                existing = getattr(spawn, "clipping_range", None)
                if spawn is None or not hasattr(spawn, "clipping_range"):
                    raise ValueError(f"camera {camera.prim_path!r} has no `spawn.clipping_range`")
                near = (
                    float(existing[0])
                    if isinstance(existing, (tuple, list)) and len(existing) == 2
                    else _NEAR_CLIP_METRES
                )
                spawn.clipping_range = (near, config.camera_far_clip)
        IsaacSimSimulator._add_touch_sensors(env_cfg, scene_cfg, config)

    @property
    def supported_versions(self) -> dict[str, str]:
        """The version range of Isaac Lab's `isaaclab` package.

        The range covers `isaaclab` 0.44 on Isaac Sim 5.0 up to `isaaclab` 6.1 on
        Isaac Sim 6.0. Isaac Sim installs no Python package with its version.
        """
        return {"isaaclab": ">=0.44,<6.2"}

    def start(self) -> None:
        """Raise the limit on open files, launch Isaac Sim, and take SIGSEGV back.

        The soft limit on open files is raised to 16384, or to the hard limit when that
        is lower. A higher soft limit is kept. Isaac Sim launches with the config's
        `app`. After the launch, the core file size limit is set to 0, SIGSEGV goes to
        `faulthandler`, and SIGUSR1 prints the stack of each thread.
        """
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        wanted = _FILE_LIMIT if hard == resource.RLIM_INFINITY else min(_FILE_LIMIT, hard)
        if soft != resource.RLIM_INFINITY and wanted > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))
        # The SDK does not depend on Isaac Sim. An Isaac Sim benchmark image installs it.
        from isaaclab.app import AppLauncher  # ty: ignore[unresolved-import]

        self._app = AppLauncher(**self._config.app).app
        # A segfault ends the process without a core file. A core file of an Isaac Sim
        # process is as large as its memory, tens of gigabytes.
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        _take_sigsegv()
        # SIGUSR1 prints every thread's stack, so a stalled run can be read with
        # `docker kill -s USR1 <container>`.
        faulthandler.register(signal.SIGUSR1, all_threads=True)

    def stop(self) -> None:
        """End the process with the runner's exit code.

        Kit's `close` ends the process with exit code 0, even while an exception
        propagates out of the runner, and Python does not print that exception. So
        `stop` does not call `close`. It prints the exception, flushes the output,
        and calls `os._exit` with exit code 1 after an exception, or 0 otherwise.
        """
        failure = sys.exc_info()[1]
        if failure is not None:
            traceback.print_exception(failure)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if failure is not None else 0)

    def attach(self, env: Any, env_ids: Sequence[int]) -> None:
        """Attach env ids `env_ids` to the Isaac Lab env at the config's path in `env`.

        `attach` uses the env's `unwrapped` when it has one. Env id `env_ids[i]` is
        row `i` of Isaac Lab's tensors. `attach` also gives SIGSEGV to `faulthandler`
        again, because the env build can load an extension that installs its own
        handler.
        """
        _take_sigsegv()
        found = attrgetter(self._config.path)(env)
        isaac = getattr(found, "unwrapped", found)
        self._scalar_last = self._isaaclab_orders_xyzw()
        for row, env_id in enumerate(env_ids):
            self._envs[env_id] = (isaac, row)
            self._caches.pop(env_id, None)

    def replay_bodies(self, env_id: int) -> tuple[ReplaySceneBody, ...]:
        """Each body of the env's USD prim with its drawn parts, read again for each episode.

        Each part's vertices are in its body's frame. A body that Isaac Lab tracks
        is framed at the pose that Isaac Lab reports, and a fixed body at the
        pose of its prim on the stage. `replay_bodies` also fills the episode cache
        that the other replay methods read.
        """
        # The SDK does not depend on Isaac Sim. An Isaac Sim benchmark image installs it.
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade  # ty: ignore[unresolved-import]

        isaac, row = self._envs[env_id]
        scene = isaac.scene
        stage = scene.stage
        origin = self._origin(scene, row)
        metres = float(UsdGeom.GetStageMetersPerUnit(stage))
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        parts: dict[str, list[ReplayScenePart]] = {}
        tracked_bodies: dict[str, _TrackedBody | ReplayPose] = {}
        labels: dict[str, tuple[ReplaySceneBodyRole, str | None]] = {}
        owners: dict[str, str] = {}
        skipped: dict[str, int] = {}
        boxed: list[str] = []
        root = stage.GetPrimAtPath(scene.env_prim_paths[row])
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            reason = self._skip_reason(prim, UsdGeom)
            shape = self._prim_geometry(prim, UsdGeom) if reason is None else None
            if shape is None:
                reason = reason or f"{prim.GetTypeName()} prims"
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            owner = self._owner(prim, UsdPhysics)
            path = str(owner.GetPath())
            name = self._unique_name(str(owner.GetName()) or path, path, owners)
            tracked = self._tracked_body(path, scene)
            labels.setdefault(name, self._body_labels(tracked))
            if tracked is None:
                pose = self._pose(self._matrix(cache.GetLocalToWorldTransform(owner)), metres)
                local_position = (pose.position - origin).astype(np.float32)
                tracked_bodies.setdefault(name, ReplayPose(local_position, pose.orientation))
            else:
                pose = self._world_pose(scene, tracked, row)
                tracked_bodies.setdefault(name, tracked)
            # The vertices are moved into the body's frame by subtracting its position
            # and undoing its rotation. Inverting the prim's matrix instead would also
            # undo the scale that a USD asset is authored with.
            homogeneous = np.c_[shape[0], np.ones(len(shape[0]))]
            world = (homogeneous @ self._matrix(cache.GetLocalToWorldTransform(prim)).T)[
                :, :3
            ] * metres
            rotation = rotation_matrix_from_xyzw(pose.orientation)
            local = (world - np.asarray(pose.position, dtype=np.float64)) @ rotation
            found = self._parts(prim, shape, local, (UsdGeom, UsdShade))
            if len(local) > BOX_VERTICES and (tracked is None or tracked[0] != "root"):
                boxed.append(name)
                found = [bounding_box_part(found)]
            parts.setdefault(name, []).extend(found)
        if boxed or skipped:
            _logger.info(
                "replay scene draws %s as boxes and omits %s",
                ", ".join(boxed) or "nothing",
                ", ".join(f"{count} {reason}" for reason, count in sorted(skipped.items()))
                or "nothing",
            )
        self._caches[env_id] = self._episode_cache(isaac, tracked_bodies)
        return tuple(
            ReplaySceneBody(
                name=name, parts=tuple(found), role=labels[name][0], object=labels[name][1]
            )
            for name, found in parts.items()
        )

    def replay_poses(self, env_id: int) -> dict[str, ReplayPose]:
        """Each scene body's pose in the env's frame now, in `xyzw`.

        A tracked body's pose is read from the asset's `body_link_pos_w` and
        `body_link_quat_w` for a link, or `root_link_pos_w` and `root_link_quat_w`
        for a rigid object. A fixed body keeps the pose that it had on the stage.
        """
        isaac, row = self._envs[env_id]
        origin = self._origin(isaac.scene, row)
        poses = {}
        for name, held in self._caches[env_id].bodies.items():
            if isinstance(held, ReplayPose):
                poses[name] = held
                continue
            pose = self._world_pose(isaac.scene, held, row)
            poses[name] = ReplayPose(
                position=(pose.position - origin).astype(np.float32), orientation=pose.orientation
            )
        return poses

    def replay_extrinsics(self, env_id: int) -> dict[str, ReplayPose]:
        """An empty dict, because the config does not state which sensor produces each camera."""
        return {}

    def replay_images(self, env_id: int) -> dict[str, np.ndarray]:
        """The frames of the camera sensors that produce none of the spec's cameras.

        Each frame is the sensor's `rgb` output, box-averaged by 2 along each side.
        The dict maps each sensor's name in the scene to its frame.
        """
        isaac, row = self._envs[env_id]
        frames = {}
        for name in self._caches[env_id].cameras:
            output = isaac.scene.sensors[name].data.output
            frames[name] = self._halved(self._numpy(output["rgb"])[row])
        return frames

    def replay_contacts(self, env_id: int) -> frozenset[tuple[str, str]]:
        """Each robot part that pressed on a scene asset with at least 0.1 N in the last action.

        The touch sensors that `apply_settings` adds keep each physics step of an action,
        so a touch between two actions is seen. An asset is named as the scene names it.
        """
        isaac, row = self._envs[env_id]
        sensors = isaac.scene.sensors
        touching = set()
        for link, part in (self._config.touch_links or {}).items():
            sensor = sensors.get(self._touch_sensor_name(link))
            if sensor is None:
                continue
            history = self._numpy(sensor.data.force_matrix_w_history)
            if history.size == 0:
                continue
            # (envs, steps, bodies, filters, 3), so the largest force for each filter.
            norms = np.linalg.norm(np.asarray(history[row], dtype=np.float64), axis=-1)
            peaks = norms.reshape(-1, norms.shape[-1]).max(axis=0)
            names = self._touch_filter_names(isaac.scene, sensor.cfg.filter_prim_paths_expr)
            if len(names) != len(peaks):
                # A filter that matches several bodies, or none, gives the forces no names.
                self._warn_unnamed_touches(link, len(names), len(peaks))
                continue
            touching.update(
                (part, name)
                for name, force in zip(names, peaks, strict=True)
                if force >= _PART_TOUCH_NEWTONS
            )
        return frozenset(touching)

    def replay_scalars(self, env_id: int, ee_position: np.ndarray | None) -> list[RerunScalar]:
        """The robot's joints, the task's terms and distances, and the physics forces.

        The arm's joints are the first joints of the config's robot, as many as the
        spec gives each arm, and the gripper's are the rest.
        """
        isaac, row = self._envs[env_id]
        cache = self._caches[env_id]
        scene = isaac.scene
        origin = self._origin(scene, row)
        positions = {
            name: self._row(scene[name].data, "root_link_pos_w", row, 3) - origin
            for name in cache.objects
        }
        return [
            *self._robot_scalars(scene, cache, row, origin),
            *self._task_scalars(isaac, cache, row, positions, ee_position),
            *self._physics_scalars(scene, cache, row, positions, ee_position),
        ]

    def _robot_scalars(
        self, scene: Any, cache: _IsaacSimEpisodeCache, row: int, origin: np.ndarray
    ) -> list[RerunScalar]:
        """The arm's joints, the gripper's joint velocities, and the robot's base pose."""
        robot = scene[self._config.robot].data
        arm = cache.arm
        angles = self._row(robot, "joint_pos", row, cache.joints)
        velocities = self._row(robot, "joint_vel", row, cache.joints)
        accelerations = self._row(robot, "joint_acc", row, cache.joints)
        return [
            *rerun_scalars("joints", RerunTab.ROBOT, angles[:arm]),
            *rerun_scalars("joints_vel", RerunTab.ROBOT, velocities[:arm]),
            *rerun_scalars("joints_acc", RerunTab.ROBOT, accelerations[:arm]),
            *rerun_scalars("joint_pos_cos", RerunTab.ROBOT, np.cos(angles[:arm])),
            *rerun_scalars("joint_pos_sin", RerunTab.ROBOT, np.sin(angles[:arm])),
            *rerun_scalars("gripper_qvel", RerunTab.ROBOT, velocities[arm:]),
            *rerun_scalars(
                "base_pos", RerunTab.ROBOT, self._row(robot, "root_link_pos_w", row, 3) - origin
            ),
            *rerun_scalars(
                "base_quat",
                RerunTab.ROBOT,
                self._xyzw(self._row(robot, "root_link_quat_w", row, 4)),
            ),
        ]

    def _task_scalars(
        self,
        isaac: Any,
        cache: _IsaacSimEpisodeCache,
        row: int,
        positions: dict[str, np.ndarray],
        ee_position: np.ndarray | None,
    ) -> list[RerunScalar]:
        """Whether each termination term fires, and each object's distance from the end effector.

        A distance is 0 when `ee_position` is None.
        """
        scalars = []
        for term in cache.terms:
            fired = self._numpy(isaac.termination_manager.get_term(term))
            value = float(fired.reshape(-1)[row])
            scalars.append(RerunScalar(f"terms/{term}", RerunTab.TASK, value, None))
        for name, position in positions.items():
            distance = 0.0
            if ee_position is not None:
                distance = float(np.linalg.norm(position - ee_position[:3]))
            scalars.append(RerunScalar(f"object_dist/{name}", RerunTab.TASK, distance, None))
        return scalars

    def _physics_scalars(
        self,
        scene: Any,
        cache: _IsaacSimEpisodeCache,
        row: int,
        positions: dict[str, np.ndarray],
        ee_position: np.ndarray | None,
    ) -> list[RerunScalar]:
        """The robot's torques and joint state, the contacts, and the objects' motion.

        `gripper_near/<object>` is whether the first contact sensor's force against
        the object reaches 0.5 N, or, when no contact sensor filters on objects,
        whether the object's origin is within 8 cm of `ee_position`.
        `obj_near/<a>__<b>` is whether two objects' origins are within 10 cm.
        """
        robot = scene[self._config.robot].data
        arm = cache.arm
        applied = self._row(robot, "applied_torque", row, cache.joints)
        scalars = [
            *rerun_scalars("joint_torques", RerunTab.PHYSICS, applied[:arm]),
            *rerun_scalars("gripper_torques", RerunTab.PHYSICS, applied[arm:]),
            *rerun_scalars(
                "computed_torques",
                RerunTab.PHYSICS,
                self._row(robot, "computed_torque", row, cache.joints)[:arm],
            ),
            *rerun_scalars(
                "qpos_robot", RerunTab.PHYSICS, self._row(robot, "joint_pos", row, cache.joints)
            ),
            *rerun_scalars(
                "qvel_robot", RerunTab.PHYSICS, self._row(robot, "joint_vel", row, cache.joints)
            ),
        ]
        contacts, forces = self._contact_scalars(scene, cache, row)
        scalars += contacts
        for name in cache.objects:
            scalars += rerun_scalars(f"obj_pos/{name}", RerunTab.PHYSICS, positions[name])
            scalars += rerun_scalars(
                f"obj_vel/{name}",
                RerunTab.PHYSICS,
                self._row(scene[name].data, "root_com_lin_vel_w", row, 3),
            )
        for name, position in positions.items():
            if forces is not None:
                near = forces.get(name, 0.0) >= _TOUCH_NEWTONS
            elif ee_position is not None:
                near = float(np.linalg.norm(position - ee_position[:3])) < _GRIPPER_NEAR_METRES
            else:
                near = False
            scalars.append(RerunScalar(f"gripper_near/{name}", RerunTab.PHYSICS, float(near), None))
        for first_index, first in enumerate(cache.objects):
            for second in cache.objects[first_index + 1 :]:
                distance = float(np.linalg.norm(positions[first] - positions[second]))
                near = float(distance < _OBJECT_NEAR_METRES)
                scalars.append(
                    RerunScalar(f"obj_near/{first}__{second}", RerunTab.PHYSICS, near, None)
                )
        return scalars

    def _contact_scalars(
        self, scene: Any, cache: _IsaacSimEpisodeCache, row: int
    ) -> tuple[list[RerunScalar], dict[str, float] | None]:
        """The contact forces, and the first contact sensor's force against each object.

        `contacts/<sensor>/<object>` is the largest force between the sensor's bodies
        and that object, and `contacts/<sensor>` is the sensor's net force when it
        filters on no object. `contacts/<link>` is the norm of the wrench through the
        link's joint. Isaac Lab 3 does not report this wrench, so the value is 0 there.
        The forces are None when no contact sensor filters on objects.
        """
        scalars = []
        forces: dict[str, float] | None = None
        for name, objects in cache.sensors:
            data = scene.sensors[name].data
            if len(objects) == 0:
                net = float(np.linalg.norm(self._row(data, "net_forces_w", row, None)))
                scalars.append(RerunScalar(f"contacts/{name}", RerunTab.PHYSICS, net, None))
                continue
            matrix = getattr(data, "force_matrix_w", None)
            values = np.zeros(len(objects))
            if matrix is not None and self._numpy(matrix).ndim == 4:
                lane = self._numpy(matrix)[row]
                values = self._resized(np.linalg.norm(lane, axis=-1).max(axis=0), len(objects))
            scalars += [
                RerunScalar(f"contacts/{name}/{obj}", RerunTab.PHYSICS, float(value), None)
                for obj, value in zip(objects, values, strict=True)
            ]
            if forces is None:
                forces = {obj: float(value) for obj, value in zip(objects, values, strict=True)}
        robot = scene[self._config.robot].data
        found = getattr(robot, "body_incoming_joint_wrench_b", None)
        wrench = self._numpy(found) if found is not None else np.zeros((0, 0, 0))
        for link_index, link in enumerate(cache.links):
            magnitude = 0.0
            if wrench.ndim == 3 and row < len(wrench) and link_index < wrench.shape[1]:
                magnitude = float(np.linalg.norm(wrench[row, link_index]))
            scalars.append(RerunScalar(f"contacts/{link}", RerunTab.PHYSICS, magnitude, None))
        return scalars, forces

    def _xyzw(self, quaternion: np.ndarray) -> np.ndarray:
        """An Isaac Lab quaternion in `xyzw` order."""
        values = np.asarray(quaternion, dtype=np.float64).reshape(-1)[:4]
        return values if self._scalar_last else wxyz_to_xyzw(values)

    def _world_pose(self, scene: Any, tracked: _TrackedBody, row: int) -> ReplayPose:
        """A tracked body's world pose now, from Isaac Lab's tensors."""
        kind, asset, link = tracked
        data = scene[asset].data
        if kind == "link":
            position = self._numpy(data.body_link_pos_w)[row, link]
            quaternion = self._numpy(data.body_link_quat_w)[row, link]
        else:
            position = self._numpy(data.root_link_pos_w)[row]
            quaternion = self._numpy(data.root_link_quat_w)[row]
        return ReplayPose(
            position=np.asarray(position, dtype=np.float64).reshape(3),
            orientation=self._xyzw(quaternion).astype(np.float32),
        )

    def _episode_cache(
        self, isaac: Any, bodies: dict[str, _TrackedBody | ReplayPose]
    ) -> _IsaacSimEpisodeCache:
        """The values that the replay methods read at each step of an episode."""
        scene = isaac.scene
        robot = scene[self._config.robot]
        joints = len(robot.joint_names)
        manager = getattr(isaac, "termination_manager", None)
        return _IsaacSimEpisodeCache(
            bodies=bodies,
            joints=joints,
            arm=joints if self._arm is None else min(self._arm, joints),
            links=tuple(str(name) for name in robot.body_names),
            objects=tuple(sorted(str(name) for name in scene.rigid_objects)),
            terms=tuple(str(term) for term in (getattr(manager, "active_terms", None) or ())),
            sensors=self._contact_sensors(scene, robot.cfg.prim_path),
            cameras=self._extra_cameras(scene, self._spec_cameras),
        )

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        """An Isaac Lab tensor as a numpy array on the host.

        Isaac Lab 3 returns warp arrays wrapped in a `ProxyArray`, whose `torch`
        property is a torch tensor. Earlier releases return torch tensors.
        """
        value = getattr(value, "torch", value)
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _resized(values: np.ndarray, size: int) -> np.ndarray:
        """`values` cut or padded with zeros to `size` values."""
        fitted = np.zeros(size)
        count = min(size, len(values))
        fitted[:count] = values[:count]
        return fitted

    @staticmethod
    def _row(data: Any, name: str, row: int, size: int | None) -> np.ndarray:
        """Row `row` of the `(num_envs, ...)` tensor `data.<name>`, flat.

        The row is cut or padded to `size` values, so that a channel keeps its width
        when Isaac Lab reports less, or zeros when it reports nothing. A `size` of None
        keeps the row as it is.
        """
        value = getattr(data, name, None)
        array = IsaacSimSimulator._numpy(value) if value is not None else np.zeros(0)
        if array.size == 0:
            return np.zeros(size or 0)
        flat = array.reshape(array.shape[0], -1)[row] if array.ndim > 1 else array.reshape(-1)
        flat = np.asarray(flat, dtype=np.float64)
        return flat if size is None else IsaacSimSimulator._resized(flat, size)

    @staticmethod
    def _isaaclab_orders_xyzw() -> bool:
        """Whether the running Isaac Lab orders quaternions `xyzw`."""
        # The SDK does not depend on Isaac Sim. An Isaac Sim benchmark image installs it.
        import isaaclab  # ty: ignore[unresolved-import]

        major = str(getattr(isaaclab, "__version__", "0")).split(".")[0]
        return major.isdigit() and int(major) >= 1

    @staticmethod
    def _arm_joints(config: BenchmarkConfig) -> int | None:
        """The number of joints of one arm, from the spec's joint action or joint observation.

        None means that the spec counts no joints, so all joints belong to the arm.
        """
        embodiment = config.spec.embodiment
        if isinstance(embodiment.action, JointActionSpace):
            return embodiment.action.dof // embodiment.action.arm_count
        joints = embodiment.proprioception.joint_pos
        if joints is not None:
            return joints.dof // joints.arm_count
        return None

    @staticmethod
    def _is_under(path: str, pattern: str) -> bool:
        """Whether the prim at `path` is the prim that `pattern` matches, or below it.

        Isaac Lab writes an asset's `cfg.prim_path` as a regular expression over the
        envs, such as `/World/envs/env_.*/robot`.
        """
        return re.fullmatch(f"{pattern}(/.*)?", path) is not None

    def _body_labels(self, tracked: _TrackedBody | None) -> tuple[ReplaySceneBodyRole, str | None]:
        """A body's role and the task's name for it, from what Isaac Lab tracks for it.

        A rigid object is an object, named by the scene. A link of the config's robot is
        the robot. A link of another articulation, such as a cabinet's door, is an object
        that the task does not name. A body that Isaac Lab does not move is a fixture, and
        is `TABLE`.
        """
        if tracked is None:
            return ReplaySceneBodyRole.FIXTURE, TABLE
        kind, asset, _index = tracked
        if kind == "root":
            return ReplaySceneBodyRole.OBJECT, asset
        if asset == self._config.robot:
            return ReplaySceneBodyRole.ROBOT, None
        return ReplaySceneBodyRole.OBJECT, None

    @staticmethod
    def _add_touch_sensors(
        env_cfg: Any, scene_cfg: Any, config: BenchmarkIsaacSimSimulatorConfig
    ) -> None:
        """Add a contact sensor to each of `config.touch_links`, if the config lists any.

        A gripper link's sensor is filtered to the assets that are not objects, the
        table included, and an arm link's sensor to every asset in the env other than
        an articulation, such as the robot. Each sensor keeps every physics step of an
        action.
        """
        if config.touch_links is None:
            return
        from isaaclab.sensors import ContactSensorCfg  # ty: ignore[unresolved-import]

        robot = getattr(scene_cfg, config.robot, None)
        if robot is None:
            raise ValueError(f"the env's scene config has no robot {config.robot!r}")
        assets = IsaacSimSimulator._touch_assets(scene_cfg)
        fixtures = [path for path, is_object in assets if not is_object]
        everything = [path for path, _is_object in assets]
        for link, part in config.touch_links.items():
            setattr(
                scene_cfg,
                IsaacSimSimulator._touch_sensor_name(link),
                ContactSensorCfg(
                    prim_path=f"{robot.prim_path}/{link}",
                    filter_prim_paths_expr=fixtures if part == GRIPPER else everything,
                    history_length=int(getattr(env_cfg, "decimation", 1)),
                ),
            )

    @staticmethod
    def _touch_assets(scene_cfg: Any) -> list[tuple[str, bool]]:
        """Each asset in an env other than an articulation, with whether it is an object.

        An asset is named by its prim path. An object is a rigid object other than the
        table: a task can make its table a rigid object, and the table is never one of
        the objects that the hooks report the gripper touching. An asset outside the
        envs, such as the ground or a light, is left out.
        """
        from isaaclab.assets import (  # ty: ignore[unresolved-import]
            ArticulationCfg,
            AssetBaseCfg,
            RigidObjectCfg,
        )

        found = []
        for name, asset in vars(scene_cfg).items():
            if not isinstance(asset, AssetBaseCfg) or isinstance(asset, ArticulationCfg):
                continue
            if "{ENV_REGEX_NS}" in asset.prim_path:
                is_object = isinstance(asset, RigidObjectCfg) and name != TABLE
                found.append((asset.prim_path, is_object))
        return found

    def _warn_unnamed_touches(self, link: str, filters: int, forces: int) -> None:
        """Log once that a link's touches are left out.

        The sensor's forces do not match its filters one to one, so no force has a name.
        """
        if link in self._unnamed_touches:
            return
        self._unnamed_touches.add(link)
        _logger.warning(
            "the touch sensor on %s has %d filters but %d forces, so its touches are left out",
            link,
            filters,
            forces,
        )

    @staticmethod
    def _touch_sensor_name(link: str) -> str:
        """The scene name of the touch sensor on `link`."""
        return _TOUCH_SENSOR_PREFIX + link.replace("/", "__")

    @staticmethod
    def _touch_filter_names(scene: Any, filters: Any) -> list[str]:
        """The name of the asset of each filter path.

        A rigid object keeps its scene name, and another asset is `TABLE`. A path is
        matched by its last part, because Isaac Lab fills in the env's part of a path
        when it builds the scene.
        """
        rigid = {
            IsaacSimSimulator._last_part(asset.cfg.prim_path): str(name)
            for name, asset in scene.rigid_objects.items()
        }
        return [rigid.get(IsaacSimSimulator._last_part(path), TABLE) for path in filters or ()]

    @staticmethod
    def _last_part(path: Any) -> str:
        """The last part of a prim path."""
        return str(path).rstrip("/").rsplit("/", 1)[-1]

    @staticmethod
    def _tracked_body(path: str, scene: Any) -> _TrackedBody | None:
        """What Isaac Lab tracks for the rigid body at `path`, or None for a fixed body."""
        for name, asset in scene.rigid_objects.items():
            if IsaacSimSimulator._is_under(path, asset.cfg.prim_path):
                return ("root", name, 0)
        leaf = path.rsplit("/", 1)[-1]
        for name, asset in scene.articulations.items():
            links = list(asset.body_names)
            if IsaacSimSimulator._is_under(path, asset.cfg.prim_path) and leaf in links:
                return ("link", name, links.index(leaf))
        return None

    @staticmethod
    def _origin(scene: Any, row: int) -> np.ndarray:
        """The world position of env row `row`'s origin."""
        origins = IsaacSimSimulator._numpy(scene.env_origins)
        return np.asarray(origins, dtype=np.float64).reshape(-1, 3)[row]

    @staticmethod
    def _contact_sensors(scene: Any, robot_path: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """The contact sensors on the robot whose forces the scalars plot.

        A sensor is on the robot when its prim is below the robot's prim. Of the
        sensors on one body, only the sensor with the most filtered objects is read,
        because its read contains the forces of the others. A task can declare a
        sensor for each pair of objects, and reading each of them is slow and can
        fail. Each object is named by the last part of its filter's prim path. A
        contact sensor is told apart from a camera by `filter_prim_paths_expr` in its
        config, so that this method reads no sensor's data.
        """
        chosen: dict[str, tuple[str, tuple[str, ...]]] = {}
        for name, sensor in scene.sensors.items():
            if str(name).startswith(_TOUCH_SENSOR_PREFIX):
                continue
            cfg = sensor.cfg
            filters = getattr(cfg, "filter_prim_paths_expr", None)
            if filters is None or not IsaacSimSimulator._is_under(cfg.prim_path, robot_path):
                continue
            objects = tuple(str(path).rstrip("/").rsplit("/", 1)[-1] for path in filters)
            held = chosen.get(cfg.prim_path)
            if held is None or len(objects) > len(held[1]):
                chosen[cfg.prim_path] = (str(name), objects)
        return tuple(chosen.values())

    @staticmethod
    def _extra_cameras(scene: Any, spec: tuple[str, ...]) -> tuple[str, ...]:
        """The camera sensors that produce none of the spec's cameras, in scene order.

        A camera sensor produces a spec camera when the sensor's name starts with the
        spec camera's name, such as `wrist_cam` for `wrist`. Each spec camera that no
        sensor's name starts with is taken to be produced by the next sensor that
        matches no spec camera. The camera sensors that are left are the extra ones.
        """
        cameras = [
            str(name)
            for name, sensor in scene.sensors.items()
            if "rgb" in tuple(getattr(sensor.cfg, "data_types", ()))
        ]
        unproduced = [camera for camera in spec if not any(n.startswith(camera) for n in cameras)]
        unmatched = [
            name for name in cameras if not any(name.startswith(camera) for camera in spec)
        ]
        return tuple(unmatched[len(unproduced) :])

    @staticmethod
    def _halved(frame: np.ndarray) -> np.ndarray:
        """An RGB frame box-averaged by `_IMAGE_DIVISOR` along each side.

        Raises:
            ValueError: If a side of the frame does not divide by `_IMAGE_DIVISOR`.
        """
        pixels = np.asarray(frame)[..., :3].astype(np.float32)
        height, width = pixels.shape[:2]
        if height % _IMAGE_DIVISOR or width % _IMAGE_DIVISOR:
            raise ValueError(
                f"cannot box-average a {height}x{width} frame by {_IMAGE_DIVISOR}: "
                "both sides must divide evenly"
            )
        blocks = pixels.reshape(
            height // _IMAGE_DIVISOR, _IMAGE_DIVISOR, width // _IMAGE_DIVISOR, _IMAGE_DIVISOR, -1
        )
        return np.ascontiguousarray(blocks.mean(axis=(1, 3)).astype(np.uint8))

    @staticmethod
    def _matrix(transform: Any) -> np.ndarray:
        """A USD transform as a 4x4 matrix that multiplies column vectors.

        USD's matrices multiply row vectors, so the matrix is transposed.
        """
        return np.array([[float(value) for value in row] for row in transform], dtype=np.float64).T

    @staticmethod
    def _pose(matrix: np.ndarray, metres: float) -> ReplayPose:
        """A 4x4 transform's position in metres and its rotation, with any scale removed."""
        rotation = matrix[:3, :3]
        scale = np.linalg.norm(rotation, axis=0)
        scale[scale == 0.0] = 1.0
        return ReplayPose(
            position=(matrix[:3, 3] * metres).astype(np.float32),
            orientation=xyzw_from_rotation_matrix(rotation / scale),
        )

    @staticmethod
    def _skip_reason(prim: Any, geom: Any) -> str | None:
        """Why a drawable prim is left out of the scene, or None to keep it.

        A prim whose purpose is `guide` or `proxy` is not for rendering, and Isaac Sim
        gives a collision shape the purpose `guide`. An invisible prim is not rendered.
        A prim below a `collisions` prim is a collision shape of an Isaac asset.
        """
        imageable = geom.Imageable(prim)
        purpose = str(imageable.ComputePurpose())
        if purpose in ("guide", "proxy"):
            return f"{purpose}-purpose prims"
        if str(imageable.ComputeVisibility()) == "invisible":
            return "invisible prims"
        if "collisions" in str(prim.GetPath()).split("/"):
            return "collision prims"
        return None

    @staticmethod
    def _triangles(counts: Any, indices: Any) -> tuple[np.ndarray, list[list[int]]]:
        """USD's polygons as triangles, and the rows of the triangles of each polygon.

        A polygon of n vertices becomes n - 2 triangles around its first vertex. A
        `UsdGeom.Subset` names polygons, so its triangles are found through the rows.
        """
        flat = np.asarray(indices, dtype=np.int64).reshape(-1)
        triangles: list[tuple[int, int, int]] = []
        rows: list[list[int]] = []
        cursor = 0
        for count in np.asarray(counts, dtype=np.int64).reshape(-1):
            size = int(count)
            face = flat[cursor : cursor + size]
            start = len(triangles)
            triangles.extend(
                (int(face[0]), int(face[i]), int(face[i + 1])) for i in range(1, size - 1)
            )
            rows.append(list(range(start, len(triangles))))
            cursor += size
        return np.asarray(triangles, dtype=np.int64).reshape(-1, 3), rows

    @staticmethod
    def _prim_geometry(
        prim: Any, geom: Any
    ) -> tuple[np.ndarray, np.ndarray, list[list[int]] | None] | None:
        """A prim's vertices and triangles in its own frame, or None for a shape not drawn.

        A mesh, a cube and a cylinder are drawn. A sphere, a capsule or a cone is left
        out, so that a replay never draws a shape that the simulator does not have.
        The third value contains the triangles of each polygon of a mesh, and is None for
        a cube or a cylinder, which have no subsets.
        """
        if prim.IsA(geom.Mesh):
            mesh = geom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            if points is None or counts is None or indices is None or len(points) == 0:
                return None
            triangles, rows = IsaacSimSimulator._triangles(counts, indices)
            if len(triangles) == 0:
                return None
            vertices = np.asarray([[float(v) for v in point] for point in points], dtype=np.float64)
            return vertices, triangles, rows
        if prim.IsA(geom.Cube):
            size = geom.Cube(prim).GetSizeAttr().Get()
            box = ReplayMesh.box([float(2.0 if size is None else size) / 2.0] * 3)
            return np.asarray(box.vertices, dtype=np.float64), np.asarray(box.faces), None
        if prim.IsA(geom.Cylinder):
            cylinder = geom.Cylinder(prim)
            radius = cylinder.GetRadiusAttr().Get()
            height = cylinder.GetHeightAttr().Get()
            built = ReplayMesh.cylinder(
                float(1.0 if radius is None else radius),
                float(2.0 if height is None else height) / 2,
            )
            vertices = np.asarray(built.vertices, dtype=np.float64)
            # `ReplayMesh.cylinder` is built along z, so a cylinder along x or y is
            # turned onto its axis.
            axis = str(cylinder.GetAxisAttr().Get() or "Z")
            if axis == "X":
                vertices = vertices[:, [2, 1, 0]] * np.array([1.0, 1.0, -1.0])
            elif axis == "Y":
                vertices = vertices[:, [0, 2, 1]] * np.array([1.0, 1.0, -1.0])
            return vertices, np.asarray(built.faces), None
        return None

    @staticmethod
    def _owner(prim: Any, physics: Any) -> Any:
        """The nearest ancestor of `prim` that PhysX moves, or `prim` when there is none."""
        node = prim
        while node and node.IsValid():
            if node.HasAPI(physics.RigidBodyAPI):
                return node
            node = node.GetParent()
        return prim

    @staticmethod
    def _unique_name(name: str, path: str, seen: dict[str, str]) -> str:
        """A body name that no other body has, because the replay stores poses by name.

        A clash falls back to the last two parts of the path, and then to the path.
        """
        segments = path.strip("/").split("/")
        for candidate in (name, "_".join(segments[-2:]), "_".join(segments)):
            if seen.get(candidate, path) == path:
                seen[candidate] = path
                return candidate
        return "_".join(segments)

    @staticmethod
    def _material_colour(prim: Any, shade: Any) -> tuple[float, float, float] | None:
        """The constant colour of the material bound to `prim`, or None when it has none."""
        material, _relationship = shade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:
            return None
        for context in ("", "mdl"):
            source = material.ComputeSurfaceSource(context)
            shader = source[0] if isinstance(source, tuple) else source
            if not shader:
                continue
            for name in _COLOUR_INPUTS:
                channel = shader.GetInput(name)
                value = channel.Get() if channel else None
                if value is None:
                    continue
                try:
                    channels = [float(component) for component in value]
                except TypeError:
                    channels = [float(value)] * 3
                if len(channels) >= 3:
                    return channels[0], channels[1], channels[2]
        return None

    @staticmethod
    def _display_colour(prim: Any, geom: Any) -> tuple[float, float, float, float]:
        """The prim's first display colour, or grey when it has none."""
        values = geom.Gprim(prim).GetDisplayColorAttr().Get()
        if values is None or len(values) == 0:
            return _GREY
        red, green, blue = (float(channel) for channel in tuple(values[0])[:3])
        return red, green, blue, 1.0

    @staticmethod
    def _part(vertices: np.ndarray, faces: np.ndarray, colour: Any) -> ReplayScenePart:
        """A part that contains only the vertices that its faces use."""
        used = np.unique(faces)
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        return ReplayScenePart(
            mesh=ReplayMesh(
                vertices=vertices[used].astype(np.float32), faces=remap[faces].astype(np.uint32)
            ),
            color=colour,
        )

    @staticmethod
    def _parts(
        prim: Any,
        shape: tuple[np.ndarray, np.ndarray, list[list[int]] | None],
        vertices: np.ndarray,
        modules: tuple[Any, Any],
    ) -> list[ReplayScenePart]:
        """A part for each material of a prim, with `vertices` in its body's frame.

        The colour is the material's when one is bound to the prim. Otherwise each
        `UsdGeom.Subset` of a mesh that has a material is its own part in that
        material's colour. The rest is drawn in the prim's display colour, or grey.
        """
        geom, shade = modules
        _, triangles, rows = shape
        colour = IsaacSimSimulator._material_colour(prim, shade)
        if colour is not None:
            return [IsaacSimSimulator._part(vertices, triangles, (*colour, 1.0))]
        fallback = IsaacSimSimulator._display_colour(prim, geom)
        parts = []
        covered = np.zeros(len(triangles), dtype=bool)
        if rows is not None:
            for subset in geom.Subset.GetAllGeomSubsets(geom.Imageable(prim)):
                subset_colour = IsaacSimSimulator._material_colour(subset.GetPrim(), shade)
                faces = subset.GetIndicesAttr().Get() or ()
                picked = [row for face in faces for row in rows[int(face)]]
                if subset_colour is None or len(picked) == 0:
                    continue
                covered[picked] = True
                parts.append(
                    IsaacSimSimulator._part(vertices, triangles[picked], (*subset_colour, 1.0))
                )
        if not covered.all():
            parts.append(IsaacSimSimulator._part(vertices, triangles[~covered], fallback))
        return parts


__all__ = [
    "BenchmarkIsaacSimSimulatorConfig",
    "IsaacSimSimulator",
]
