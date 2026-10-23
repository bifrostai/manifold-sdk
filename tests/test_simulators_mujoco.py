"""Tests for `MujocoSimulator`, against a stand-in for a MuJoCo handle."""

from __future__ import annotations

import sys
from enum import IntEnum
from types import SimpleNamespace

import numpy as np
import pytest

from manifold.benchmark.simulators import Simulator
from manifold.benchmark.simulators.mujoco import BenchmarkMujocoSimulatorConfig, MujocoSimulator
from manifold.replay import ReplaySceneBodyRole

_BOX = 6
_PLANE = 0

# MuJoCo's geom types, as `mujoco.mjtGeom` numbers them.
mjtGeom = IntEnum(  # noqa: N816 - named as `mujoco.mjtGeom`
    "mjtGeom",
    {
        "mjGEOM_PLANE": 0,
        "mjGEOM_HFIELD": 1,
        "mjGEOM_SPHERE": 2,
        "mjGEOM_CAPSULE": 3,
        "mjGEOM_ELLIPSOID": 4,
        "mjGEOM_CYLINDER": 5,
        "mjGEOM_BOX": 6,
        "mjGEOM_MESH": 7,
        "mjGEOM_SDF": 8,
    },
)


@pytest.fixture(autouse=True)
def _mujoco(monkeypatch):
    """A stand-in for the `mujoco` package, which the SDK's tests do not install."""
    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(mjtGeom=mjtGeom))


_CONFIG = BenchmarkMujocoSimulatorConfig(
    name="mujoco",
    path="env.sim",
    cameras={"agentview": "frontview"},
    geom_groups=(1, 2),
    arm_prefix="robot0_",
    gripper_prefix="gripper0_",
    robot_prefixes=("robot0_", "gripper0_", "mount0_"),
    contact_prefixes=("gripper0_",),
    object_body_suffix="_main",
)


def _env(geoms: list[dict], joints: tuple[tuple[str, int], ...] = ()) -> SimpleNamespace:
    """An env whose handle holds `geoms`, each on its own body, and hinge `joints` on bodies.

    Body 0 is the world. Geom `i` is on body `i + 1`.
    """
    count = len(geoms)
    bodies = {0: "world", **{index + 1: geom["body"] for index, geom in enumerate(geoms)}}
    joint_names = dict(enumerate(name for name, _ in joints))
    jntnum = np.zeros(count + 1, dtype=int)
    for _, body_id in joints:
        jntnum[body_id] += 1
    model = SimpleNamespace(
        ngeom=count,
        nbody=count + 1,
        njnt=len(joints),
        nq=len(joints),
        nv=len(joints),
        geom_group=np.array([geom["group"] for geom in geoms]),
        geom_type=np.array([geom.get("type", _BOX) for geom in geoms]),
        geom_size=np.full((count, 3), 0.1),
        geom_bodyid=np.arange(1, count + 1),
        geom_pos=np.zeros((count, 3)),
        geom_quat=np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)),
        geom_rgba=np.tile([0.5, 0.5, 0.5, 1.0], (count, 1)),
        geom_matid=np.full(count, -1),
        body_jntnum=jntnum,
        jnt_qposadr=np.arange(len(joints)),
        jnt_dofadr=np.arange(len(joints)),
        body=lambda body_id: SimpleNamespace(name=bodies.get(body_id, "")),
        joint=lambda joint_id: SimpleNamespace(name=joint_names.get(joint_id, "")),
        camera=lambda name: SimpleNamespace(id={"frontview": 0}[name]),
    )
    data = SimpleNamespace(
        xpos=np.arange((count + 1) * 3, dtype=float).reshape(-1, 3),
        xquat=np.tile([1.0, 0.0, 0.0, 0.0], (count + 1, 1)),
        cam_xpos=np.array([[1.0, 2.0, 3.0]]),
        cam_xmat=np.eye(3).reshape(1, 9),
        qpos=np.arange(len(joints), dtype=float),
        qvel=np.zeros(len(joints)),
        qacc=np.zeros(len(joints)),
        qfrc_actuator=np.zeros(len(joints)),
        actuator_force=np.zeros(1),
        ctrl=np.zeros(1),
        cfrc_ext=np.ones((count + 1, 6)),
        sensordata=np.zeros(6),
    )
    return SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(model=model, data=data)))


def _attached(env: SimpleNamespace) -> MujocoSimulator:
    simulator = MujocoSimulator(_CONFIG)
    simulator.attach(env, [0])
    return simulator


def test_the_scene_holds_the_bodies_with_a_geom_in_the_groups_given():
    env = _env(
        [
            {"body": "robot0_link0", "group": 1},
            {"body": "lid_handle", "group": 2},
            {"body": "robot0_link0_collision", "group": 0},
            {"body": "hidden_marker", "group": 3},
        ]
    )

    simulator = _attached(env)

    assert isinstance(simulator, Simulator)
    assert [body.name for body in simulator.replay_bodies(0)] == ["robot0_link0", "lid_handle"]


def test_each_body_has_its_role_and_the_tasks_name_for_an_object():
    env = _env(
        [
            {"body": "robot0_link0", "group": 1},
            {"body": "bowl_main", "group": 1},
            {"body": "table_main", "group": 1},
            {"body": "drawer", "group": 1},
            {"body": "study_table", "group": 1},
            {"body": "stove_1_main", "group": 1},
            {"body": "stove_1_burner", "group": 1},
        ],
        joints=(("robot0_joint1", 1), ("bowl_joint0", 2), ("drawer_slide", 4)),
    )

    labels = {body.name: (body.role, body.object) for body in _attached(env).replay_bodies(0)}

    assert labels == {
        "robot0_link0": (ReplaySceneBodyRole.ROBOT, None),
        "bowl_main": (ReplaySceneBodyRole.OBJECT, "bowl"),
        "table_main": (ReplaySceneBodyRole.FIXTURE, "table"),
        "drawer": (ReplaySceneBodyRole.OBJECT, None),
        "study_table": (ReplaySceneBodyRole.FIXTURE, "table"),
        "stove_1_main": (ReplaySceneBodyRole.FIXTURE, "stove_1"),
        "stove_1_burner": (ReplaySceneBodyRole.FIXTURE, "stove_1"),
    }


def test_the_contacts_name_the_arm_and_the_gripper_and_what_each_presses_on(monkeypatch):
    env = _env(
        [
            {"body": "robot0_link1", "group": 1},
            {"body": "gripper0_finger", "group": 1},
            {"body": "bowl_main", "group": 1},
            {"body": "study_table", "group": 1},
        ],
        joints=(("robot0_joint1", 1), ("bowl_joint0", 3)),
    )
    # Geom i is on body i + 1: the arm, the finger, the bowl and the table.
    pairs = [(0, 2, 1.0), (1, 2, 1.0), (1, 3, 1.0), (0, 3, 0.05)]
    data = env.env.sim.data
    data.ncon = len(pairs)
    data.contact = [SimpleNamespace(geom1=first, geom2=second) for first, second, _ in pairs]

    def contact_force(_model, _data, index, wrench):
        wrench[0] = pairs[index][2]

    monkeypatch.setitem(
        sys.modules, "mujoco", SimpleNamespace(mjtGeom=mjtGeom, mj_contactForce=contact_force)
    )
    simulator = _attached(env)
    simulator.replay_bodies(0)

    # The gripper touching the bowl is the hooks' to report, and 0.05 N is no touch.
    assert simulator.replay_contacts(0) == {("arm", "bowl"), ("gripper", "table")}


def test_a_drawn_plane_is_left_out_of_the_scene():
    simulator = _attached(_env([{"body": "floor", "group": 1, "type": _PLANE}]))

    assert simulator.replay_bodies(0) == ()


def test_the_poses_are_each_scene_bodys_world_pose_in_xyzw():
    simulator = _attached(_env([{"body": "link0", "group": 1}, {"body": "marker", "group": 3}]))
    simulator.replay_bodies(0)

    poses = simulator.replay_poses(0)

    assert list(poses) == ["link0"]
    np.testing.assert_array_equal(poses["link0"].position, [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(poses["link0"].orientation, [0.0, 0.0, 0.0, 1.0])


def test_a_camera_pose_is_in_opencv_axes():
    simulator = _attached(_env([{"body": "link0", "group": 1}]))

    pose = simulator.replay_extrinsics(0)["agentview"]

    np.testing.assert_array_equal(pose.position, [1.0, 2.0, 3.0])
    # MuJoCo's camera looks down -z; turning it to look down +z is a half turn about x.
    np.testing.assert_allclose(np.abs(pose.orientation), [1.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_the_readings_sort_the_joints_into_the_arm_the_gripper_and_the_objects():
    env = _env(
        [
            {"body": "robot0_link1", "group": 1},
            {"body": "gripper0_finger", "group": 1},
            {"body": "bowl_1_main", "group": 1},
        ],
        joints=(("robot0_joint1", 1), ("gripper0_finger_joint", 2), ("bowl_1_joint0", 3)),
    )
    simulator = _attached(env)
    simulator.replay_bodies(0)

    scalars = {scalar.name: scalar for scalar in simulator.replay_scalars(0, np.zeros(3))}

    assert scalars["joints/0"].value == 0.0
    assert scalars["qpos_gripper/0"].value == 1.0
    assert scalars["qpos_objects/0"].value == 2.0
    assert scalars["object_dist/bowl_1_main"].tab == "Task"
    assert "object_dist/gripper0_finger" not in scalars
    assert scalars["contacts/gripper0_finger"].tab == "Physics"


def test_each_read_uses_the_handle_that_the_env_has_now():
    env = _env([{"body": "robot0_link0", "group": 1}])
    simulator = _attached(env)
    simulator.replay_bodies(0)
    # robosuite builds a new handle on each reset, and frees the old one.
    replaced = _env([{"body": "robot0_link0", "group": 1}]).env.sim
    replaced.data.xpos = replaced.data.xpos + 10.0
    env.env.sim = replaced

    poses = simulator.replay_poses(0)

    assert poses["robot0_link0"].position.tolist() == replaced.data.xpos[1].tolist()


def test_a_mujoco_handle_holds_one_env():
    simulator = MujocoSimulator(_CONFIG)

    with pytest.raises(ValueError, match="one env"):
        simulator.attach(_env([]), [0, 1])
