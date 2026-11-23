from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from manifold.benchmark.config import BenchmarkConfig
from manifold.benchmark.simulators import ARM, BOX_VERTICES, GRIPPER, TABLE, make_simulator
from manifold.benchmark.simulators.genesis import (
    BenchmarkGenesisSimulatorConfig,
    GenesisSimulator,
)
from manifold.replay import ReplaySceneBodyRole
from tests.test_benchmark_runtime import _raw_config

_CONFIG = BenchmarkGenesisSimulatorConfig(
    name="genesis",
    path="gym_env.unwrapped",
    robot="agent.robot",
    arm_joints=("joint1",),
    gripper_joints=("finger_joint1", "finger_joint2"),
    touch_links={"link1": "arm", "hand": "gripper", "left_finger": "gripper"},
)

_PADS = ("left_finger",)


class _Plane:
    pass


class _Box:
    pass


@pytest.fixture(autouse=True)
def genesis(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stand-in for the `genesis.morphs` type that the simulator checks for."""
    module = cast(Any, ModuleType("genesis"))
    module.morphs = SimpleNamespace(Plane=_Plane)
    monkeypatch.setitem(sys.modules, "genesis", module)


def _vgeom(colour: tuple[float, ...] = (0.1, 0.2, 0.3, 1.0)) -> SimpleNamespace:
    surface = SimpleNamespace(get_rgba=lambda: SimpleNamespace(color=colour))
    return SimpleNamespace(
        init_vverts=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        init_vfaces=np.array([[0, 1, 2]]),
        init_pos=np.array([0.0, 0.0, 0.5]),
        init_quat=np.array([0.0, 0.0, 0.0, 1.0]),
        vmesh=SimpleNamespace(surface=surface),
    )


def _link(index: int, name: str, *, fixed: bool = False, drawn: bool = True) -> SimpleNamespace:
    return SimpleNamespace(idx=index, name=name, is_fixed=fixed, vgeoms=[_vgeom()] if drawn else [])


def _entity(index: int, links: list[SimpleNamespace], morph: Any = None) -> SimpleNamespace:
    return SimpleNamespace(idx=index, links=links, morph=morph or _Box())


def _robot() -> SimpleNamespace:
    robot = _entity(
        0,
        [
            _link(0, "link0", fixed=True),
            _link(1, "link1"),
            _link(2, "hand"),
            _link(3, "left_finger"),
        ],
    )
    joints = {"joint1": [0], "finger_joint1": [1], "finger_joint2": [2]}
    robot.get_joint = lambda name: SimpleNamespace(dofs_idx_local=joints[name])
    robot.get_dofs_position = lambda: np.array([0.5, 0.04, 0.03])
    robot.get_dofs_velocity = lambda: np.array([0.1, 0.0, 0.0])
    robot.get_dofs_control_force = lambda: np.array([7.0, 1.0, 2.0])
    return robot


def _scene(contacts: dict[str, np.ndarray] | None = None, n_envs: int = 0) -> SimpleNamespace:
    """A scene with the robot, a cube, two blocks in a list, a table, a floor and a stray body."""
    robot = _robot()
    cube = _entity(1, [_link(4, "baselink")])
    blocks = [_entity(2, [_link(5, "baselink")]), _entity(3, [_link(6, "baselink")])]
    table = _entity(4, [_link(7, "baselink", fixed=True)])
    floor = _entity(5, [_link(8, "baselink", fixed=True)], morph=_Plane())
    stray_collider = _entity(6, [_link(9, "baselink", fixed=True, drawn=False)])
    stray_object = _entity(7, [_link(10, "baselink")])
    entities = [robot, cube, *blocks, table, floor, stray_collider, stray_object]
    positions = np.arange(33, dtype=np.float64).reshape(11, 3)
    orientations = np.tile([0.8, 0.0, 0.6, 0.0], (11, 1))
    found = contacts or {
        "link_a": np.zeros(0),
        "link_b": np.zeros(0),
        "force": np.zeros((0, 3)),
    }
    solver = SimpleNamespace(
        n_envs=n_envs,
        entities=entities,
        get_links_pos=lambda: positions,
        get_links_quat=lambda: orientations,
        collider=SimpleNamespace(get_contacts=lambda as_tensor, to_torch: found),
    )
    task_env = SimpleNamespace(
        scene=SimpleNamespace(rigid_solver=solver),
        agent=SimpleNamespace(robot=robot),
        cube=cube,
        _blocks=blocks,
        table=table,
        plane=floor,
    )
    return SimpleNamespace(gym_env=SimpleNamespace(unwrapped=task_env))


def _attached(env: SimpleNamespace, pads: tuple[str, ...] = _PADS) -> GenesisSimulator:
    simulator = GenesisSimulator(_CONFIG, finger_pads=pads)
    simulator.attach(env, [0])
    return simulator


def _contacts(*pairs: tuple[int, int, float]) -> dict[str, np.ndarray]:
    return {
        "link_a": np.array([first for first, _, _ in pairs]),
        "link_b": np.array([second for _, second, _ in pairs]),
        "force": np.array([[0.0, 0.0, force] for _, _, force in pairs]),
    }


def test_each_drawn_link_is_a_body_named_by_the_attribute_that_holds_its_entity():
    simulator = _attached(_scene())

    bodies = {body.name: body for body in simulator.replay_bodies(0)}

    assert sorted(bodies) == [
        "blocks_0",
        "blocks_1",
        "cube",
        "entity7_baselink",
        "hand",
        "left_finger",
        "link0",
        "link1",
        "table",
    ]
    assert (bodies["cube"].role, bodies["cube"].object) == (ReplaySceneBodyRole.OBJECT, "cube")
    assert bodies["blocks_1"].object == "blocks_1"
    assert (bodies["table"].role, bodies["table"].object) == (ReplaySceneBodyRole.FIXTURE, "table")
    assert (bodies["hand"].role, bodies["hand"].object) == (ReplaySceneBodyRole.ROBOT, None)
    # An entity that no attribute holds has no name for the task.
    assert bodies["entity7_baselink"].object is None


def test_a_part_keeps_its_vertices_its_pose_in_the_link_and_its_colour():
    simulator = _attached(_scene())

    bodies = {body.name: body for body in simulator.replay_bodies(0)}
    (part,) = bodies["cube"].parts

    assert part.mesh.vertices[1].tolist() == [1.0, 0.0, 0.0]
    assert part.position.tolist() == [0.0, 0.0, 0.5]
    # Genesis orders the quaternion wxyz, and the replay orders it xyzw.
    assert part.orientation.tolist() == [0.0, 0.0, 1.0, 0.0]
    assert part.color == pytest.approx((0.1, 0.2, 0.3, 1.0))


def test_a_part_with_too_many_vertices_is_its_bounding_box_in_the_links_frame():
    vgeom = _vgeom()
    vgeom.init_vverts = np.zeros((BOX_VERTICES + 1, 3))
    vgeom.init_vverts[1] = [2.0, 1.0, 0.2]
    # A quarter turn about z, in Genesis' wxyz order.
    vgeom.init_quat = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])

    part = GenesisSimulator._part(vgeom)

    assert len(part.mesh.vertices) == 8
    assert part.position.tolist() == pytest.approx([-0.5, 1.0, 0.6])
    assert np.abs(part.mesh.vertices).max(axis=0).tolist() == pytest.approx([0.5, 1.0, 0.1])
    assert part.color == pytest.approx((0.1, 0.2, 0.3, 1.0))


def test_an_image_surface_gives_its_mean_colour():
    image = np.zeros((2, 2, 4), dtype=np.uint8)
    image[..., 0] = 255
    texture = SimpleNamespace(image_array=image, image_color=(1.0, 1.0, 1.0, 1.0))

    colour = GenesisSimulator._surface_colour(SimpleNamespace(get_rgba=lambda: texture))

    assert colour == pytest.approx((1.0, 0.0, 0.0, 1.0))


def test_a_pose_is_its_links_position_and_orientation():
    simulator = _attached(_scene())
    simulator.replay_bodies(0)

    poses = simulator.replay_poses(0)

    assert poses["cube"].position.tolist() == [12.0, 13.0, 14.0]
    assert poses["cube"].orientation.tolist() == pytest.approx([0.0, 0.6, 0.0, 0.8])


def test_a_touch_is_named_by_the_robot_part_and_the_body():
    simulator = _attached(
        _scene(
            _contacts(
                # The arm touches the table and the cube.
                (1, 7, 1.0),
                (4, 1, 1.0),
                # The hand touches the cube, which the hooks record, and the table.
                (2, 4, 1.0),
                (2, 7, 1.0),
                # The left finger, a pad, touches the second block.
                (3, 6, 1.0),
                # A brush below the touch force, and a touch of a body with no name.
                (1, 5, 0.01),
                (1, 10, 1.0),
            )
        )
    )
    simulator.replay_bodies(0)

    assert simulator.replay_contacts(0) == {
        (ARM, "table"),
        (ARM, "cube"),
        (GRIPPER, "table"),
        ("left_finger", "blocks_1"),
    }


def test_a_touch_of_an_unnamed_fixture_is_a_touch_of_the_table():
    scene = _scene(_contacts((1, 9, 1.0)))
    simulator = _attached(scene)
    simulator.replay_bodies(0)

    assert simulator.replay_contacts(0) == {(ARM, TABLE)}


def test_the_readings_are_the_arm_and_gripper_joints_of_the_robot():
    simulator = _attached(_scene())
    simulator.replay_bodies(0)

    scalars = {scalar.name: scalar.value for scalar in simulator.replay_scalars(0, None)}

    assert scalars["joints/0"] == 0.5
    assert [scalars["qpos_gripper/0"], scalars["qpos_gripper/1"]] == [0.04, 0.03]
    assert scalars["joint_torques/0"] == 7.0
    assert "joints/1" not in scalars


def test_a_finger_pad_that_the_robot_lacks_stops_the_episode():
    simulator = _attached(_scene(), pads=("left_finger", "right_finger"))

    with pytest.raises(ValueError, match="right_finger"):
        simulator.replay_bodies(0)


def test_a_touch_link_that_the_robot_lacks_stops_the_episode():
    simulator = GenesisSimulator(
        _CONFIG.model_copy(update={"touch_links": {"link1": "arm", "wrist": "arm"}})
    )
    simulator.attach(_scene(), [0])

    with pytest.raises(ValueError, match="wrist"):
        simulator.replay_bodies(0)


def test_a_scene_of_several_envs_is_refused():
    simulator = _attached(_scene(n_envs=2))

    with pytest.raises(ValueError, match="one env"):
        simulator.replay_bodies(0)


def test_a_genesis_scene_holds_one_env():
    with pytest.raises(ValueError, match="one env"):
        GenesisSimulator(_CONFIG).attach(_scene(), [0, 1])


def test_make_simulator_gives_a_genesis_simulator_the_configs_finger_pads():
    raw = _raw_config({"type": "single"})
    raw["replay"]["finger_pads"] = [
        {"body": "left_finger", "offset": [0.0, 0.0, 0.04]},
        {"body": "right_finger", "offset": [0.0, 0.0, 0.04]},
    ]
    raw["simulator"] = _CONFIG.model_dump(mode="json")

    simulator = make_simulator(BenchmarkConfig.model_validate(raw))

    assert isinstance(simulator, GenesisSimulator)
    assert simulator.replay_finger_pads() == ("left_finger", "right_finger")
