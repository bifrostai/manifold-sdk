from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from manifold.benchmark.simulators import BOX_VERTICES, Simulator
from manifold.benchmark.simulators.sapien import BenchmarkSapienSimulatorConfig, SapienSimulator

_CONFIG = BenchmarkSapienSimulatorConfig(
    name="sapien",
    path="task_env.scene",
    cameras={"head": "head_camera"},
    arm_joints=("arm_joint1", "arm_joint2"),
    gripper_joints=("finger_joint",),
)


class _RenderBodyComponent:
    def __init__(self, shapes: list[Any]) -> None:
        self.render_shapes = shapes


class _RenderCameraComponent:
    def __init__(self, name: str, model: np.ndarray) -> None:
        self._name = name
        self._model = model

    def get_name(self) -> str:
        return self._name

    def get_model_matrix(self) -> np.ndarray:
        return self._model


class _RenderShapePlane:
    pass


@pytest.fixture(autouse=True)
def sapien(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand-ins for the `sapien.render` types that the simulator checks for."""
    module = cast(Any, ModuleType("sapien"))
    module.render = SimpleNamespace(
        RenderBodyComponent=_RenderBodyComponent,
        RenderCameraComponent=_RenderCameraComponent,
        RenderShapePlane=_RenderShapePlane,
    )
    monkeypatch.setitem(sys.modules, "sapien", module)


def _pose(position=(0.0, 0.0, 0.0), wxyz=(1.0, 0.0, 0.0, 0.0)) -> SimpleNamespace:
    return SimpleNamespace(p=np.array(position), q=np.array(wxyz))


def _shape(vertices: np.ndarray, **size: Any) -> SimpleNamespace:
    part = SimpleNamespace(
        get_vertices=lambda: vertices,
        get_triangles=lambda: np.array([[0, 1, 2]]),
        material=SimpleNamespace(base_color=[0.1, 0.2, 0.3, 1.0]),
    )
    return SimpleNamespace(
        local_pose=_pose(position=(0.0, 0.0, 1.0)), get_parts=lambda: [part], **size
    )


def _entity(name: str, scene_id: int, components: list[Any], position=(0.0, 0.0, 0.0)) -> Any:
    body = next((c for c in components if isinstance(c, _RenderBodyComponent)), None)
    return SimpleNamespace(
        get_name=lambda: name,
        per_scene_id=scene_id,
        get_pose=lambda: _pose(position, (0.8, 0.0, 0.6, 0.0)),
        get_components=lambda: components,
        find_component_by_type=lambda kind: body if kind is _RenderBodyComponent else None,
    )


def _articulation(prefix: str = "") -> SimpleNamespace:
    joints = [
        SimpleNamespace(get_name=lambda n=n: prefix + n)
        for n in ("wheel", "arm_joint1", "arm_joint2", "finger_joint")
    ]
    return SimpleNamespace(
        get_active_joints=lambda: joints,
        get_qpos=lambda: np.array([9.0, 0.1, 0.2, 0.04]),
        get_qvel=lambda: np.zeros(4),
        get_qf=lambda: np.array([0.0, 1.0, 2.0, 3.0]),
    )


def _env(entities: list[Any], joint_prefix: str = "") -> SimpleNamespace:
    scene = SimpleNamespace(
        get_entities=lambda: entities,
        get_all_articulations=lambda: [_articulation(joint_prefix)],
    )
    return SimpleNamespace(task_env=SimpleNamespace(scene=scene))


def _attached(env: SimpleNamespace) -> SapienSimulator:
    simulator = SapienSimulator(_CONFIG)
    simulator.attach(env, [0])
    return simulator


def test_a_shape_is_scaled_and_posed_in_its_entitys_frame():
    unit_box = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0], [1.0, -1.0, 1.0]])
    box = _shape(unit_box, half_size=np.array([0.5, 0.25, 2.0]))
    simulator = _attached(_env([_entity("table", 1, [_RenderBodyComponent([box])])]))

    (body,) = simulator.replay_bodies(0)

    assert isinstance(simulator, Simulator)
    assert body.name == "table"
    assert body.parts[0].mesh.vertices[0].tolist() == [0.5, 0.25, 3.0]
    assert body.parts[0].color == pytest.approx((0.1, 0.2, 0.3, 1.0))


def test_a_plane_is_left_out_and_a_large_mesh_becomes_its_box():
    plane = _RenderShapePlane()
    heavy = _shape(np.random.default_rng(0).normal(size=(BOX_VERTICES + 1, 3)), scale=np.ones(3))
    simulator = _attached(
        _env(
            [
                _entity("ground", 1, [_RenderBodyComponent([plane])]),
                _entity("fixture", 2, [_RenderBodyComponent([heavy])]),
            ]
        )
    )

    bodies = simulator.replay_bodies(0)

    assert [body.name for body in bodies] == ["fixture"]
    assert len(bodies[0].parts) == 1
    assert len(bodies[0].parts[0].mesh.vertices) == 8


def test_two_entities_with_one_name_are_told_apart_by_scene_id():
    shape = _shape(np.eye(3), scale=np.ones(3))
    simulator = _attached(
        _env(
            [
                _entity("link", 1, [_RenderBodyComponent([shape])], position=(1.0, 0.0, 0.0)),
                _entity("link", 2, [_RenderBodyComponent([shape])], position=(2.0, 0.0, 0.0)),
            ]
        )
    )
    simulator.replay_bodies(0)

    poses = simulator.replay_poses(0)

    assert sorted(poses) == ["link", "link#2"]
    assert poses["link#2"].position.tolist() == [2.0, 0.0, 0.0]
    # SAPIEN orders the quaternion wxyz, and the replay orders it xyzw.
    assert poses["link"].orientation.tolist() == pytest.approx([0.0, 0.6, 0.0, 0.8])


def test_a_camera_pose_is_in_opencv_axes():
    model = np.eye(4)
    model[:3, 3] = [0.0, 0.5, 1.5]
    camera = _RenderCameraComponent("head_camera", model)
    simulator = _attached(_env([_entity("head_camera", 1, [camera])]))
    simulator.replay_bodies(0)

    (pose,) = simulator.replay_extrinsics(0).values()

    assert pose.position.tolist() == [0.0, 0.5, 1.5]
    # OpenGL looks down -z with y up; OpenCV looks down +z with y down: a half turn about x.
    assert np.abs(pose.orientation).tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_the_readings_are_the_arm_and_gripper_joints_of_the_robot():
    simulator = _attached(_env([]))
    simulator.replay_bodies(0)

    scalars = {scalar.name: scalar.value for scalar in simulator.replay_scalars(0, None)}

    assert [scalars["joints/0"], scalars["joints/1"]] == [0.1, 0.2]
    assert scalars["qpos_gripper/0"] == 0.04
    assert scalars["joint_torques/1"] == 2.0
    assert "joints/2" not in scalars


def test_a_joint_name_with_the_scene_and_robot_in_front_matches_its_config_name():
    simulator = _attached(_env([], joint_prefix="scene-0-panda_"))
    simulator.replay_bodies(0)

    scalars = {scalar.name: scalar.value for scalar in simulator.replay_scalars(0, None)}

    assert [scalars["joints/0"], scalars["joints/1"]] == [0.1, 0.2]
    assert scalars["qpos_gripper/0"] == 0.04


def test_a_sapien_scene_holds_one_env():
    with pytest.raises(ValueError, match="one env"):
        SapienSimulator(_CONFIG).attach(_env([]), [0, 1])


def _visual_body(kind: str, vertices: np.ndarray, **size: Any) -> SimpleNamespace:
    shape = SimpleNamespace(
        mesh=SimpleNamespace(vertices=vertices, indices=np.array([0, 1, 2])),
        material=SimpleNamespace(base_color=[0.4, 0.5, 0.6, 1.0]),
    )
    return SimpleNamespace(type=kind, local_pose=_pose(), get_render_shapes=lambda: [shape], **size)


def _actor(name: str, actor_id: int, visual_bodies: list[Any], position=(0.0, 0.0, 0.0)) -> Any:
    return SimpleNamespace(
        get_name=lambda: name,
        get_id=lambda: actor_id,
        get_pose=lambda: _pose(position),
        get_visual_bodies=lambda: visual_bodies,
    )


def test_a_sapien_2_scene_gives_its_actors_and_links_with_each_shape_scaled_by_its_type():
    unit = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0], [1.0, -1.0, 1.0]])
    box = _actor("box", 1, [_visual_body("box", unit, half_lengths=np.array([0.1, 0.2, 0.3]))])
    ground = _actor("ground", 2, [_visual_body("plane", unit)])
    ball = _actor("ball", 3, [_visual_body("sphere", unit, radius=0.5)], position=(1.0, 0.0, 0.0))
    link = _actor("hand", 4, [_visual_body("mesh", unit, scale=np.array([2.0, 2.0, 2.0]))])
    camera = SimpleNamespace(get_name=lambda: "head_camera", get_model_matrix=lambda: np.eye(4))
    articulation = _articulation()
    articulation.get_links = lambda: [link]
    scene = SimpleNamespace(
        get_all_actors=lambda: [box, ground, ball],
        get_all_articulations=lambda: [articulation],
        get_cameras=lambda: [camera],
    )
    simulator = _attached(SimpleNamespace(task_env=SimpleNamespace(scene=scene)))

    bodies = {body.name: body for body in simulator.replay_bodies(0)}

    assert sorted(bodies) == ["ball", "box", "hand"]
    assert bodies["box"].parts[0].mesh.vertices[0].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert bodies["ball"].parts[0].mesh.vertices[0].tolist() == [0.5, 0.5, 0.5]
    assert bodies["hand"].parts[0].mesh.vertices[0].tolist() == [2.0, 2.0, 2.0]
    assert simulator.replay_poses(0)["ball"].position.tolist() == [1.0, 0.0, 0.0]
    assert list(simulator.replay_extrinsics(0)) == ["head"]
