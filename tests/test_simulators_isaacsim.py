"""Tests for `IsaacSimSimulator`, against stand-ins for `pxr`, `isaaclab` and an Isaac Lab env."""

from __future__ import annotations

import faulthandler
import os
import resource
import signal
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from manifold.benchmark.config import BenchmarkConfig
from manifold.benchmark.simulators import Simulator
from manifold.benchmark.simulators.isaacsim import (
    BenchmarkIsaacSimSimulatorConfig,
    IsaacSimSimulator,
)
from manifold.embodiments import DROID_JOINT_ABSOLUTE
from manifold.replay import RerunTab

_CONFIG = BenchmarkIsaacSimSimulatorConfig(
    name="isaacsim",
    path="inner",
    app={"headless": True, "enable_cameras": True},
    robot="robot",
    max_position_iterations=None,
    env_spacing=None,
    camera_far_clip=None,
)


@pytest.fixture(autouse=True)
def _faulthandler_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep `start` and `attach` from changing the test process's SIGSEGV handler."""
    _record_faulthandler(monkeypatch)


# Each env's origin on Isaac Lab's grid.
_ORIGINS = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])


class _Attr:
    def __init__(self, value: Any) -> None:
        self._value = value

    def Get(self) -> Any:  # noqa: N802
        return self._value


class _Prim:
    """A USD prim: its path, its schema kinds, its applied APIs and its attributes."""

    def __init__(
        self,
        path: str,
        kinds: tuple[str, ...] = (),
        *,
        attrs: dict[str, Any] | None = None,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rigid: bool = False,
        material: tuple[float, float, float] | None = None,
        subsets: tuple[_Prim, ...] = (),
    ) -> None:
        self.path = path
        self.kinds = kinds
        self.attrs = attrs or {}
        self.matrix = np.eye(4)
        self.matrix[3, :3] = translate
        self.rigid = rigid
        self.material = material
        self.subsets = subsets
        self.parent: _Prim | None = None
        self.children: list[_Prim] = []

    def add(self, child: _Prim) -> _Prim:
        child.parent = self
        child.matrix = child.matrix @ self.matrix
        self.children.append(child)
        return child

    def IsA(self, kind: Any) -> bool:  # noqa: N802
        return kind.kind in self.kinds

    def HasAPI(self, api: Any) -> bool:  # noqa: N802
        return self.rigid

    def IsValid(self) -> bool:  # noqa: N802
        return True

    def GetParent(self) -> _Prim | None:  # noqa: N802
        return self.parent

    def GetPath(self) -> str:  # noqa: N802
        return self.path

    def GetName(self) -> str:  # noqa: N802
        return self.path.rsplit("/", 1)[-1]

    def GetTypeName(self) -> str:  # noqa: N802
        return self.kinds[-1] if self.kinds else "Xform"

    def GetPrim(self) -> _Prim:  # noqa: N802
        return self

    def GetIndicesAttr(self) -> _Attr:  # noqa: N802
        return _Attr(self.attrs.get("indices"))

    def walk(self) -> list[_Prim]:
        return [self, *(prim for child in self.children for prim in child.walk())]


class _Schema:
    kind = ""

    def __init__(self, prim: _Prim) -> None:
        self._prim = prim

    def __getattr__(self, name: str) -> Any:
        key = name.removeprefix("Get").removesuffix("Attr")
        return lambda: _Attr(self._prim.attrs.get(key))

    def ComputePurpose(self) -> str:  # noqa: N802
        return self._prim.attrs.get("purpose", "default")

    def ComputeVisibility(self) -> str:  # noqa: N802
        return self._prim.attrs.get("visibility", "inherited")


def _schema(kind: str) -> type:
    return type(kind, (_Schema,), {"kind": kind})


class _Material:
    def __init__(self, colour: tuple[float, float, float]) -> None:
        self._colour = colour

    def ComputeSurfaceSource(self, context: str) -> tuple[Any, str, str]:  # noqa: N802
        shader = SimpleNamespace(
            GetInput=lambda name: _Attr(self._colour) if name == "diffuseColor" else None
        )
        return shader, "surface", "out"


def _pxr() -> ModuleType:
    """A stand-in for the parts of `pxr` that the simulator reads."""
    module: Any = ModuleType("pxr")
    module.Usd = SimpleNamespace(
        PrimRange=lambda root, _predicate: root.walk(),
        TraverseInstanceProxies=lambda: None,
        TimeCode=SimpleNamespace(Default=lambda: 0),
    )
    module.UsdGeom = SimpleNamespace(
        Gprim=_schema("Gprim"),
        Mesh=_schema("Mesh"),
        Cube=_schema("Cube"),
        Cylinder=_schema("Cylinder"),
        Imageable=_schema("Imageable"),
        Subset=SimpleNamespace(GetAllGeomSubsets=lambda imageable: imageable._prim.subsets),
        XformCache=lambda _time: SimpleNamespace(GetLocalToWorldTransform=lambda p: p.matrix),
        GetStageMetersPerUnit=lambda _stage: 1.0,
    )
    module.UsdPhysics = SimpleNamespace(RigidBodyAPI=object())
    module.UsdShade = SimpleNamespace(
        MaterialBindingAPI=lambda prim: SimpleNamespace(
            ComputeBoundMaterial=lambda: (
                _Material(prim.material) if prim.material is not None else None,
                None,
            )
        )
    )
    return module


def _cube(path: str, **kwargs: Any) -> _Prim:
    return _Prim(path, ("Gprim", "Cube"), attrs={"Size": 0.2, **kwargs.pop("attrs", {})}, **kwargs)


def _square(path: str, subsets: tuple[_Prim, ...]) -> _Prim:
    """A unit square mesh of two triangles."""
    return _Prim(
        path,
        ("Gprim", "Mesh"),
        attrs={
            "Points": [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
            "FaceVertexCounts": [3, 3],
            "FaceVertexIndices": [0, 1, 2, 0, 2, 3],
        },
        subsets=subsets,
    )


def _stage(env_index: int) -> _Prim:
    """Env `env_index`'s prim: a robot with two links, a banana, a bowl and a table."""
    root = _Prim(f"/World/envs/env_{env_index}", translate=tuple(_ORIGINS[env_index]))
    robot = root.add(_Prim(f"{root.path}/robot"))
    base = robot.add(_Prim(f"{robot.path}/base_link", rigid=True))
    base.add(_cube(f"{base.path}/visuals", attrs={"DisplayColor": [(1.0, 0.0, 0.0)]}))
    base.add(_cube(f"{base.path}/collisions/box"))
    finger = robot.add(_Prim(f"{robot.path}/finger", rigid=True, translate=(0.0, 0.0, 0.5)))
    blue = _Prim("subset", attrs={"indices": [1]}, material=(0.0, 0.0, 1.0))
    finger.add(_square(f"{finger.path}/mesh", (blue,)))
    banana = root.add(_Prim(f"{root.path}/banana", rigid=True, translate=(0.3, 0.0, 0.0)))
    banana.add(_cube(f"{banana.path}/visuals", material=(1.0, 1.0, 0.0)))
    bowl = root.add(_Prim(f"{root.path}/bowl", rigid=True, translate=(0.0, 0.3, 0.0)))
    bowl.add(_square(f"{bowl.path}/visuals", ()))
    table = root.add(_Prim(f"{root.path}/table", translate=(0.0, 0.0, -0.1)))
    table.add(_cube(f"{table.path}/top"))
    table.add(_cube(f"{table.path}/guide", attrs={"purpose": "guide"}))
    table.add(_Prim(f"{table.path}/ball", ("Gprim", "Sphere")))
    return root


class _Scene:
    """A stand-in for Isaac Lab's `InteractiveScene` of two envs."""

    def __init__(self, sensors: dict[str, Any]) -> None:
        roots = {f"/World/envs/env_{index}": _stage(index) for index in range(2)}
        self.stage = SimpleNamespace(GetPrimAtPath=roots.__getitem__)
        self.env_prim_paths = list(roots)
        self.env_origins = _ORIGINS
        joint_pos = np.array([[0.1 * i for i in range(9)], [0.2 * i for i in range(9)]])
        link_pos = _ORIGINS[:, None, :] + np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])
        self.robot = SimpleNamespace(
            cfg=SimpleNamespace(prim_path="/World/envs/env_.*/robot"),
            body_names=["base_link", "finger"],
            joint_names=[f"j{i}" for i in range(9)],
            data=SimpleNamespace(
                joint_pos=joint_pos,
                joint_vel=joint_pos * 2,
                joint_acc=joint_pos * 3,
                applied_torque=joint_pos * 4,
                computed_torque=joint_pos * 5,
                root_link_pos_w=_ORIGINS.copy(),
                root_link_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
                body_link_pos_w=link_pos,
                body_link_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (2, 2, 1)),
                body_incoming_joint_wrench_b=np.ones((2, 2, 6)),
            ),
        )
        self.rigid_objects = {
            "banana": self._object("banana", [0.3, 0.0, 0.0]),
            "bowl": self._object("bowl", [0.0, 0.3, 0.0]),
        }
        self.articulations = {"robot": self.robot}
        self.sensors = sensors

    def _object(self, name: str, position: list[float]) -> SimpleNamespace:
        return SimpleNamespace(
            cfg=SimpleNamespace(prim_path=f"/World/envs/env_.*/{name}"),
            data=SimpleNamespace(
                root_link_pos_w=_ORIGINS + position,
                root_link_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
                root_com_lin_vel_w=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]),
            ),
        )

    def __getitem__(self, name: str) -> Any:
        return {"robot": self.robot, **self.rigid_objects, **self.sensors}[name]


def _contact(prim_path: str, filters: list[str], matrix: np.ndarray | None) -> SimpleNamespace:
    return SimpleNamespace(
        cfg=SimpleNamespace(prim_path=prim_path, filter_prim_paths_expr=filters),
        data=SimpleNamespace(force_matrix_w=matrix, net_forces_w=np.zeros((2, 1, 3))),
    )


def _camera(frame: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        cfg=SimpleNamespace(prim_path="/World/envs/env_.*/cam", data_types=["rgb"]),
        data=SimpleNamespace(output={"rgb": np.stack([frame, frame + 1])}),
    )


def _sensors() -> dict[str, Any]:
    """RoboLab's sensors: two policy cameras, an egocentric one, and three contact sensors."""
    finger = "/World/envs/env_.*/robot/finger"
    batch = np.zeros((2, 1, 2, 3))
    batch[1, 0, 0] = [0.0, 3.0, 4.0]
    return {
        "over_shoulder_left_camera": _camera(np.zeros((4, 6, 3), dtype=np.uint8)),
        "wrist_cam": _camera(np.zeros((4, 6, 3), dtype=np.uint8)),
        "egocentric_mirrored_camera": _camera(
            np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        ),
        "gripper__all_objs": _contact(
            finger, ["/World/envs/env_.*/banana", "/World/envs/env_.*/bowl"], batch
        ),
        "gripper__banana": _contact(finger, ["/World/envs/env_.*/banana"], np.zeros((2, 1, 1, 3))),
        "banana__bowl": _contact(
            "/World/envs/env_.*/banana", ["/World/envs/env_.*/bowl"], np.zeros((2, 1, 1, 3))
        ),
    }


def _benchmark(spec_cameras: list[str]) -> BenchmarkConfig:
    """A benchmark config on DROID's 7-joint arm whose spec has the cameras `spec_cameras`."""
    return BenchmarkConfig.model_validate(
        {
            "benchmark": {"version": "0.0.1", "rubric": "rubric.json"},
            "upstream": {
                "repo": "https://example.com/robolab",
                "commit": "abc",
                "patches": [],
                "path": "/opt/upstream",
                "data": "/data/robolab",
            },
            "service": {"type": "batched", "reset_together": True},
            "extras": {},
            "env": {"num_envs": 2},
            "run": {"seed": 0, "max_steps": 5, "step_seconds": 0.05, "image_format": "jpeg"},
            "simulator": _CONFIG.model_dump(mode="json"),
            "replay": {"cameras": [spec_cameras[0]]},
            "spec": {
                "name": "robolab",
                "embodiment": DROID_JOINT_ABSOLUTE.model_dump(mode="json"),
                "sensors": [{"name": name, "shape": [4, 6, 3]} for name in spec_cameras],
                "instruction": True,
            },
        }
    )


@pytest.fixture
def isaac(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Stand-ins for `pxr` and for `isaaclab` 0.45, which orders quaternions `wxyz`."""
    monkeypatch.setitem(sys.modules, "pxr", _pxr())
    package: Any = ModuleType("isaaclab")
    package.__version__ = "0.45.9"
    monkeypatch.setitem(sys.modules, "isaaclab", package)
    return SimpleNamespace(package=package)


def _attached(
    sensors: dict[str, Any], spec_cameras: list[str], env_ids: list[int]
) -> tuple[IsaacSimSimulator, _Scene]:
    scene = _Scene(sensors)
    terms = {"success": np.array([0.0, 1.0]), "time_out": np.array([0.0, 0.0])}
    unwrapped = SimpleNamespace(
        scene=scene,
        termination_manager=SimpleNamespace(active_terms=list(terms), get_term=terms.__getitem__),
    )
    env = SimpleNamespace(inner=SimpleNamespace(unwrapped=unwrapped))
    simulator = IsaacSimSimulator(_CONFIG, _benchmark(spec_cameras))
    simulator.attach(env, env_ids)
    for env_id in env_ids:
        simulator.replay_bodies(env_id)
    return simulator, scene


def test_start_raises_the_file_limit_and_launches_isaac_sim(
    isaac: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_to: list[tuple[int, int]] = []
    monkeypatch.setattr(resource, "getrlimit", lambda _kind: (1024, 8192))
    monkeypatch.setattr(resource, "setrlimit", lambda _kind, limits: set_to.append(limits))
    launched: list[dict[str, Any]] = []

    def launcher(**kwargs: Any) -> SimpleNamespace:
        launched.append(kwargs)
        return SimpleNamespace(app=SimpleNamespace())

    monkeypatch.setitem(sys.modules, "isaaclab.app", SimpleNamespace(AppLauncher=launcher))
    _record_faulthandler(monkeypatch)
    IsaacSimSimulator(_CONFIG, _benchmark(["over_shoulder_left"])).start()
    assert set_to == [(8192, 8192), (0, 0)]
    assert launched == [{"headless": True, "enable_cameras": True}]


def _record_faulthandler(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace `faulthandler`'s calls with stand-ins that record each call."""
    calls: list[str] = []
    monkeypatch.setattr(faulthandler, "enable", lambda **_: calls.append("enable"))
    monkeypatch.setattr(
        faulthandler, "register", lambda signum, **_: calls.append(f"register {signum}")
    )
    return calls


def test_start_gives_sigsegv_to_faulthandler_after_isaac_lab_takes_it(
    isaac: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_faulthandler(monkeypatch)
    monkeypatch.setattr(resource, "setrlimit", lambda _kind, _limits: None)
    monkeypatch.setitem(
        sys.modules,
        "isaaclab.app",
        SimpleNamespace(
            AppLauncher=lambda **_: calls.append("launch") or SimpleNamespace(app=None)
        ),
    )
    IsaacSimSimulator(_CONFIG, _benchmark(["over_shoulder_left"])).start()
    assert calls == ["launch", "enable", f"register {signal.SIGUSR1}"]


def test_attach_gives_sigsegv_to_faulthandler_again(
    isaac: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_faulthandler(monkeypatch)
    _attached({}, ["over_shoulder_left"], [0])
    assert calls == ["enable"]


def _env_cfg(cameras: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        sim=SimpleNamespace(physx=SimpleNamespace(max_position_iteration_count=255)),
        scene=SimpleNamespace(env_spacing=10.0, **cameras),
    )


def _settings(**values: Any) -> BenchmarkIsaacSimSimulatorConfig:
    return _CONFIG.model_copy(update=values)


def test_the_settings_apply_to_the_env_config(monkeypatch: pytest.MonkeyPatch) -> None:
    class CameraCfg:
        def __init__(self, clipping_range: Any) -> None:
            self.prim_path = "/World/envs/env_.*/camera"
            self.spawn = SimpleNamespace(clipping_range=clipping_range)

    monkeypatch.setitem(
        sys.modules,
        "isaaclab.sensors",
        SimpleNamespace(CameraCfg=CameraCfg, TiledCameraCfg=CameraCfg),
    )
    wrist, shoulder = CameraCfg((0.05, 1e6)), CameraCfg(None)
    env_cfg = _env_cfg({"wrist_cam": wrist, "shoulder_cam": shoulder})

    IsaacSimSimulator.apply_settings(
        env_cfg, _settings(max_position_iterations=32, env_spacing=4.0, camera_far_clip=20.0)
    )

    assert env_cfg.sim.physx.max_position_iteration_count == 32
    assert env_cfg.scene.env_spacing == 4.0
    assert wrist.spawn.clipping_range == (0.05, 20.0)
    assert shoulder.spawn.clipping_range == (0.01, 20.0)


def test_a_null_setting_keeps_the_envs_value() -> None:
    env_cfg = _env_cfg({})

    IsaacSimSimulator.apply_settings(env_cfg, _CONFIG)

    assert env_cfg.sim.physx.max_position_iteration_count == 255
    assert env_cfg.scene.env_spacing == 10.0


def test_a_setting_without_its_target_raises() -> None:
    env_cfg = SimpleNamespace(sim=SimpleNamespace(), scene=SimpleNamespace())

    with pytest.raises(ValueError, match="max_position_iteration_count"):
        IsaacSimSimulator.apply_settings(env_cfg, _settings(max_position_iterations=32))


class _ProcessEnded(Exception):
    """Raised by the stand-in for `os._exit`, with the exit code."""


def _stop(simulator: IsaacSimSimulator, monkeypatch: pytest.MonkeyPatch) -> int:
    def end_process(code: int) -> None:
        raise _ProcessEnded(code)

    monkeypatch.setattr(os, "_exit", end_process)
    with pytest.raises(_ProcessEnded) as ended:
        simulator.stop()
    return ended.value.args[0]


def test_stop_ends_the_process_with_0_after_a_run_without_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulator = IsaacSimSimulator(_CONFIG, _benchmark(["over_shoulder_left"]))

    assert _stop(simulator, monkeypatch) == 0


def test_stop_prints_the_error_on_its_way_out_and_ends_the_process_with_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    simulator = IsaacSimSimulator(_CONFIG, _benchmark(["over_shoulder_left"]))
    code = None
    try:
        try:
            raise RuntimeError("the env failed to step")
        finally:
            code = _stop(simulator, monkeypatch)
    except RuntimeError:
        pass

    assert code == 1
    assert "the env failed to step" in capsys.readouterr().err


def test_start_keeps_a_higher_file_limit(
    isaac: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_to: list[tuple[int, int]] = []
    monkeypatch.setattr(resource, "getrlimit", lambda _kind: (65536, 65536))
    monkeypatch.setattr(resource, "setrlimit", lambda _kind, limits: set_to.append(limits))
    monkeypatch.setitem(
        sys.modules,
        "isaaclab.app",
        SimpleNamespace(AppLauncher=lambda **_: SimpleNamespace(app=None)),
    )
    _record_faulthandler(monkeypatch)
    IsaacSimSimulator(_CONFIG, _benchmark(["over_shoulder_left"])).start()
    assert set_to == [(0, 0)]


def test_the_scene_groups_drawn_prims_under_their_rigid_bodies(isaac: SimpleNamespace) -> None:
    simulator, _ = _attached({}, ["over_shoulder_left"], [0, 1])
    bodies = {body.name: body for body in simulator.replay_bodies(1)}
    assert list(bodies) == ["base_link", "finger", "banana", "bowl", "top"]
    assert isinstance(simulator, Simulator)
    # The table has no rigid body, so its cube is a body named for its own prim. The
    # table's guide cube and sphere and the base's collision box are left out.
    assert len(bodies["top"].parts) == 1
    assert len(bodies["base_link"].parts) == 1


def test_a_parts_colour_comes_from_its_material_then_its_display_colour(
    isaac: SimpleNamespace,
) -> None:
    simulator, _ = _attached({}, ["over_shoulder_left"], [0])
    bodies = {body.name: body for body in simulator.replay_bodies(0)}
    assert bodies["banana"].parts[0].color == (1.0, 1.0, 0.0, 1.0)
    assert bodies["base_link"].parts[0].color == (1.0, 0.0, 0.0, 1.0)
    assert bodies["top"].parts[0].color == (0.7, 0.7, 0.7, 1.0)
    finger = bodies["finger"].parts
    assert [part.color for part in finger] == [(0.0, 0.0, 1.0, 1.0), (0.7, 0.7, 0.7, 1.0)]
    assert [len(part.mesh.faces) for part in finger] == [1, 1]


def test_the_vertices_are_in_their_bodys_frame(isaac: SimpleNamespace) -> None:
    simulator, _ = _attached({}, ["over_shoulder_left"], [0, 1])
    bodies = {body.name: body for body in simulator.replay_bodies(1)}
    vertices = np.asarray(bodies["finger"].parts[1].mesh.vertices)
    np.testing.assert_allclose(vertices.min(axis=0), [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(vertices.max(axis=0), [1.0, 1.0, 0.0], atol=1e-6)


def test_a_heavy_fixed_body_is_drawn_as_its_box_but_a_rigid_object_is_not(
    isaac: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("manifold.benchmark.simulators.isaacsim._BOX_VERTICES", 3)
    simulator, _ = _attached({}, ["over_shoulder_left"], [0])
    bodies = {body.name: body for body in simulator.replay_bodies(0)}
    assert [len(part.mesh.vertices) for part in bodies["finger"].parts] == [8]
    np.testing.assert_allclose(bodies["finger"].parts[0].position, [0.5, 0.5, 0.0])
    assert [len(part.mesh.vertices) for part in bodies["bowl"].parts] == [4]


def test_the_poses_are_in_the_envs_frame_and_xyzw(isaac: SimpleNamespace) -> None:
    simulator, _ = _attached({}, ["over_shoulder_left"], [0, 1])
    poses = simulator.replay_poses(1)
    np.testing.assert_allclose(poses["finger"].position, [0.0, 0.0, 0.5])
    np.testing.assert_allclose(poses["banana"].position, [0.3, 0.0, 0.0])
    np.testing.assert_allclose(poses["top"].position, [0.0, 0.0, -0.1], atol=1e-6)
    np.testing.assert_allclose(poses["finger"].orientation, [0.0, 0.0, 0.0, 1.0])


def test_an_isaac_lab_from_version_1_is_read_as_xyzw(isaac: SimpleNamespace) -> None:
    isaac.package.__version__ = "6.1.17"
    simulator, scene = _attached({}, ["over_shoulder_left"], [0])
    scene.rigid_objects["banana"].data.root_link_quat_w = np.tile([0.0, 0.0, 1.0, 0.0], (2, 1))
    np.testing.assert_allclose(
        simulator.replay_poses(0)["banana"].orientation, [0.0, 0.0, 1.0, 0.0]
    )


def test_a_single_env_is_read_from_its_first_row(isaac: SimpleNamespace) -> None:
    simulator, _ = _attached({}, ["over_shoulder_left"], [5])
    np.testing.assert_allclose(simulator.replay_poses(5)["banana"].position, [0.3, 0.0, 0.0])


def test_the_readings_are_the_robots_the_tasks_and_the_physics(isaac: SimpleNamespace) -> None:
    simulator, _ = _attached(_sensors(), ["over_shoulder_left", "wrist"], [0, 1])
    scalars = simulator.replay_scalars(1, np.array([0.3, 0.0, 0.05]))
    values = {scalar.name: scalar.value for scalar in scalars}
    groups = {scalar.name.split("/")[0]: scalar.tab for scalar in scalars}
    assert len(values) == len(scalars)
    assert [name for name in values if name.startswith("joints/")] == [
        f"joints/{i}" for i in range(7)
    ]
    assert values["joints/6"] == pytest.approx(1.2)
    assert list(name for name in values if name.startswith("gripper_qvel/")) == [
        "gripper_qvel/0",
        "gripper_qvel/1",
    ]
    assert values["gripper_torques/1"] == pytest.approx(0.2 * 8 * 4)
    assert values["base_quat/3"] == 1.0
    assert values["terms/success"] == 1.0
    assert values["object_dist/banana"] == pytest.approx(0.05)
    assert values["contacts/gripper__all_objs/banana"] == pytest.approx(5.0)
    assert values["contacts/finger"] == pytest.approx(np.sqrt(6))
    assert values["obj_vel/bowl/2"] == 2.0
    assert values["gripper_near/banana"] == 1.0
    assert values["gripper_near/bowl"] == 0.0
    assert values["obj_near/banana__bowl"] == 0.0
    assert not any(
        name.startswith(("contacts/gripper__banana", "contacts/banana__bowl")) for name in values
    )
    assert groups == {
        **dict.fromkeys(
            [
                "joints",
                "joints_vel",
                "joints_acc",
                "joint_pos_cos",
                "joint_pos_sin",
                "gripper_qvel",
                "base_pos",
                "base_quat",
            ],
            RerunTab.ROBOT,
        ),
        **dict.fromkeys(["terms", "object_dist"], RerunTab.TASK),
        **dict.fromkeys(
            [
                "joint_torques",
                "gripper_torques",
                "computed_torques",
                "qpos_robot",
                "qvel_robot",
                "contacts",
                "obj_pos",
                "obj_vel",
                "gripper_near",
                "obj_near",
            ],
            RerunTab.PHYSICS,
        ),
    }


def test_the_reading_names_stay_the_same_when_isaac_lab_reports_nothing(
    isaac: SimpleNamespace,
) -> None:
    simulator, scene = _attached(_sensors(), ["over_shoulder_left", "wrist"], [0, 1])
    before = [scalar.name for scalar in simulator.replay_scalars(0, None)]
    scene.robot.data = SimpleNamespace()
    scene.sensors["gripper__all_objs"].data.force_matrix_w = None
    assert [scalar.name for scalar in simulator.replay_scalars(0, None)] == before


def test_without_a_contact_sensor_the_gripper_is_near_an_object_within_8_cm(
    isaac: SimpleNamespace,
) -> None:
    simulator, _ = _attached({}, ["over_shoulder_left"], [0])
    values = {r.name: r.value for r in simulator.replay_scalars(0, np.array([0.0, 0.25, 0.0]))}
    assert values["gripper_near/bowl"] == 1.0
    assert values["gripper_near/banana"] == 0.0


def test_the_images_are_the_cameras_that_produce_no_spec_camera_halved(
    isaac: SimpleNamespace,
) -> None:
    simulator, _ = _attached(_sensors(), ["over_shoulder_left", "wrist"], [0, 1])
    images = simulator.replay_images(1)
    assert list(images) == ["egocentric_mirrored_camera"]
    frame = np.arange(4 * 6 * 3).reshape(4, 6, 3) + 1
    expected = frame.reshape(2, 2, 3, 2, 3).mean(axis=(1, 3)).astype(np.uint8)
    np.testing.assert_array_equal(images["egocentric_mirrored_camera"], expected)


def test_a_spec_camera_that_matches_no_sensor_name_takes_the_next_unmatched_sensor(
    isaac: SimpleNamespace,
) -> None:
    sensors = {
        "external_camera": _camera(np.zeros((4, 6, 3), dtype=np.uint8)),
        "wrist_camera": _camera(np.zeros((4, 6, 3), dtype=np.uint8)),
    }
    simulator, _ = _attached(sensors, ["agentview", "wrist"], [0])
    assert simulator.replay_images(0) == {}


def test_no_camera_pose_is_read(isaac: SimpleNamespace) -> None:
    simulator, _ = _attached(_sensors(), ["over_shoulder_left", "wrist"], [0])
    assert simulator.replay_extrinsics(0) == {}
