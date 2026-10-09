from typing import Any

import numpy as np
import pytest

import manifold.recipes.local as local
from manifold.adapters.observation.camera_resolution import ResizeCameras
from manifold.adapters.observation.camera_rotate_180 import Rotate180Cameras
from manifold.adapters.observation.camera_vertical_flip import FlipVerticalCameras
from manifold.adapters.observation.proprio_rotation import ProprioRotationAdapter
from manifold.core.conventions import GripperFormat, RotationFormat
from manifold.core.embodiment import (
    EEActionSpace,
    EEObservationSpec,
    GripperObservationSpec,
    Proprioception,
)
from manifold.core.pipeline import Pipeline
from manifold.core.policy import PolicySignature
from manifold.core.sensor import Camera, CameraOrientation
from manifold.core.values import Action, Observation
from manifold.recipes.function import StatefulEndpoint
from manifold.recipes.local import _default_pool
from manifold.recipes.resolve import resolve
from manifold.sensors import CameraName
from manifold.sensors.cameras import agentview, wrist
from tests._benchmarks import CAMERA_RES, TABLETOP


def _signature(*cameras: Camera) -> PolicySignature:
    return PolicySignature(
        cameras=list(cameras),
        proprioception=Proprioception(
            ee_pose=EEObservationSpec(
                rotation=RotationFormat.AXIS_ANGLE,
                gripper=GripperObservationSpec(dim=2),
            ),
        ),
        action_space=EEActionSpace(
            rotation=RotationFormat.AXIS_ANGLE,
            gripper=GripperFormat.SIGNED_OPEN_LOW,
            delta=True,
        ),
        instruction=True,
    )


def test_the_default_pool_bridges_openpi_to_the_tabletop() -> None:
    signature = _signature(
        agentview((224, 224, 3), orientation=CameraOrientation.FLIPPED_HORIZONTAL),
        wrist((224, 224, 3), orientation=CameraOrientation.FLIPPED_HORIZONTAL),
    )

    assert resolve(signature, TABLETOP, _default_pool(signature, TABLETOP)) is not None


def test_a_camera_that_already_matches_gets_no_orientation_adapter() -> None:
    signature = _signature(
        agentview(
            (CAMERA_RES, CAMERA_RES, 3),
            orientation=CameraOrientation.FLIPPED_VERTICAL,
        ),
        wrist((CAMERA_RES, CAMERA_RES, 3), orientation=CameraOrientation.UPRIGHT),
    )

    pool = _default_pool(signature, TABLETOP)

    flipped = [
        adapter.cameras
        for adapter in pool
        if isinstance(adapter, FlipVerticalCameras | Rotate180Cameras)
    ]
    assert flipped == [(CameraName.WRIST,), (CameraName.WRIST,)]
    assert not any(isinstance(adapter, ResizeCameras) for adapter in pool)


def test_a_camera_the_benchmark_does_not_publish_gets_no_adapter() -> None:
    signature = _signature(agentview((224, 224, 3)), wrist((224, 224, 3)))
    no_wrist = TABLETOP.model_copy(
        update={"sensors": [c for c in TABLETOP.sensors if c.name != CameraName.WRIST]}
    )

    pool = _default_pool(signature, no_wrist)

    for adapter in pool:
        if isinstance(adapter, FlipVerticalCameras | Rotate180Cameras):
            assert adapter.cameras == (CameraName.AGENTVIEW,)
        if isinstance(adapter, ResizeCameras):
            assert set(adapter.targets) == {CameraName.AGENTVIEW}


def test_matching_ee_pose_formats_get_no_proprioception_rotation() -> None:
    signature = _signature(agentview((224, 224, 3)))
    same_rotation = signature.model_copy(
        update={"proprioception": TABLETOP.embodiment.proprioception}
    )

    pool = _default_pool(same_rotation, TABLETOP)

    assert not any(isinstance(adapter, ProprioRotationAdapter) for adapter in pool)


def _predict(_observation: Observation) -> Action:
    return Action.from_array(np.zeros(7, dtype=np.float32))


def test_a_passed_pipeline_reaches_the_server_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served: dict[str, Any] = {}
    monkeypatch.setattr(local, "serve_endpoint", lambda endpoint, **kwargs: served.update(kwargs))
    chosen = Pipeline()
    signature = _signature(agentview((224, 224, 3)))

    local.serve(_predict, signature, pipeline=chosen, server="tcp:127.0.0.1:9")

    assert served["pipeline"] is chosen


def test_without_a_pipeline_the_server_resolves_one_per_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served: dict[str, Any] = {}
    monkeypatch.setattr(local, "serve_endpoint", lambda endpoint, **kwargs: served.update(kwargs))
    signature = _signature(
        agentview((224, 224, 3), orientation=CameraOrientation.FLIPPED_HORIZONTAL),
        wrist((224, 224, 3), orientation=CameraOrientation.FLIPPED_HORIZONTAL),
    )

    local.serve(_predict, signature, server="tcp:127.0.0.1:9")

    assert isinstance(served["pipeline"](TABLETOP), Pipeline)


def _chunk_predict(shape: tuple[int, ...]) -> Any:
    def predict(_observation: Observation) -> Action:
        return Action.from_array(np.zeros(shape, dtype=np.float32))

    return predict


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((8, 7), "returned 8 actions of 7 values"),
        ((10, 6), "returned 10 actions of 6 values"),
        ((7,), "returned a flat array of 7 values"),
    ],
)
def test_serve_rejects_a_predict_that_misses_the_declared_chunk_before_binding(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...], message: str
) -> None:
    served: list[object] = []
    monkeypatch.setattr(local, "serve_endpoint", lambda endpoint, **kwargs: served.append(endpoint))
    signature = _signature(agentview((224, 224, 3))).model_copy(
        update={"chunk_size": 10, "execution_steps": 5}
    )

    with pytest.raises(ValueError, match=message):
        local.serve(_chunk_predict(shape), signature, server="tcp:127.0.0.1:9")

    assert served == []


def test_serve_probes_predict_once_with_a_placeholder_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served: list[object] = []
    monkeypatch.setattr(local, "serve_endpoint", lambda endpoint, **kwargs: served.append(endpoint))
    seen: list[Observation] = []

    def predict(observation: Observation) -> Action:
        seen.append(observation)
        return _predict(observation)

    local.serve(predict, _signature(agentview((224, 224, 3))), server="tcp:127.0.0.1:9")

    assert len(seen) == 1
    assert seen[0].instruction == "probe"
    assert seen[0].sensors[CameraName.AGENTVIEW].shape == (224, 224, 3)
    assert len(served) == 1


class _Policy:
    """Stateful class where reset() is called at the start of every episode.

    Do not load weights in this class.
    """

    created = 0

    def __init__(self) -> None:
        type(self).created += 1
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def predict(self, observation: Observation) -> Action:
        assert self.resets == 1
        return _predict(observation)


def test_serve_probes_a_stateful_class_on_its_own_reset_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served: list[object] = []
    monkeypatch.setattr(local, "serve_endpoint", lambda endpoint, **kwargs: served.append(endpoint))
    monkeypatch.setattr(_Policy, "created", 0)

    local.serve(_Policy, _signature(agentview((224, 224, 3))), server="tcp:127.0.0.1:9")

    assert _Policy.created == 1
    assert isinstance(served[0], StatefulEndpoint)
    assert served[0].policy is _Policy


class _NoReset:
    def predict(self, observation: Observation) -> Action:
        return _predict(observation)


def test_serve_rejects_a_class_without_reset_before_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served: list[object] = []
    monkeypatch.setattr(local, "serve_endpoint", lambda endpoint, **kwargs: served.append(endpoint))

    with pytest.raises(TypeError, match=r"_NoReset must define reset\(\) and predict\(\)"):
        local.serve(_NoReset, _signature(agentview((224, 224, 3))), server="tcp:127.0.0.1:9")

    assert served == []
