from typing import Any

import numpy as np
import pytest

import manifold.recipes.local as local
from manifold.adapters.observation.camera_resolution import ResizeCameras
from manifold.adapters.observation.camera_rotate_180 import Rotate180Cameras
from manifold.adapters.observation.camera_vertical_flip import FlipVerticalCameras
from manifold.adapters.observation.proprio_rotation import ProprioRotationAdapter
from manifold.benchmarks.libero import CAMERA_RES, LIBERO
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
from manifold.recipes.local import _default_pool
from manifold.recipes.resolve import resolve
from manifold.sensors.cameras import agentview, wrist


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


def test_the_default_pool_bridges_openpi_to_libero() -> None:
    signature = _signature(
        agentview((224, 224, 3), orientation=CameraOrientation.FLIPPED_HORIZONTAL),
        wrist((224, 224, 3), orientation=CameraOrientation.FLIPPED_HORIZONTAL),
    )

    assert resolve(signature, LIBERO, _default_pool(signature, LIBERO)) is not None


def test_a_camera_that_already_matches_gets_no_orientation_adapter() -> None:
    signature = _signature(
        agentview(
            (CAMERA_RES, CAMERA_RES, 3),
            orientation=CameraOrientation.FLIPPED_VERTICAL,
        ),
        wrist((CAMERA_RES, CAMERA_RES, 3), orientation=CameraOrientation.UPRIGHT),
    )

    pool = _default_pool(signature, LIBERO)

    flipped = [
        adapter.cameras
        for adapter in pool
        if isinstance(adapter, FlipVerticalCameras | Rotate180Cameras)
    ]
    assert flipped == [("wrist",), ("wrist",)]
    assert not any(isinstance(adapter, ResizeCameras) for adapter in pool)


def test_a_camera_the_benchmark_does_not_publish_gets_no_adapter() -> None:
    signature = _signature(agentview((224, 224, 3)), wrist((224, 224, 3)))
    no_wrist = LIBERO.model_copy(
        update={"sensors": [c for c in LIBERO.sensors if c.name != "wrist"]}
    )

    pool = _default_pool(signature, no_wrist)

    for adapter in pool:
        if isinstance(adapter, FlipVerticalCameras | Rotate180Cameras):
            assert adapter.cameras == ("agentview",)
        if isinstance(adapter, ResizeCameras):
            assert set(adapter.targets) == {"agentview"}


def test_matching_ee_pose_formats_get_no_proprioception_rotation() -> None:
    signature = _signature(agentview((224, 224, 3)))
    same_rotation = signature.model_copy(
        update={"proprioception": LIBERO.embodiment.proprioception}
    )

    pool = _default_pool(same_rotation, LIBERO)

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

    assert isinstance(served["pipeline"](LIBERO), Pipeline)
