import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from manifold.core import (
    Benchmark,
    Camera,
    CameraName,
    Compatibility,
    Embodiment,
    PolicySignature,
    check_compatibility,
)
from manifold.core.sensor import Modality, Mount
from manifold.embodiments import FRANKA_EE_DELTA
from manifold.recipes import from_lerobot_checkpoint
from manifold.sensors import (
    agentview,
    agentview_left,
    agentview_right,
    head,
    over_shoulder_left,
    over_shoulder_right,
    wrist,
)


def _advertised(camera: str) -> dict[str, Any]:
    """A hello's benchmark payload publishing the agentview and one more camera."""
    benchmark = Benchmark(
        name="suite",
        embodiment=FRANKA_EE_DELTA,
        sensors=[Camera(name=CameraName.AGENTVIEW, shape=(8, 8, 3))],
        instruction=False,
    ).model_dump(mode="json")
    benchmark["sensors"].append({**benchmark["sensors"][0], "name": camera})
    return benchmark


def test_a_camera_takes_a_name_that_the_sdk_does_not_list():
    assert Camera(name="egocentric", shape=(8, 8, 3)).key == "egocentric"


def test_an_embodiment_refuses_an_unlisted_name():
    with pytest.raises(ValidationError, match="name"):
        Embodiment(
            name=cast(Any, "aloha_joint_absolute"),
            action=FRANKA_EE_DELTA.action,
            proprioception=FRANKA_EE_DELTA.proprioception,
        )


def test_a_listed_name_given_as_its_string_becomes_the_member():
    # A spec read back from JSON carries plain strings, and must compare equal to the
    # one built in code.
    camera = Camera.model_validate({"name": "wrist", "shape": (8, 8, 3)})
    assert camera == Camera(name=CameraName.WRIST, shape=(8, 8, 3))


def test_a_benchmark_refuses_an_unlisted_embodiment():
    payload = _advertised("wrist")
    payload["embodiment"]["name"] = "aloha_joint_absolute"

    with pytest.raises(ValidationError, match="name"):
        Benchmark.model_validate(payload)


def test_a_policy_pairs_with_a_benchmark_that_also_publishes_an_unlisted_camera():
    benchmark = Benchmark.model_validate(_advertised("egocentric"))
    policy = PolicySignature(
        action_space=FRANKA_EE_DELTA.action,
        proprioception=FRANKA_EE_DELTA.proprioception,
        cameras=[Camera(name=CameraName.AGENTVIEW, shape=(8, 8, 3))],
        instruction=False,
    )

    assert check_compatibility(policy, benchmark).status is Compatibility.COMPATIBLE


def test_a_lerobot_draft_leaves_out_a_camera_with_an_unlisted_name(tmp_path: Path):
    image = {"type": "VISUAL", "shape": [3, 224, 224]}
    config = {
        "input_features": {"observation.images.wrist": image, "observation.images.top": image},
        "output_features": {"action": {"type": "ACTION", "shape": [7]}},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    draft = cast(PolicySignature, from_lerobot_checkpoint(tmp_path).draft_policy_spec())

    assert [camera.name for camera in draft.cameras] == [CameraName.WRIST]


_CONSTRUCTORS = [
    (agentview, CameraName.AGENTVIEW, Mount.SCENE),
    (agentview_left, CameraName.AGENTVIEW_LEFT, Mount.SCENE),
    (agentview_right, CameraName.AGENTVIEW_RIGHT, Mount.SCENE),
    (head, CameraName.HEAD, Mount.SCENE),
    (over_shoulder_left, CameraName.OVER_SHOULDER_LEFT, Mount.SCENE),
    (over_shoulder_right, CameraName.OVER_SHOULDER_RIGHT, Mount.SCENE),
    (wrist, CameraName.WRIST, Mount.WRIST),
]


@pytest.mark.parametrize(("constructor", "name", "mount"), _CONSTRUCTORS)
def test_a_camera_constructor_builds_colour_by_default(constructor, name, mount):
    camera = constructor((8, 8, 3))

    assert (camera.name, camera.key, camera.mount, camera.modality, camera.dtype) == (
        name,
        name,
        mount,
        Modality.RGB,
        "uint8",
    )


@pytest.mark.parametrize(("constructor", "name", "mount"), _CONSTRUCTORS)
def test_a_camera_constructor_builds_metric_depth(constructor, name, mount):
    camera = constructor((8, 8, 1), modality=Modality.DEPTH)

    assert (camera.name, camera.key, camera.mount, camera.modality, camera.dtype) == (
        name,
        f"{name}_depth",
        mount,
        Modality.DEPTH,
        "float32",
    )
