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
from manifold.embodiments import FRANKA_EE_DELTA
from manifold.recipes import from_lerobot_checkpoint


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


def test_a_camera_refuses_an_unlisted_name():
    with pytest.raises(ValidationError, match="name"):
        Camera(name=cast(Any, "head"), shape=(8, 8, 3))


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
    assert camera.name is CameraName.WRIST


def test_a_received_benchmark_drops_a_camera_this_sdk_does_not_list():
    benchmark = Benchmark.from_received(_advertised("head"))

    assert [camera.name for camera in benchmark.sensors] == [CameraName.AGENTVIEW]


def test_a_received_benchmark_keeps_every_listed_camera():
    benchmark = Benchmark.from_received(_advertised("wrist"))

    assert [camera.name for camera in benchmark.sensors] == [
        CameraName.AGENTVIEW,
        CameraName.WRIST,
    ]


def test_a_received_benchmark_refuses_an_unlisted_embodiment():
    payload = _advertised("wrist")
    payload["embodiment"]["name"] = "aloha_joint_absolute"

    with pytest.raises(ValidationError, match="name"):
        Benchmark.from_received(payload)


def test_a_policy_pairs_with_a_benchmark_that_also_publishes_an_unlisted_camera():
    # A peer on a newer SDK may publish a camera this SDK has no name for. A policy
    # that does not read it still pairs.
    benchmark = Benchmark.from_received(_advertised("head"))
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
