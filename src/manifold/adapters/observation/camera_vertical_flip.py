"""Flip named camera images top-to-bottom on the observation side.

Some checkpoints are trained on frames the benchmark publishes upside down:
RLDX-1-FT-ROBOCASA, for instance, is evaluated through robocasa's
`gymnasium_basic` env, whose `process_img` flips every camera vertically
(`img[::-1, :, :]`) before the model sees it — so the model expects right-side-up
frames, while a runner driving the raw robosuite env ships them bottom-up. This
adapter mirrors the named camera arrays top-to-bottom and — because camera
orientation is a declared convention — also advances the spec: it sets the
declared `orientation` of each named camera from UPRIGHT to FLIPPED_VERTICAL. So a
static `check_compatibility` can prove the flip is what bridges a policy that
consumes FLIPPED_VERTICAL frames against a benchmark that publishes UPRIGHT ones;
the flip is no longer a spec-invisible side effect. It is lossless — a
representation transform that reorients pixels.

This is a PURE vertical flip (the height axis only), distinct from
`Rotate180Cameras`, which reverses both height AND width (a 180-degree rotation).
Using the rotation where only a vertical mirror is wanted would add a spurious
left-right swap; the two are not interchangeable.

Everything else about the observation — the proprioception, the instruction, and
any cameras not named — passes through.

Parameterized by the camera names to flip: `FlipVerticalCameras(cameras=
("agentview_left", "agentview_right", "wrist"))`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel

from manifold.core.adapter import ObservationAdapter
from manifold.core.observation_space import ObservationSpace
from manifold.core.sensor import CameraOrientation
from manifold.core.values import Observation


class FlipVerticalCameras(ObservationAdapter):
    """Flip the named camera channels top-to-bottom and set their declared orientation."""

    from_spec: ClassVar[type[BaseModel]] = ObservationSpace
    to_spec: ClassVar[type[BaseModel]] = ObservationSpace
    lossless: ClassVar[bool] = True

    def __init__(self, cameras: Sequence[str]) -> None:
        self.cameras = tuple(cameras)

    def applies(self, source: BaseModel) -> bool:
        """True when any named camera is still UPRIGHT (so a flip is needed)."""
        if not isinstance(source, ObservationSpace):
            return False
        return any(
            (camera := source.camera(name)) is not None
            and camera.orientation is CameraOrientation.UPRIGHT
            for name in self.cameras
        )

    def produce(self, source: BaseModel) -> ObservationSpace:
        """The same spec with each named camera's orientation set to FLIPPED_VERTICAL."""
        if not isinstance(source, ObservationSpace):
            raise TypeError("FlipVerticalCameras transforms an ObservationSpace source only")
        out = source
        for name in self.cameras:
            camera = source.camera(name)
            if camera is not None:
                out = out.with_camera(
                    camera.model_copy(update={"orientation": CameraOrientation.FLIPPED_VERTICAL})
                )
        return out

    def adapt(self, observation: Any, *, source: BaseModel) -> Observation:
        """Flip each named camera top-to-bottom; other sensors and state pass through.

        A camera that is absent from `observation.sensors` is skipped, so the
        adapter is safe to point at a superset of the cameras a given observation
        carries. The flip reverses the height (row) axis of the HWC array only and
        keeps the result C-contiguous uint8.
        """
        sensors = dict(observation.sensors)
        for name in self.cameras:
            frame = sensors.get(name)
            if frame is None:
                continue
            flipped = np.asarray(frame)[::-1, :, :]  # vertical: reverse the height axis only
            sensors[name] = np.ascontiguousarray(flipped)  # preserve dtype (cf. SwapChannelOrder)
        return Observation(
            state=observation.state, sensors=sensors, instruction=observation.instruction
        )


__all__ = ["FlipVerticalCameras"]
