"""Rotate named camera images 180 degrees on the observation side.

Some checkpoints train on a different orientation than the benchmark publishes (the
RLDX LIBERO checkpoint expects robosuite's frames rotated 180°). Because orientation is
a declared convention, this adapter both rotates the named arrays AND advances each
camera's `orientation` UPRIGHT -> ROTATED_180, so `check_compatibility` can prove the
flip is what bridges the pairing rather than it being a spec-invisible side effect.
Lossless (a pixel reorientation).
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


class Rotate180Cameras(ObservationAdapter):
    """Rotate the named camera channels 180 degrees and flip their declared orientation."""

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
        """The same spec with each named camera's orientation flipped to ROTATED_180."""
        if not isinstance(source, ObservationSpace):
            raise TypeError("Rotate180Cameras transforms an ObservationSpace source only")
        out = source
        for name in self.cameras:
            camera = source.camera(name)
            if camera is not None:
                out = out.with_camera(
                    camera.model_copy(update={"orientation": CameraOrientation.ROTATED_180})
                )
        return out

    def adapt(self, observation: Any, *, source: BaseModel) -> Observation:
        """Rotate each named camera 180°; other sensors and state pass through.

        A camera absent from `observation.sensors` is skipped, so the adapter is safe to
        point at a superset of the cameras an observation carries.
        """
        sensors = dict(observation.sensors)
        for name in self.cameras:
            frame = sensors.get(name)
            if frame is None:
                continue
            rotated = np.asarray(frame)[::-1, ::-1]  # 180 degrees: flip height and width
            sensors[name] = np.ascontiguousarray(rotated)  # preserve dtype (cf. SwapChannelOrder)
        return Observation(
            state=observation.state, sensors=sensors, instruction=observation.instruction
        )


__all__ = ["Rotate180Cameras"]
