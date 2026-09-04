"""Rotate named camera images 180 degrees on the observation side.

Renamed from `camera_orientation.py`: orientation has three adapters, so each one has
its own file.

Reverses both the row and the column order of each named camera image and updates the
camera's declared `orientation` to match, so the compatibility check can see the
rotation. The new orientation is worked out from whatever the source declares, and no
pixel values change, so the calibration is left alone (ADR 0008).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
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
    operation: ClassVar[CameraOrientation] = CameraOrientation.ROTATED_180

    def __init__(self, cameras: Sequence[str]) -> None:
        self.cameras = tuple(cameras)

    def applies(self, source: BaseModel) -> bool:
        """True when the source publishes any of the named cameras.

        The rotation works from any starting orientation, so only a missing camera
        rules it out. Whether the rotation is the one a pairing needs is decided by
        `check_compatibility`, which matches the transformations required against the
        declared orientations of the policy's and benchmark's cameras.
        """
        if not isinstance(source, ObservationSpace):
            return False
        return any(source.camera(name) is not None for name in self.cameras)

    def produce(self, source: BaseModel) -> ObservationSpace:
        """The same spec with each named camera's orientation rotated 180 degrees."""
        if not isinstance(source, ObservationSpace):
            raise TypeError("Rotate180Cameras transforms an ObservationSpace source only")
        out = source
        for name in self.cameras:
            camera = source.camera(name)
            if camera is not None:
                out = out.with_camera(
                    camera.model_copy(
                        update={"orientation": camera.orientation.flipped(self.operation)}
                    )
                )
        return out

    def adapt(self, observation: Any, *, source: BaseModel) -> Observation:
        """Rotate each named camera 180°; other sensors and state pass through.

        A camera absent from `observation.sensors` is skipped, so the adapter is safe to
        point at a superset of the cameras an observation carries.

        The height and width axes are addressed from the END of the shape, so a stacked
        rank-4 clip is rotated per frame instead of having its time axis reversed.
        """
        sensors = dict(observation.sensors)
        for name in self.cameras:
            frame = sensors.get(name)
            if frame is None:
                continue
            rotated = np.asarray(frame)[..., ::-1, ::-1, :]  # 180°: height and width
            sensors[name] = np.ascontiguousarray(rotated)  # preserve dtype
        # `replace` carries the poses through unchanged: the camera did not move.
        return replace(observation, sensors=sensors)


__all__ = ["Rotate180Cameras"]
