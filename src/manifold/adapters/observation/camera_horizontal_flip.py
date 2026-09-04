"""Mirror named camera images left-to-right on the observation side.

Reverses the column order of each named camera image and updates the camera's declared
`orientation` to match. Nothing asks for this flip directly, but a bottom-up frame
rotated 180 degrees lands on it, so that state needs an adapter of its own.
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


class FlipHorizontalCameras(ObservationAdapter):
    """Mirror the named camera channels left-to-right and advance their orientation."""

    from_spec: ClassVar[type[BaseModel]] = ObservationSpace
    to_spec: ClassVar[type[BaseModel]] = ObservationSpace
    lossless: ClassVar[bool] = True
    operation: ClassVar[CameraOrientation] = CameraOrientation.FLIPPED_HORIZONTAL

    def __init__(self, cameras: Sequence[str]) -> None:
        self.cameras = tuple(cameras)

    def applies(self, source: BaseModel) -> bool:
        """True when the source publishes any named camera.

        Any orientation is a legal source: the mirror composes with whatever is
        declared, and being its own inverse it bridges either direction. What the
        edge cannot do is change which cameras exist.
        """
        if not isinstance(source, ObservationSpace):
            return False
        return any(source.camera(name) is not None for name in self.cameras)

    def produce(self, source: BaseModel) -> ObservationSpace:
        """The same spec with each named camera mirrored left-to-right."""
        if not isinstance(source, ObservationSpace):
            raise TypeError("FlipHorizontalCameras transforms an ObservationSpace source only")
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
        """Mirror each named camera left-to-right; other sensors and state pass through.

        A camera that is absent from `observation.sensors` is skipped, so the adapter is
        safe to point at a superset of the cameras a given observation carries. The width
        axis is addressed from the end of the shape, so a stacked rank-4 clip is mirrored
        per frame rather than along some other axis.
        """
        sensors = dict(observation.sensors)
        for name in self.cameras:
            frame = sensors.get(name)
            if frame is None:
                continue
            mirrored = np.asarray(frame)[..., :, ::-1, :]  # horizontal: the width axis only
            sensors[name] = np.ascontiguousarray(mirrored)  # preserve dtype
        return Observation(
            state=observation.state, sensors=sensors, instruction=observation.instruction
        )


__all__ = ["FlipHorizontalCameras"]
