"""Exteroceptive input a policy consumes. A camera is the only kind today.

A sensor is benchmark-owned, not embodiment-intrinsic: the same robot under a
different task exposes different cameras. A benchmark publishes a list of sensors
(`benchmark.py`); a policy declares the sensor names it consumes (`policy.py`); a
check confirms the policy's required sensors are a subset of what the benchmark
publishes.

Depth, segmentation, and force/torque share this shape — a named, typed source —
and can extend the `Sensor` alias without changing the types that reference it.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from manifold.lib.compat import StrEnum


class Mount(StrEnum):
    """Where a camera is mounted."""

    SCENE = "scene"  # Third-person / external view.
    WRIST = "wrist"  # Wrist-mounted, moves with the end-effector.


class Modality(StrEnum):
    """What a camera measures. Implies the trailing-axis size and dtype."""

    RGB = "rgb"
    DEPTH = "depth"
    SEGMENTATION = "segmentation"


class CameraOrientation(StrEnum):
    """Image orientation relative to an upright scene.

    A mismatch is silent and plausible — a valid-looking but upside-down frame
    that sails through a shape check — so it is a declared, checkable convention
    (the pattern gripper polarity already set). It is lossless-bridgeable by a
    flip adapter (a 180-degree rotation or a pure vertical flip), or else
    INCOMPATIBLE.
    """

    UPRIGHT = "upright"  # row 0 = top of the scene.
    ROTATED_180 = "rotated_180"  # rows and columns reversed (a 180-degree flip).
    FLIPPED_VERTICAL = "flipped_vertical"  # rows reversed only (a top-bottom mirror).


class ChannelOrder(StrEnum):
    """Colour-channel byte order on the trailing axis."""

    RGB = "rgb"
    BGR = "bgr"


class Camera(BaseModel):
    """A single camera view.

    Shape and dtype describe the tensor; `shape` IS the declared target
    resolution, so a resolution mismatch is bridged by a (lossy) resize adapter
    keyed on it rather than a separate field. `orientation` and `channel_order`
    are the small typed conventions whose mismatch is silent and plausible
    (a valid-looking but flipped or colour-swapped frame), declared so a check
    can catch them and a lossless adapter can bridge them. `name` is how a policy
    and a benchmark refer to the same view ("agentview", "wrist") without either
    hardcoding the other's internal naming. `mount` is provenance — a deferred
    tag, not a bridging axis — so it is not compared when matching conventions.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    shape: tuple[int, ...] = Field(description="Tensor shape, typically (H, W, 3) for RGB.")
    dtype: str = "uint8"
    mount: Mount = Mount.SCENE
    modality: Modality = Modality.RGB
    orientation: CameraOrientation = CameraOrientation.UPRIGHT
    channel_order: ChannelOrder = ChannelOrder.RGB

    def example(self) -> np.ndarray:
        """A zero-filled array of this camera's shape and dtype."""
        return np.zeros(self.shape, dtype=self.dtype)

    def validate_value(self, value: np.ndarray) -> None:
        """Check `value` matches this camera's shape and dtype."""
        actual_shape = tuple(int(d) for d in value.shape)
        if actual_shape != self.shape:
            raise ValueError(
                f"Camera {self.name!r}: shape mismatch: expected {self.shape}, got {actual_shape}"
            )
        if str(value.dtype) != self.dtype:
            raise ValueError(
                f"Camera {self.name!r}: dtype mismatch: expected {self.dtype}, got {value.dtype}"
            )


# The sensor kinds a benchmark may publish. One kind today; depth and force can
# extend this alias without changing the benchmark or policy types that use it.
Sensor = Camera


__all__ = [
    "Camera",
    "CameraOrientation",
    "ChannelOrder",
    "Modality",
    "Mount",
    "Sensor",
]
