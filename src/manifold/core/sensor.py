"""Exteroceptive input a policy consumes. A camera is the only kind today.

A sensor is benchmark-owned, not embodiment-intrinsic: the same robot under a
different task exposes different cameras. A benchmark publishes a list of sensors
(`benchmark.py`); a policy declares the sensor names it consumes (`policy.py`); a
check confirms the policy's required sensors are a subset of what the benchmark
publishes.

Depth is a `Camera` and not a kind of its own (ADR 0007): a named, shaped, dtyped
array read from a viewpoint is exactly what `Camera` describes, so `modality` is
all it needs. Force/torque does not share that shape, and can extend the `Sensor`
alias without changing the types that reference it.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from manifold.core.conventions import CameraAxes, Frame
from manifold.lib.compat import StrEnum


class Mount(StrEnum):
    """Where a camera is mounted."""

    SCENE = "scene"  # Third-person / external view.
    WRIST = "wrist"  # Wrist-mounted, moves with the end-effector.


class Modality(StrEnum):
    """What a camera measures. Implies the trailing-axis size and dtype.

    A depth camera's values are metres and its no-hit marker is `inf` — the
    convention `manifold.replay` already fixed, adopted here rather than declared
    as a second axis to check and bridge (ADR 0007). Both are silent when wrong:
    millimetres, a normalised buffer, and a far-plane constant are all plausible
    arrays that no shape check can reject.
    """

    RGB = "rgb"  # uint8 colour on the trailing axis, (H, W, 3).
    DEPTH = "depth"  # Floating-point metres, (H, W, 1); `inf` where the ray hit nothing.
    SEGMENTATION = "segmentation"


class CameraOrientation(StrEnum):
    """Image orientation relative to an upright scene.

    An upside-down frame has the right shape, so a shape check will not catch it;
    declaring the orientation makes the mismatch visible. The rows and the columns are
    the only two things that can be reversed, so these four are every possible
    orientation.
    """

    UPRIGHT = "upright"  # row 0 = top of the scene.
    ROTATED_180 = "rotated_180"  # rows and columns reversed (a 180-degree flip).
    FLIPPED_VERTICAL = "flipped_vertical"  # rows reversed only (a top-bottom mirror).
    FLIPPED_HORIZONTAL = "flipped_horizontal"  # columns reversed only (a left-right mirror).

    def flipped(self, by: CameraOrientation) -> CameraOrientation:
        """This orientation with `by`'s mirrors applied to it.

        An orientation is two yes/no facts — rows reversed, columns reversed — so
        combining two of them is an XOR of each fact. That lets one adapter handle any
        starting orientation instead of only the `UPRIGHT -> X` case.
        """
        mine, theirs = _ORIENTATION_MIRRORS[self], _ORIENTATION_MIRRORS[by]
        return _MIRRORS_ORIENTATION[(mine[0] ^ theirs[0], mine[1] ^ theirs[1])]


# Each orientation as (rows reversed, columns reversed) against an upright scene, and
# the way back. Kept beside the enum rather than inside it because a StrEnum member's
# value is its wire form, and these axes are the algebra over the members, not data on
# one: `flipped` is the only reader.
_ORIENTATION_MIRRORS: dict[CameraOrientation, tuple[bool, bool]] = {
    CameraOrientation.UPRIGHT: (False, False),
    CameraOrientation.FLIPPED_VERTICAL: (True, False),
    CameraOrientation.FLIPPED_HORIZONTAL: (False, True),
    CameraOrientation.ROTATED_180: (True, True),
}
_MIRRORS_ORIENTATION: dict[tuple[bool, bool], CameraOrientation] = {
    mirrors: orientation for orientation, mirrors in _ORIENTATION_MIRRORS.items()
}


class ChannelOrder(StrEnum):
    """Colour-channel byte order on the trailing axis."""

    RGB = "rgb"
    BGR = "bgr"


# Relative tolerance when two declarations of one pinhole are compared. All four
# values are derived rather than typed: `from_fov` runs a tangent, `scaled` and
# `translated` carry the result through a resize, and a policy states the same
# camera by whichever route its own config took. An exact comparison would refuse
# a pairing over the last bits of that arithmetic, so compare them with a
# relative tolerance and never for equality.
_INTRINSICS_REL_TOL = 1e-6


class CameraIntrinsics(BaseModel):
    """A pinhole camera's focal lengths and principal point, in pixels.

    What turns a depth frame into metric points: a pixel `(u, v)` at depth `d`
    back-projects to `((u - cx) * d / fx, (v - cy) * d / fy, d)` in the camera's own
    frame. Without it a depth image is a per-pixel number with no scale — the same
    array describes a different scene at every field of view.

    An ideal pinhole, with no distortion coefficients. The simulators shipped so far
    render exactly that; a real camera's distortion is a per-unit calibration a
    benchmark would have to publish per device rather than per model, which is a
    different problem and is not modelled ahead of one (ADR 0008).

    Compared when matching, unlike the extrinsic: a policy trained at one field of
    view served another is being shown a differently-projected world, and a resize
    matches pixel counts without matching focal length.
    """

    model_config = ConfigDict(frozen=True)

    fx: float = Field(gt=0.0, description="Focal length along x, in pixels.")
    fy: float = Field(gt=0.0, description="Focal length along y, in pixels.")
    cx: float = Field(description="Principal point x, in pixels from the left edge.")
    cy: float = Field(description="Principal point y, in pixels from the declared top row.")

    @classmethod
    def from_fov(cls, *, fovy_degrees: float, height: int, width: int) -> CameraIntrinsics:
        """Intrinsics for a renderer that describes its camera by vertical field of view.

        The pinhole a vertical FOV implies: one focal length for both axes, and the
        principal point at the image centre. This is how MuJoCo and SAPIEN describe a
        camera, so it is how a benchmark on either derives what it publishes.
        """
        focal = 0.5 * height / math.tan(math.radians(fovy_degrees) * 0.5)
        return cls(fx=focal, fy=focal, cx=width / 2.0, cy=height / 2.0)

    def matrix(self) -> np.ndarray:
        """The 3x3 K these values stand for, row-major float64."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=float
        )

    def scaled(self, *, x: float, y: float) -> CameraIntrinsics:
        """The same camera resampled by `x` and `y`, which scales all four values.

        Resampling an image scales the focal lengths and the principal point alike:
        the pinhole is unchanged and only the pixel grid it is measured in moves.
        """
        return CameraIntrinsics(fx=self.fx * x, fy=self.fy * y, cx=self.cx * x, cy=self.cy * y)

    def translated(self, *, x: float, y: float) -> CameraIntrinsics:
        """The same camera with its principal point moved, for a crop or a pad."""
        return CameraIntrinsics(fx=self.fx, fy=self.fy, cx=self.cx + x, cy=self.cy + y)

    def is_close_to(self, other: CameraIntrinsics) -> bool:
        """Whether `other` describes the same pinhole, to floating-point tolerance.

        What matching compares, rather than `==`: all four values are derived, so
        two correct descriptions of one camera differ in their last bits. See
        `_INTRINSICS_REL_TOL` for why the tolerance is safe.
        """
        return all(
            math.isclose(mine, theirs, rel_tol=_INTRINSICS_REL_TOL)
            for mine, theirs in (
                (self.fx, other.fx),
                (self.fy, other.fy),
                (self.cx, other.cx),
                (self.cy, other.cy),
            )
        )


class CameraCalibration(BaseModel):
    """A camera's calibration: its intrinsics, and the conventions its pose follows.

    Declaring this on a camera means the benchmark publishes that camera's 4x4
    camera-to-`frame` extrinsic in every observation, under the camera's name (see
    `Observation.extrinsics`). The matrix is data and lives there rather than here,
    because a wrist camera's pose changes every step; the intrinsics and the two
    conventions are constant and live here, where a check can compare them.

    The pose is not kept on the spec even for a camera that never moves. One
    mechanism covers both, a still camera simply repeats itself, and 16 floats a
    step is nothing beside the depth frame they describe (ADR 0008).
    """

    model_config = ConfigDict(frozen=True)

    intrinsics: CameraIntrinsics
    axes: CameraAxes = Field(
        default=CameraAxes.OPENCV, description="Axis convention of the camera's own frame."
    )
    frame: Frame = Field(
        default=Frame.WORLD, description="The frame the extrinsic maps the camera INTO."
    )


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
    calibration: CameraCalibration | None = None

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
    "CameraCalibration",
    "CameraIntrinsics",
    "CameraOrientation",
    "ChannelOrder",
    "Modality",
    "Mount",
    "Sensor",
]
