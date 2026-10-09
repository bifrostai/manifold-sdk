"""The standard camera views, by canonical name and mount.

`agentview` is the third-person scene camera; `wrist` is mounted on the
end-effector. A policy declares these camera names for the views it consumes,
and a check matches on the name.

`calibration` is optional on every one of these: a benchmark that can say where its
camera is and how it projects passes one, and a benchmark that cannot omits it. It is
the same calibration for a colour view and the depth view beside it — one physical
camera, one pinhole — so both are passed the same value.

A `*_depth` constructor is the metric-depth view from the same viewpoint as the
colour camera it is named for, published as a separate channel so a colour-only
policy still pairs (ADR 0007). It carries the same orientation as its sibling: a
renderer hands both frames back from one read, so a benchmark that renders bottom-up
passes the same `orientation` to both.

`orientation` defaults to `UPRIGHT` and is passed by a benchmark whose renderer hands
back rows in some other order. It is a property of the frames a benchmark actually
publishes, not of the view, which is why it is a parameter here rather than a value
baked into a constructor.

These are constructors, not constants, because a camera is not a complete value
until a benchmark sets its resolution: the name and mount are fixed and shared,
the shape is benchmark-specific. (Embodiments and benchmarks, which have no such
free parameter, are module constants.) The benchmark passes the shape.
"""

from __future__ import annotations

from manifold.core.sensor import Camera, CameraCalibration, CameraOrientation, Modality, Mount
from manifold.lib.compat import StrEnum


class CameraName(StrEnum):
    """The name of each camera in the catalogue, one per viewpoint."""

    AGENTVIEW = "agentview"
    AGENTVIEW_DEPTH = "agentview_depth"
    AGENTVIEW_LEFT = "agentview_left"
    AGENTVIEW_RIGHT = "agentview_right"
    OVER_SHOULDER_LEFT = "over_shoulder_left"
    OVER_SHOULDER_RIGHT = "over_shoulder_right"
    HEAD = "head"
    WRIST = "wrist"
    WRIST_DEPTH = "wrist_depth"


def agentview(
    shape: tuple[int, ...],
    calibration: CameraCalibration | None = None,
    orientation: CameraOrientation = CameraOrientation.UPRIGHT,
) -> Camera:
    """The third-person scene camera, at the given shape."""
    return Camera(
        name=CameraName.AGENTVIEW,
        shape=shape,
        calibration=calibration,
        orientation=orientation,
        mount=Mount.SCENE,
    )


def agentview_depth(
    shape: tuple[int, ...],
    calibration: CameraCalibration | None = None,
    orientation: CameraOrientation = CameraOrientation.UPRIGHT,
) -> Camera:
    """The third-person scene camera's metric depth, at the given shape."""
    return Camera(
        name=CameraName.AGENTVIEW_DEPTH,
        shape=shape,
        calibration=calibration,
        orientation=orientation,
        dtype="float32",
        mount=Mount.SCENE,
        modality=Modality.DEPTH,
    )


def agentview_left(
    shape: tuple[int, ...],
    calibration: CameraCalibration | None = None,
    orientation: CameraOrientation = CameraOrientation.UPRIGHT,
) -> Camera:
    """The left of a pair of third-person workspace cameras, at the given shape."""
    return Camera(
        name=CameraName.AGENTVIEW_LEFT,
        shape=shape,
        calibration=calibration,
        orientation=orientation,
        mount=Mount.SCENE,
    )


def agentview_right(
    shape: tuple[int, ...],
    calibration: CameraCalibration | None = None,
    orientation: CameraOrientation = CameraOrientation.UPRIGHT,
) -> Camera:
    """The right third-person workspace camera, at the given shape."""
    return Camera(
        name=CameraName.AGENTVIEW_RIGHT,
        shape=shape,
        calibration=calibration,
        orientation=orientation,
        mount=Mount.SCENE,
    )


def over_shoulder_left(
    shape: tuple[int, ...],
    calibration: CameraCalibration | None = None,
    orientation: CameraOrientation = CameraOrientation.UPRIGHT,
) -> Camera:
    """The scene camera behind and to the left of the arm, at the given shape.

    It is not `agentview`, because a camera's name is the only viewpoint the
    contract carries, so a policy trained on a front-facing view must not pair
    against this one.
    """
    return Camera(
        name=CameraName.OVER_SHOULDER_LEFT,
        shape=shape,
        calibration=calibration,
        orientation=orientation,
        mount=Mount.SCENE,
    )


def over_shoulder_right(
    shape: tuple[int, ...],
    calibration: CameraCalibration | None = None,
    orientation: CameraOrientation = CameraOrientation.UPRIGHT,
) -> Camera:
    """The scene camera behind and to the right of the arm, at the given shape."""
    return Camera(
        name=CameraName.OVER_SHOULDER_RIGHT,
        shape=shape,
        calibration=calibration,
        orientation=orientation,
        mount=Mount.SCENE,
    )


def head(
    shape: tuple[int, ...],
    calibration: CameraCalibration | None = None,
    orientation: CameraOrientation = CameraOrientation.UPRIGHT,
) -> Camera:
    """The scene camera in front of and above the arm, facing it, at the given shape."""
    return Camera(
        name=CameraName.HEAD,
        shape=shape,
        calibration=calibration,
        orientation=orientation,
        mount=Mount.SCENE,
    )


def wrist(
    shape: tuple[int, ...],
    calibration: CameraCalibration | None = None,
    orientation: CameraOrientation = CameraOrientation.UPRIGHT,
) -> Camera:
    """The wrist-mounted camera, at the given shape."""
    return Camera(
        name=CameraName.WRIST,
        shape=shape,
        calibration=calibration,
        orientation=orientation,
        mount=Mount.WRIST,
    )


def wrist_depth(
    shape: tuple[int, ...],
    calibration: CameraCalibration | None = None,
    orientation: CameraOrientation = CameraOrientation.UPRIGHT,
) -> Camera:
    """The wrist-mounted camera's metric depth, at the given shape."""
    return Camera(
        name=CameraName.WRIST_DEPTH,
        shape=shape,
        calibration=calibration,
        orientation=orientation,
        dtype="float32",
        mount=Mount.WRIST,
        modality=Modality.DEPTH,
    )


__all__ = [
    "CameraName",
    "agentview",
    "agentview_depth",
    "agentview_left",
    "agentview_right",
    "head",
    "over_shoulder_left",
    "over_shoulder_right",
    "wrist",
    "wrist_depth",
]
