"""The standard camera views, by canonical name and mount.

`agentview` is the third-person scene camera; `wrist` is mounted on the
end-effector. A policy declares these camera names for the views it consumes,
and a check matches on the name.

`calibration` is optional on every one of these: a benchmark that can say where its
camera is and how it projects passes one, and a benchmark that cannot omits it. It is
the same calibration for a colour view and the depth view beside it — one physical
camera, one pinhole — so both are passed the same value.

Each constructor takes a `modality`. With `Modality.DEPTH` the camera publishes the
metric depth from the same viewpoint as the colour camera, as a separate channel
under the key `<name>_depth`, so a colour-only policy still pairs (ADR 0007). A
depth camera takes the same orientation as the colour camera: a renderer returns
both frames from one read, so a benchmark that renders bottom-up passes the same
`orientation` to both.

`orientation` defaults to `UPRIGHT` and is passed by a benchmark whose renderer hands
back rows in some other order. It is a property of the frames a benchmark actually
publishes, not of the view, which is why it is a parameter here rather than a value
baked into a constructor.

These are constructors, not constants, because a camera is not a complete value
until a benchmark sets its resolution: the name and mount are fixed and shared,
the shape is benchmark-specific. (Embodiments, which have no such free parameter,
are module constants.) The benchmark passes the shape.
"""

from __future__ import annotations

from manifold.core.names import CameraName
from manifold.core.sensor import Camera, CameraCalibration, CameraOrientation, Modality, Mount


def _camera(
    name: CameraName,
    mount: Mount,
    shape: tuple[int, ...],
    calibration: CameraCalibration | None,
    orientation: CameraOrientation,
    modality: Modality,
) -> Camera:
    return Camera(
        name=name,
        shape=shape,
        dtype="float32" if modality is Modality.DEPTH else "uint8",
        calibration=calibration,
        orientation=orientation,
        mount=mount,
        modality=modality,
    )


def agentview(
    shape: tuple[int, ...],
    calibration: CameraCalibration | None = None,
    orientation: CameraOrientation = CameraOrientation.UPRIGHT,
    modality: Modality = Modality.RGB,
) -> Camera:
    """The third-person scene camera, at the given shape."""
    return _camera(CameraName.AGENTVIEW, Mount.SCENE, shape, calibration, orientation, modality)


def agentview_left(
    shape: tuple[int, ...],
    calibration: CameraCalibration | None = None,
    orientation: CameraOrientation = CameraOrientation.UPRIGHT,
    modality: Modality = Modality.RGB,
) -> Camera:
    """The left third-person workspace camera, at the given shape.

    One of a stereo pair of workspace views, published in place of a single
    `agentview`.
    """
    return _camera(
        CameraName.AGENTVIEW_LEFT, Mount.SCENE, shape, calibration, orientation, modality
    )


def agentview_right(
    shape: tuple[int, ...],
    calibration: CameraCalibration | None = None,
    orientation: CameraOrientation = CameraOrientation.UPRIGHT,
    modality: Modality = Modality.RGB,
) -> Camera:
    """The right third-person workspace camera, at the given shape."""
    return _camera(
        CameraName.AGENTVIEW_RIGHT, Mount.SCENE, shape, calibration, orientation, modality
    )


def head(
    shape: tuple[int, ...],
    calibration: CameraCalibration | None = None,
    orientation: CameraOrientation = CameraOrientation.UPRIGHT,
    modality: Modality = Modality.RGB,
) -> Camera:
    """The head camera: a scene camera that looks at the robot from in front of it."""
    return _camera(CameraName.HEAD, Mount.SCENE, shape, calibration, orientation, modality)


def over_shoulder_left(
    shape: tuple[int, ...],
    calibration: CameraCalibration | None = None,
    orientation: CameraOrientation = CameraOrientation.UPRIGHT,
    modality: Modality = Modality.RGB,
) -> Camera:
    """The over-the-shoulder scene camera on the left, at the given shape.

    A fixed third-person view set behind and to the left of the arm. It gets its
    own name rather than reusing `agentview` because the name is the only viewpoint
    semantics the contract carries (a camera declares name, mount, and shape —
    never extrinsics), so folding a behind-the-shoulder view into the front-facing
    `agentview` would let a policy trained on a front view pair against a viewpoint
    it never saw.
    """
    return _camera(
        CameraName.OVER_SHOULDER_LEFT, Mount.SCENE, shape, calibration, orientation, modality
    )


def over_shoulder_right(
    shape: tuple[int, ...],
    calibration: CameraCalibration | None = None,
    orientation: CameraOrientation = CameraOrientation.UPRIGHT,
    modality: Modality = Modality.RGB,
) -> Camera:
    """The over-the-shoulder scene camera on the right, at the given shape."""
    return _camera(
        CameraName.OVER_SHOULDER_RIGHT, Mount.SCENE, shape, calibration, orientation, modality
    )


def wrist(
    shape: tuple[int, ...],
    calibration: CameraCalibration | None = None,
    orientation: CameraOrientation = CameraOrientation.UPRIGHT,
    modality: Modality = Modality.RGB,
) -> Camera:
    """The wrist-mounted camera, at the given shape."""
    return _camera(CameraName.WRIST, Mount.WRIST, shape, calibration, orientation, modality)


__all__ = [
    "agentview",
    "agentview_left",
    "agentview_right",
    "head",
    "over_shoulder_left",
    "over_shoulder_right",
    "wrist",
]
