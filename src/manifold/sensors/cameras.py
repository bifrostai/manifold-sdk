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
policy still pairs (ADR 0007). It carries the same orientation as its sibling,
which is why neither sets one here: a renderer hands both frames back from one
read, so a pairing that reorients one reorients both.

These are constructors, not constants, because a camera is not a complete value
until a benchmark sets its resolution: the name and mount are fixed and shared,
the shape is benchmark-specific. (Embodiments and benchmarks, which have no such
free parameter, are module constants.) The benchmark passes the shape.
"""

from __future__ import annotations

from manifold.core.sensor import Camera, CameraCalibration, Modality, Mount


def agentview(shape: tuple[int, ...], calibration: CameraCalibration | None = None) -> Camera:
    """The third-person scene camera, at the given shape."""
    return Camera(name="agentview", shape=shape, mount=Mount.SCENE, calibration=calibration)


def agentview_depth(shape: tuple[int, ...], calibration: CameraCalibration | None = None) -> Camera:
    """The third-person scene camera's metric depth, at the given shape."""
    return Camera(
        name="agentview_depth",
        shape=shape,
        dtype="float32",
        mount=Mount.SCENE,
        modality=Modality.DEPTH,
        calibration=calibration,
    )


def agentview_left(shape: tuple[int, ...]) -> Camera:
    """The left third-person workspace camera, at the given shape.

    RoboCasa publishes a stereo pair of workspace views (left + right) alongside
    the wrist camera, rather than the single `agentview` LIBERO uses.
    """
    return Camera(name="agentview_left", shape=shape, mount=Mount.SCENE)


def agentview_right(shape: tuple[int, ...]) -> Camera:
    """The right third-person workspace camera, at the given shape."""
    return Camera(name="agentview_right", shape=shape, mount=Mount.SCENE)


def over_shoulder_left(shape: tuple[int, ...]) -> Camera:
    """The over-the-shoulder scene camera on the left, at the given shape.

    RoboLab's `OverShoulderLeftCameraCfg` — a fixed third-person view set behind
    and to the left of the arm. It gets its own name rather than reusing
    `agentview` because the name is the only viewpoint semantics the contract
    carries (a camera declares name, mount, and shape — never extrinsics), so
    folding a behind-the-shoulder view into the front-facing `agentview` would
    let a LIBERO-trained policy pair against a viewpoint it never saw.
    """
    return Camera(name="over_shoulder_left", shape=shape, mount=Mount.SCENE)


def wrist(shape: tuple[int, ...], calibration: CameraCalibration | None = None) -> Camera:
    """The wrist-mounted camera, at the given shape."""
    return Camera(name="wrist", shape=shape, mount=Mount.WRIST, calibration=calibration)


def wrist_depth(shape: tuple[int, ...], calibration: CameraCalibration | None = None) -> Camera:
    """The wrist-mounted camera's metric depth, at the given shape."""
    return Camera(
        name="wrist_depth",
        shape=shape,
        dtype="float32",
        mount=Mount.WRIST,
        modality=Modality.DEPTH,
        calibration=calibration,
    )


__all__ = [
    "agentview",
    "agentview_depth",
    "agentview_left",
    "agentview_right",
    "over_shoulder_left",
    "wrist",
    "wrist_depth",
]
