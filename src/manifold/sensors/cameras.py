"""The standard camera views, by canonical name and mount.

`agentview` is the third-person scene camera; `wrist` is mounted on the
end-effector. A policy declares these camera names for the views it consumes,
and a check matches on the name.

These are constructors, not constants, because a camera is not a complete value
until a benchmark sets its resolution: the name and mount are fixed and shared,
the shape is benchmark-specific. (Embodiments and benchmarks, which have no such
free parameter, are module constants.) The benchmark passes the shape.
"""

from __future__ import annotations

from manifold.core.sensor import Camera, Mount


def agentview(shape: tuple[int, ...]) -> Camera:
    """The third-person scene camera, at the given shape."""
    return Camera(name="agentview", shape=shape, mount=Mount.SCENE)


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


def wrist(shape: tuple[int, ...]) -> Camera:
    """The wrist-mounted camera, at the given shape."""
    return Camera(name="wrist", shape=shape, mount=Mount.WRIST)


__all__ = [
    "agentview",
    "agentview_left",
    "agentview_right",
    "over_shoulder_left",
    "wrist",
]
