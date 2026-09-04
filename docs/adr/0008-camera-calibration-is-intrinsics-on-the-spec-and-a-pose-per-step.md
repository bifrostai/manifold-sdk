# 0008 - Camera calibration is intrinsics on the spec and a pose per step

Status: Accepted

## Context

A policy that reasons about where things are, rather than only what a frame
looks like, needs to map image pixels into world space. That needs a camera's
intrinsics and its extrinsics.

The two differ in lifetime, and that is what decides where each belongs.
Intrinsics are static per camera setup - a field of view does not change while
a run proceeds - so they are a property of the camera. Extrinsics change
whenever the camera moves, so a wrist camera has a new pose every step.

## Decision

Intrinsics are declared on the spec and cannot be bridged. Extrinsics are
pulled at every sim step.

Extrinsics are encoded as a 4x4 camera-to-frame pose under
`Observation.extrinsics[camera.name]`, for every camera on every step.
Sealing into `.replay`, and eventually `.rrd`, compresses that to a position
and an `xyzw` quaternion - 7 floats rather than 16.

Forward-filling is the reader's job. A log stores a pose only where one moved
and fills the rest forward, so a consumer never looks backwards for the last one
written.

The pose is camera-to-world, the wrist camera's included. A consumer expecting
camera-to-arm composes it with the end-effector pose itself.

A calibration is valid when `orientation` agrees with what `axes` implies: for
`OPENCV`, whose +Y runs down the image, that means `UPRIGHT`.

## Consequences

`BRIDGE_PROTOCOL_VERSION` is incremented. `extrinsics` is omitted when the
benchmark does not emit it.

A benchmark that emits calibration can still pair with a policy that does not
need it.

There is no distortion model. Simulators all render an ideal pinhole camera.

`ResizeCameras` scales the intrinsics and shifts the principal point when
`pad=True`; every other observation adapter leaves intrinsics and pose alone.
Resampling changes the size of a pixel, and a focal length is measured in
pixels.

Poses are float64. They are geometry rather than network input.

The model never receives a pose. `SourceKind` is
`CAMERA | STATE | INSTRUCTION`, so a packing pairing cannot address
`extrinsics`; a geometry-aware policy needs a fourth kind, or an endpoint
reading the `Observation` directly.

Nothing consumes calibration yet, so `verify` is the only gate - no run-time
check compares a published pose against the declared conventions.
