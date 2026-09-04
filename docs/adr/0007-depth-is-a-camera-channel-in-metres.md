# 0007 - Depth is a camera channel in metres

Status: Accepted

## Context

ADR 0003 already puts depth inside the pairing contract: an RGB-D policy is a
real consumer, `Camera.dtype` expresses a float camera, and only the codec
cannot carry one yet. This decision settles how such a channel is named, what
its values mean and how it is encoded.

The HF LIBERO wrapper renders colour only, so a benchmark driver renders depth
itself off the live sim it already has open.

## Decision

A depth channel is a camera, named after the colour camera it accompanies -
`agentview` becomes `agentview_depth` - and does not need a new spec type or a new
`Observation` field. Depth is encoded as a plain array of `<f2` or `<f4` floats,
packed by its own dtype rather than as an encoded image. Resizing switches to
nearest-neighbour and pads with `inf`.

Values are in metres and `inf` means the ray hit nothing. Both are fixed by
convention and both match `manifold.replay`. A benchmark rendering anything
else converts before publishing.

A sensor is uint8 or floating-point and nothing else - the bridge refuses an
integer array outright, so raw integer depth cannot be sent. A consumer that
needs raw units and a scale quantises on its own side.

A depth frame is `(H, W, 1)`, not `(H, W)`. The trailing axis is the
modality's, as colour's 3 is.

## Alternatives considered

- **TIFF, or another depth image encoding.** It makes depth lossy by default
  where every other sensor is lossless.
- **A sensor kind of its own.** It repeats every convention field across four
  modules for an array `Camera` already describes.

## Consequences

`BRIDGE_PROTOCOL_VERSION` is incremented.

**Amended.** Version 2 put depth under `sensors`, and the consequence recorded
here was that "older peers expect an image wrapper under every key in `sensors`
and raise on a depth array". That is not a consequence worth accepting: it means
a published RGB-only policy cannot be paired with an RGB-D benchmark at all, and
it fails on the first observation of every run rather than at the handshake.

The placement is what did it, and the rule is exact, because
`decode_observation` reads the keys it supports through `.get()` and never
validates the payload's key set:

> A new top-level payload key is additive. A new wrapper kind inside a dict a
> peer already iterates is not.

`extrinsics` (ADR 0008) is additive under that rule; depth under `sensors` was
not. The latest version carries depth in its own top-level `depth` key.

**This decision is unchanged by the amendment.** Depth is still a camera, still
named after the colour camera it accompanies, still metres with `inf` for a
no-hit, and still does not need a new spec type or a new `Observation` field. Depth
remains in `Observation.sensors` on both sides of the wire; only the payload
keeps the two apart, and `encode_observation` routes on the wrapper the dtype
already chose. The domain model did not have to dictate the wire layout, and
conflating the two is what made an addition breaking.

A peer built before `depth` sees a colour-only `sensors` and runs, losing only
the channel it does not consume — the behaviour `check_compatibility` accepts
for such a pairing. `_decode_sensor` still accepts an
`__ndarray__` node under `sensors`, so payloads on older versions decode unchanged,
and the key is omitted entirely for an observation without depth.

The latest version does not suppress depth for a peer that discards it, so an RGB-only
pairing still carries it on the wire (~512 KiB/step against ~20 KiB of JPEG
colour, and one offscreen render per camera per step). Suppression depends on
what the peer consumes; the platform holds both specs before dispatch, which is
a better place to decide it than in-band negotiation, and it is not settled
here.

Depth is uncompressed. Two 256x256 float32 frames are ~512 KiB per step
against ~20 KiB for the two JPEG colour frames.

A benchmark performs one more offscreen render per camera per step.

`inf` is not feedable to a network. Every consumer maps it to its own invalid
marker, usually 0.

Declaring `float16` halves the wire and loses precision - ~1 mm at 1 m, ~2 mm
at 3 m. That is fine for a scene camera and coarse against a wrist camera's
native resolution.

Nothing checks a published sensor against its declared shape at run time, so a
benchmark that advertises depth and sends none is caught by `verify` or a test
and not otherwise.
