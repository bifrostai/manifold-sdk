# 0004 - The replay log describes itself

Status: Accepted

## Context

Rendering an episode's scene needs a model of the robot and the objects in it.
Either the benchmark publishes geometry that stands on its own and whatever reads
the log draws what it is handed, or the reader holds a model per benchmark family
and the log carries only the values that drive it.

A simulator already exposes the generic form. Walking LIBERO's `sim.model` yields
named bodies with visual meshes, and `sim.data` yields a position and a
quaternion for each of them: the arm links, the gripper fingers, the objects and
the fixtures, without any reader holding a model of a Franka. Filtering to
the visual geoms is what makes this tractable, since a scene's collision
decomposition runs to a couple of hundred boxes that physics needs and rendering
does not.

## Decision

A replay log carries scene bodies: a name, a visual mesh and a pose per step,
published by the benchmark. Nothing that reads a log holds a kinematic model, a
joint-name list, a gripper polarity or a camera quirk for any benchmark.

Depth and mesh vertices in a log are float16, both narrowed from the float32 a
benchmark hands in. Depth is metres; a vertex is in its mesh's local frame, where
magnitudes are sub-metre and the relative step stays under what the scene resolves.
A log carries the visual geoms, which are exactly the ones nothing collides with,
so discarded precision cannot reach the physics. Both encodings are
self-describing, so nothing carries a scale factor.

## Consequences

An arm renders as its meshes at their world poses rather than as an articulated
chain. Playback looks the same; nothing downstream can reason about joints,
which forecloses any view of the kinematics rather than the geometry.

The work falls to the benchmark author. A benchmark publishes scene bodies or it
has no rendered scene, and there is no path where the reader compensates with
a model of the embodiment.

Meshes dominate a log's first frame. They are written once per episode to a
local mount rather than sent anywhere, which is what makes the size acceptable.

Face indices stay 32-bit. A single LIBERO mesh addresses more vertices than a
16-bit index reaches, and narrowing the ones that would fit loses more than it
saves, because a 32-bit index's zero high bytes are what a compressor removes.

Depth carries a range limit with the precision one, and the writer enforces it.
float16 holds 65504, and a cast past that yields `inf` with nothing but a numpy
warning, so a sim whose "no hit" is a large finite far plane would store a
sentinel indistinguishable from a measurement. `ReplayLogWriter` refuses such a
frame and asks for `inf` - which float16 does hold, and which a reader can act on
- or for a clip to the sensor's range. Only the benchmark can say what its own
sentinel means, so the check cannot silently pick for it.

An image channel's dtype is what separates colour from depth, since the header
declares a channel as an image without saying which kind. uint8 is colour and
floating-point is depth, and the writer refuses anything else rather than storing
one kind as the other: a uint16 depth in millimetres would otherwise decode as
float16 metres. A per-channel kind in the header would settle it outright, but it
is a format change, so the writer's refusal is what closes the hazard instead.

We accept a reader that cannot special-case a benchmark. A model
held per benchmark family is additive in the wrong direction: the second
embodiment lands beside the first rather than generalising it, and the module
holding them becomes where the camera flips and the grasp thresholds accumulate.
Refusing the first one is what prevents the rest.
