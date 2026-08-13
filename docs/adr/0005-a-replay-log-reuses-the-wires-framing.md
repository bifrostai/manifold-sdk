# 0005 - A replay log reuses the wire's framing

Status: Accepted

## Context

ADR 0003 puts the replay in a file beside the rollup, so the file needs a
format. Three properties decide it, on top of the JSON exclusion 0003 already
records.

A log is written while the episode runs, and a shard can be killed mid-episode.
A format that writes its index at close leaves an unreadable file rather than a
short one.

The payload is arrays at two scales: the scene once, then a pose per body and a
depth image per camera per step. Nested number lists take several bytes per
component and the scene has hundreds of thousands of them.

The benchmark writes the log and the runner reads it from a different image, so
the format may not add a dependency to either side, and reading it has to decode
data rather than execute it.

The wire already meets all three. `pack_stream_frame` puts a four-byte
big-endian length in front of a msgpack body and `read_stream_frame` reads one
back, and the codec packs an array as binary with its shape alongside.

## Decision

A replay log is a stream of length-prefixed msgpack frames, packed by
`wire.pack_stream_frame` and read by `wire.read_stream_frame`, carrying frame
types of its own and a version of its own. Frames are appended as the episode
runs: the scene first, then one per step. Arrays use the codec's binary
wrappers rather than nested lists.

## Consequences

A truncated log is readable to its last complete frame. A short read ends the
stream, so an episode killed halfway still renders the steps it wrote.

A whole frame that will not decode raises instead of ending the stream. The two
cases are distinguishable - a truncated frame is short of the bytes its prefix
declared, a corrupt one is not - and conflating them would let a single flipped
byte discard every frame after it, indistinguishable from a normally short
tail. `wire.unpack_frame` returns None for both, so the replay reader tracks
whether the read was short and raises when it was not.

The writer refuses a frame larger than the cap the reader enforces. That cap
exists on the wire to bound what an untrusted peer can make a reader allocate,
and `read_stream_frame` raises rather than stopping when a prefix exceeds it - so
one oversized scene frame would invalidate the whole log rather than its own tail.
Checking at write time keeps the log readable and identifies the frame that was
too big, which is where the scene can still be decimated.

A log declares its own version rather than sharing `BRIDGE_PROTOCOL_VERSION`, so
a wire bump leaves stored logs valid and a log format change leaves the wire
alone. The framing helpers gain a second caller and stay one implementation.

Neither side gains a dependency. msgpack backs the codec the benchmark and the
runner already use for the stream between them.

Nothing a log carries is executed, which is what rules out pickle. It would
otherwise be the shortest path from a numpy array to a file.

The length prefix bounds each read, so a reader never scans for a delimiter and
a frame holding arbitrary bytes does not need escaping.

`npz` is rejected for the first property: the zip central directory lands at
close, and it would put numpy in the reader. A bespoke header and record table
is rejected for the third: it adds nothing the framing lacks and it is a parser
to write and test.
