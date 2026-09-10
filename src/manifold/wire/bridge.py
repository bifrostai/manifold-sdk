"""The public data-plane bridge protocol: domain types on the wire, framed.

`codec.py` is the low-level msgpack codec, independent of the SDK's domain
types. This module sits on top of it and owns the *public* bridge protocol: the
mapping of `Observation` and `Action` onto the wire wrappers, the frame types the
two sides exchange, and transport-agnostic length-prefixed stream framing. The
transport itself is out of scope — framing is expressed over an opaque byte stream
(`read_stream_frame` takes a `read_exactly` callable, not a socket), so the same
code drives a socket, a pipe, or a test buffer.

`BRIDGE_PROTOCOL_VERSION` marks the frame-and-codec contract; bump it when the
frame types, payload layout, or stream framing change in a way the other side must
agree on.

What makes a change additive is where it goes, and the rule is exact, because
`decode_observation` reads its supported keys through `.get()` with a default and
never validates the payload's key set:

    A new TOP-LEVEL payload key is additive. A new wrapper kind inside a dict a
    peer already iterates is not.

Version 3's `extrinsics` is the additive case: a peer built before the
key skips it and loses only the poses. Version 2's depth was the other case, and
the reason a published policy could not be paired with an RGB-D benchmark at all
— it put an `__ndarray__` wrapper under `sensors`, which every peer already walks
and decodes unconditionally, so a version-1 peer raised on the first observation
of every run rather than ignoring a channel it does not consume.

Version 4 moves depth to its own top-level `depth` key, which brings it under the
rule. `Observation` is unchanged — depth is still a camera in `sensors` to both
benchmark and policy authors (ADR 0007), and only the payload keeps the two apart,
so an old peer sees a colour-only `sensors` and runs. The key is omitted for an
observation without depth, and `_decode_sensor` still accepts an `__ndarray__`
node under `sensors`, so version 2 and 3 payloads decode unchanged.

The `lane` envelope field and the OBSERVATION/ACTION `action_prefix` and
`timestep` fields are reserved (additive, optional, defaulted), so a current
synchronous peer round-trips identically without a version bump.

Every decoder here parses an untrusted frame, so a malformed value is a
`ValueError` by design (mirroring how the wire layer returns None on bad input);
this is not a Python type-contract violation, hence the `# noqa: TRY004` guards.
"""

from __future__ import annotations

import io
import math
import socket
import struct
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from manifold.core.values import Action, Observation
from manifold.lib.compat import StrEnum
from manifold.wire.codec import (
    DEFAULT_FRAME_LANE,
    ImageFormat,
    pack_encoded_image,
    pack_frame,
    pack_ndarray,
    unpack_frame,
    unpack_ndarray,
    unpack_ndarray_full,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# The frame-and-codec contract version. Bump when that contract changes.
BRIDGE_PROTOCOL_VERSION = 4

# A length prefix is a 4-byte big-endian unsigned integer, so a single frame is
# capped at 4 GiB by the wire format alone — far above any real payload.
_LENGTH_PREFIX = ">I"
_LENGTH_PREFIX_BYTES = 4

# A reader refuses a frame whose declared length exceeds this, before reading the
# body, so an untrusted peer cannot demand an unbounded allocation. 64 MiB sits
# well above any observation or action while still bounding a hostile request.
MAX_FRAME_BYTES = 64 * 1024 * 1024


def _attach_rtc_fields(
    payload: dict[str, Any],
    *,
    action_prefix: np.ndarray | None,
    timestep: int | None,
) -> None:
    """Attach the optional RTC fields to a payload, omitting any that is None.

    Keeping the keys absent (rather than present-and-None) means a current peer's
    payload is byte-for-byte what it was before these fields existed, so the
    additive change is invisible on the wire until a future RTC path sets them.
    """
    if action_prefix is not None:
        payload["action_prefix"] = pack_ndarray(
            action_prefix.reshape(-1).tolist(), shape=list(action_prefix.shape)
        )
    if timestep is not None:
        payload["timestep"] = int(timestep)


def _encode_colour(array: np.ndarray, image_format: ImageFormat, *, name: str) -> dict[str, Any]:
    """Encode a uint8 colour frame as an `__image__` wrapper in the chosen format."""
    shape = list(array.shape)
    if image_format == "raw":
        return pack_encoded_image(array.tobytes(), format_="raw", shape=shape)
    if image_format in ("jpeg", "png"):
        data = _compress_image(array, image_format)
        return pack_encoded_image(data, format_=image_format, shape=shape)
    raise ValueError(
        f"sensor {name!r}: unsupported image_format {image_format!r}: use 'raw', 'jpeg', or 'png'"
    )


def _encode_sensor(array: np.ndarray, image_format: ImageFormat, *, name: str) -> dict[str, Any]:
    """Encode one sensor array: uint8 colour as an image, float depth as metres.

    The dtype chooses, as it does in `replay.log`: uint8 is a colour frame and
    goes through `image_format`, floating-point is depth in metres and is encoded as
    an `__ndarray__` wrapper at its own dtype, losslessly (ADR 0007). Anything
    else, and any float array shaped like colour, is refused rather than sent as
    the other kind — a payload records no per-sensor kind, so a reader has only
    the wrapper and the dtype to go on.

    Raises:
        ValueError: If the array is neither a colour nor a depth frame, its dtype
            is one the wire does not carry, or `image_format` is not a recognised
            format.
    """
    if array.dtype == np.uint8:
        return _encode_colour(array, image_format, name=name)
    if array.dtype.kind != "f":
        raise ValueError(
            f"sensor {name!r} has dtype {array.dtype}: a colour frame is uint8 and a depth "
            "frame is floating-point metres"
        )
    if array.ndim == 3 and array.shape[2] in (3, 4):
        raise ValueError(
            f"sensor {name!r} is floating-point with shape {list(array.shape)}: a depth frame "
            "has one channel, so convert a colour frame to uint8"
        )
    # The wire dtype follows the array's own, so a benchmark declaring float16
    # depth halves its payload without a second option to select.
    return pack_ndarray(
        array.reshape(-1),
        dtype=array.dtype.newbyteorder("<").str,
        shape=list(array.shape),
    )


def _decode_depth(name: str, node: Any) -> np.ndarray:
    """Decode an `__ndarray__` depth node, at the dtype it was sent at.

    Not via `_decode_state_array`, which coerces to float32: a depth frame's dtype
    is declared on its `Camera`, and narrowing or widening it here would hand the
    consumer an array its own spec rejects. Writable and native-endian for the
    reason that decoder gives.
    """
    array = unpack_ndarray_full(node)
    if array is None:
        raise ValueError(f"sensor {name!r} is not a decodable ndarray wrapper")
    return np.array(array, dtype=array.dtype.newbyteorder("="))


def _decode_colour(name: str, node: dict[str, Any]) -> np.ndarray:
    """Decode one `__image__` wrapper node back into a uint8 array."""
    if node.get("__image__") is not True:
        raise ValueError(f"sensor {name!r} is neither an image nor an ndarray wrapper")
    data = node.get("data")
    fmt = node.get("format")
    if not isinstance(data, (bytes, bytearray)) or not isinstance(fmt, str):
        # Malformed frame value, not a type misuse — see decode_observation.
        raise ValueError(f"sensor {name!r} is missing its image data or format")  # noqa: TRY004
    if fmt == "raw":
        return _decode_raw_image(name, bytes(data), node.get("shape"))
    if fmt in ("jpeg", "png"):
        return _decompress_image(bytes(data))
    raise ValueError(f"sensor {name!r} has unsupported image format {fmt!r}")


def _encode_extrinsic(name: str, matrix: np.ndarray) -> dict[str, Any]:
    """Encode one camera pose as an `__ndarray__` wrapper, shape checked here.

    A pose is 4x4 by construction and a caller that sends something else has a bug
    the far side cannot diagnose: every consumer slices `[:3, :3]` and `[:3, 3]`, so a
    wrong shape becomes a wrong rotation rather than an error.
    """
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (4, 4):
        raise ValueError(f"extrinsic {name!r} must be 4x4, got {list(array.shape)}")
    return pack_ndarray(array.reshape(-1), dtype="<f8", shape=[4, 4])


def _decode_extrinsic(name: str, node: Any) -> np.ndarray:
    """Decode one `__ndarray__` camera pose, shape checked as on the way out."""
    array = unpack_ndarray_full(node)
    if array is None:
        raise ValueError(f"extrinsic {name!r} is not a decodable ndarray wrapper")
    if array.shape != (4, 4):
        raise ValueError(f"extrinsic {name!r} decoded to {list(array.shape)}, expected 4x4")
    return np.array(array, dtype=np.float64)


def _decode_sensor(name: str, node: Any) -> np.ndarray:
    """Decode one sensor node back into an array, colour or depth.

    The inverse of `_encode_sensor`: an `__image__` wrapper is a colour frame, an
    `__ndarray__` wrapper is depth. Which marker is present decides, so neither
    side has to record a per-sensor kind.
    """
    if not isinstance(node, dict):
        # Malformed frame value, not a type misuse — see decode_observation.
        raise ValueError(f"sensor {name!r} is not an encoded sensor")  # noqa: TRY004
    if node.get("__ndarray__") is True:
        return _decode_depth(name, node)
    return _decode_colour(name, node)


def _decode_raw_image(name: str, data: bytes, shape: Any) -> np.ndarray:
    """Decode a raw (uint8 bytes) image wrapper, validating its shape and length."""
    if not isinstance(shape, list) or not all(isinstance(dim, int) for dim in shape):
        raise ValueError(f"sensor {name!r} is raw but is missing a valid shape")
    expected = math.prod(shape)
    if len(data) != expected:
        raise ValueError(
            f"sensor {name!r} has {len(data)} raw bytes but shape {shape} needs {expected}"
        )
    # `frombuffer` over `bytes` yields a read-only view; copy to a writable,
    # native-uint8 array so the public consumer can mutate it (see
    # _decode_state_array for why normalization lives in this layer).
    return np.array(np.frombuffer(data, dtype=np.uint8), dtype=np.uint8).reshape(shape)


def _decode_state_array(role: str, node: Any) -> np.ndarray:
    """Decode one `__ndarray__` state node into its full array, shape preserved.

    Uses the full decoder, not `unpack_ndarray`: state has no chunk semantics, so
    a 2-D state array must round-trip with every value and its shape intact.

    Normalization lives here, in the domain layer, on purpose: `unpack_ndarray_full`
    stays a zero-copy low-level primitive (read-only, wire-endian view), and this
    decoder owns the public guarantee that callers (torch, the observation
    adapters) get a writable, native-`float32` array. `np.array(..., dtype=...)`
    copies and converts endianness in one step.
    """
    array = unpack_ndarray_full(node)
    if array is None:
        raise ValueError(f"state {role!r} is not a decodable ndarray wrapper")
    return np.array(array, dtype=np.float32)


def _compress_image(array: np.ndarray, image_format: ImageFormat) -> bytes:
    """Compress an array to JPEG or PNG bytes."""
    buffer = io.BytesIO()
    Image.fromarray(array.astype(np.uint8)).save(buffer, format=image_format.upper())
    return buffer.getvalue()


def _decompress_image(data: bytes) -> np.ndarray:
    """Decompress JPEG or PNG bytes back into a writable array."""
    # A PIL-backed asarray can be read-only; np.array copies to a writable array.
    return np.array(Image.open(io.BytesIO(data)), dtype=np.uint8)


class FrameType(StrEnum):
    """The kinds of frame the two sides of the bridge exchange.

    The benchmark opens with HELLO to advertise its contract; the policy replies
    READY once it has confirmed the pairing and built any bridging pipeline. The
    benchmark then drives the loop: RESET at each episode start, OBSERVATION every
    step, and an ACTION comes back for each. Either side sends BYE for a clean
    shutdown.
    """

    HELLO = "hello"  # benchmark advertises its contract on connect.
    READY = "ready"  # policy confirms the pairing; benchmark may start streaming.
    OBSERVATION = "observation"  # benchmark -> policy, one per step.
    ACTION = "action"  # policy -> benchmark, one per observation.
    RESET = "reset"  # benchmark -> policy, at each episode start.
    BYE = "bye"  # either side, clean shutdown.


def encode_observation(
    observation: Observation,
    *,
    image_format: ImageFormat = "raw",
    action_prefix: np.ndarray | None = None,
    timestep: int | None = None,
) -> dict[str, Any]:
    """Encode an Observation into a frame payload.

    `state` arrays are encoded as `__ndarray__` wrappers keyed by role, `sensors`
    arrays keyed by sensor name, and the instruction passes through unchanged.
    `image_format` selects the encoding for *colour* sensors: "raw" (the default)
    stores the uint8 image bytes losslessly, "jpeg" and "png" compress them.

    A floating-point sensor is depth in metres and is encoded as an `__ndarray__`
    wrapper at its own dtype, losslessly, whatever `image_format` says (ADR 0007).
    Any other dtype, and any float array shaped like colour, is refused rather
    than sent as the other kind.

    `extrinsics` are 4x4 camera poses keyed by camera name, each an `__ndarray__`
    wrapper at float64 — a pose is geometry rather than network input, so
    it keeps full precision (ADR 0008). The key is omitted entirely when the
    observation carries none.

    `action_prefix` and `timestep` are forward-looking RTC fields (see the module
    docstring). The current synchronous loop passes neither, so the payload omits
    them and a peer decodes them as None. When given, `action_prefix` is the tail
    of already-committed actions an async policy inpaints against, carried as an
    `__ndarray__` wrapper, and `timestep` is the monotonic correlation index.

    Raises:
        ValueError: If `image_format` is not a recognised format, or a sensor
            array is neither a uint8 colour frame nor a float depth frame.
    """
    state = {
        role: pack_ndarray(array.reshape(-1).tolist(), shape=list(array.shape))
        for role, array in observation.state.items()
    }
    # One encode per sensor, then routed by the wrapper it produced: `_encode_sensor`
    # stays the single place that validates a sensor and picks its wrapper, and this
    # only decides which key carries the result.
    encoded = {
        name: _encode_sensor(array, image_format, name=name)
        for name, array in observation.sensors.items()
    }
    sensors = {name: node for name, node in encoded.items() if "__image__" in node}
    depth = {name: node for name, node in encoded.items() if "__ndarray__" in node}
    payload: dict[str, Any] = {
        "state": state,
        "sensors": sensors,
        "instruction": observation.instruction,
    }
    # Omitted entirely for an observation without depth, so a colour-only
    # benchmark's payload is byte-identical to the one it sent before version 4.
    if depth:
        payload["depth"] = depth
    if observation.extrinsics:
        payload["extrinsics"] = {
            name: _encode_extrinsic(name, matrix) for name, matrix in observation.extrinsics.items()
        }
    _attach_rtc_fields(payload, action_prefix=action_prefix, timestep=timestep)
    return payload


def decode_observation(payload: dict[str, Any]) -> Observation:
    """Decode a frame payload back into an Observation.

    The inverse of `encode_observation`; parses an untrusted frame, so every
    field is validated structurally.

    Raises:
        ValueError: If a required field is missing or any array or image is
            malformed.
    """
    # A malformed frame value is a ValueError by contract, not a TypeError (see the
    # module docstring), so the isinstance guards here suppress TRY004.
    if not isinstance(payload, dict):
        raise ValueError(  # noqa: TRY004
            f"observation payload must be a dict, got {type(payload).__name__}"
        )

    raw_state = payload.get("state", {})
    if not isinstance(raw_state, dict):
        raise ValueError("observation 'state' must be a dict")  # noqa: TRY004
    state = {role: _decode_state_array(role, node) for role, node in raw_state.items()}

    raw_sensors = payload.get("sensors", {})
    if not isinstance(raw_sensors, dict):
        raise ValueError("observation 'sensors' must be a dict")  # noqa: TRY004
    sensors = {name: _decode_sensor(name, node) for name, node in raw_sensors.items()}

    # Depth rejoins `sensors` here: it is a camera to everything above the wire (ADR
    # 0007), and only the payload keeps the two apart. `_decode_sensor` still reads an
    # `__ndarray__` node under `sensors`, so a version-2 or -3 peer's payload — which
    # put depth there — decodes unchanged.
    raw_depth = payload.get("depth", {})
    if not isinstance(raw_depth, dict):
        raise ValueError("observation 'depth' must be a dict")  # noqa: TRY004
    sensors.update({name: _decode_depth(name, node) for name, node in raw_depth.items()})

    raw_extrinsics = payload.get("extrinsics", {})
    if not isinstance(raw_extrinsics, dict):
        raise ValueError("observation 'extrinsics' must be a dict")  # noqa: TRY004
    extrinsics = {name: _decode_extrinsic(name, node) for name, node in raw_extrinsics.items()}

    return Observation(
        state=state,
        sensors=sensors,
        extrinsics=extrinsics,
        instruction=payload.get("instruction"),
    )


def encode_action(
    action: Action,
    *,
    action_prefix: np.ndarray | None = None,
    timestep: int | None = None,
) -> dict[str, Any]:
    """Encode an Action into a frame payload, preserving its shape.

    The values are flattened onto the wire but the declared shape is carried
    alongside, so a chunked (chunk, dim) action keeps its layout. See
    `decode_action` for how the receiving side consumes a chunk.

    `action_prefix` and `timestep` are the same forward-looking RTC fields
    `encode_observation` carries (see the module docstring): the current
    synchronous loop passes neither, so the payload omits them.
    """
    wrapper = pack_ndarray(action.values.reshape(-1).tolist(), shape=list(action.values.shape))
    payload: dict[str, Any] = {"action": wrapper}
    _attach_rtc_fields(payload, action_prefix=action_prefix, timestep=timestep)
    return payload


def decode_action(payload: dict[str, Any]) -> Action:
    """Decode a frame payload back into an Action.

    A chunked (chunk, dim) action decodes to its *first* step only: per
    `wire.unpack_ndarray`, actions are sent chunked but consumed one step at a
    time.

    Raises:
        ValueError: If the action field is missing or undecodable.
    """
    if not isinstance(payload, dict) or "action" not in payload:
        raise ValueError("action payload missing 'action'")
    values = unpack_ndarray(payload["action"])
    if values is None:
        raise ValueError("action 'action' is not a decodable ndarray wrapper")
    return Action.from_array(values)


def pack_stream_frame(
    frame_type: FrameType | str,
    payload: dict[str, Any],
    *,
    seq: int,
    timestamp: float | None = None,
    lane: int = DEFAULT_FRAME_LANE,
) -> bytes:
    """Frame one message for a byte stream: a length prefix, then the frame.

    The body is a `wire.pack_frame` msgpack frame; it is prefixed with its length
    as a 4-byte big-endian unsigned integer so a reader can recover the boundary
    from a stream that does not preserve message edges.

    `lane` is the per-frame correlation id (see `wire.pack_frame`): forward-
    looking, left at the default by the current single-lane loop.
    """
    body = pack_frame(str(frame_type), payload, seq=seq, timestamp=timestamp, lane=lane)
    return struct.pack(_LENGTH_PREFIX, len(body)) + body


def read_stream_frame(
    read_exactly: Callable[[int], bytes | None],
    *,
    max_frame_bytes: int = MAX_FRAME_BYTES,
) -> dict[str, Any] | None:
    """Read one length-prefixed frame from a byte stream.

    The inverse of `pack_stream_frame`. `read_exactly(n)` must return exactly `n`
    bytes, or None at end of stream or on a closed connection; this signature is
    what keeps the SDK independent of any specific transport. A None — or a short read
    — at either the length prefix or the body yields a None frame.

    The declared length is checked against `max_frame_bytes` before the body is
    read, so an untrusted peer cannot demand an unbounded allocation.

    Raises:
        ValueError: If the declared frame length exceeds `max_frame_bytes`.
    """
    header = read_exactly(_LENGTH_PREFIX_BYTES)
    if header is None or len(header) != _LENGTH_PREFIX_BYTES:
        return None
    (length,) = struct.unpack(_LENGTH_PREFIX, header)
    if length > max_frame_bytes:
        raise ValueError(f"frame length {length} exceeds max_frame_bytes {max_frame_bytes}")
    body = read_exactly(length)
    if body is None:
        return None
    return unpack_frame(body)


class FrameChannel:
    """A framed connection over a byte stream: it owns the seq and the read loop.

    A policy server, a benchmark runner, or a proxy speaks the bridge protocol by
    exchanging length-prefixed frames. Without this wrapper each side hand-rolls
    the same recv-exactly loop and tracks its own outgoing sequence number; this
    folds both into one object so callers just `send` and `recv`.

    The channel is transport-agnostic: it is built from a `read_exactly`/`write`
    pair, so the same code drives a socket, a network stream, an in-memory pipe, or a
    test buffer. The transport itself is out of scope here — `FrameChannel` frames
    over whatever byte stream those callables back onto. `from_socket` is the one
    socket-aware convenience; everything else stays independent of any transport.
    """

    def __init__(
        self,
        read_exactly: Callable[[int], bytes | None],
        write: Callable[[bytes], None],
        *,
        max_frame_bytes: int = MAX_FRAME_BYTES,
    ) -> None:
        """Build a channel over a read/write pair (the `read_stream_frame` contract)."""
        self._read_exactly = read_exactly
        self._write = write
        self._max_frame_bytes = max_frame_bytes
        self._seq = 0

    @classmethod
    def from_socket(
        cls, sock: socket.socket, *, max_frame_bytes: int = MAX_FRAME_BYTES
    ) -> FrameChannel:
        """Build a channel from a connected socket.

        Derives the `read_exactly`/`write` pair from the socket: a recv-exactly
        loop that accumulates until `n` bytes arrive (returning None if the peer
        closes mid-frame), and `sock.sendall` as the write. This is the only
        socket-aware code in the module.
        """

        def read_exactly(n: int) -> bytes | None:
            chunks: list[bytes] = []
            remaining = n
            while remaining > 0:
                chunk = sock.recv(remaining)
                if not chunk:  # peer closed the connection
                    return None
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)

        return cls(read_exactly, sock.sendall, max_frame_bytes=max_frame_bytes)

    def send(
        self,
        frame_type: FrameType | str,
        payload: dict[str, Any],
        *,
        lane: int = DEFAULT_FRAME_LANE,
    ) -> None:
        """Frame and write one message, auto-incrementing the outgoing seq.

        The channel owns a monotonically increasing sequence number (starting at
        0), so callers never pass one. `lane` is the per-frame correlation id
        (see `wire.pack_frame`): forward-looking, left at the default by the
        current single-lane loop.
        """
        frame = pack_stream_frame(frame_type, payload, seq=self._seq, lane=lane)
        self._seq += 1
        self._write(frame)

    def recv(self) -> dict[str, Any] | None:
        """Read one frame from the stream, or None at end of stream.

        Delegates to `read_stream_frame` with the channel's `max_frame_bytes`.

        Raises:
            ValueError: If the declared frame length exceeds `max_frame_bytes`.
        """
        return read_stream_frame(self._read_exactly, max_frame_bytes=self._max_frame_bytes)


def decode_rtc_fields(payload: dict[str, Any]) -> tuple[np.ndarray | None, int | None]:
    """Read the optional RTC fields from a frame payload as `(action_prefix, timestep)`.

    The inverse of `_attach_rtc_fields`. A payload from the current synchronous
    loop carries neither field, so this returns `(None, None)` — the reserved
    fields have defaults. `action_prefix` decodes to its full N-D array (no
    chunk truncation); `timestep` to an int. Because it parses an untrusted
    frame, a present-but-malformed field is a `ValueError`.

    Raises:
        ValueError: If a present `action_prefix` is not a decodable ndarray
            wrapper, or a present `timestep` is not an integer.
    """
    if not isinstance(payload, dict):
        raise ValueError(  # noqa: TRY004
            f"payload must be a dict, got {type(payload).__name__}"
        )
    action_prefix: np.ndarray | None = None
    if "action_prefix" in payload:
        action_prefix = unpack_ndarray_full(payload["action_prefix"])
        if action_prefix is None:
            raise ValueError("'action_prefix' is not a decodable ndarray wrapper")
    timestep_raw = payload.get("timestep")
    # `bool` is an `int` subclass, so reject it explicitly: a correlation index is
    # an integer count, never a flag, and silently treating True as 1 would mask a
    # malformed frame.
    if timestep_raw is not None and (
        not isinstance(timestep_raw, int) or isinstance(timestep_raw, bool)
    ):
        raise ValueError(f"'timestep' must be an int, got {type(timestep_raw).__name__}")
    return action_prefix, timestep_raw


__all__ = [
    "BRIDGE_PROTOCOL_VERSION",
    "MAX_FRAME_BYTES",
    "FrameChannel",
    "FrameType",
    "decode_action",
    "decode_observation",
    "decode_rtc_fields",
    "encode_action",
    "encode_observation",
    "pack_stream_frame",
    "read_stream_frame",
]
