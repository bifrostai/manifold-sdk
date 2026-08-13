"""Wire codec for the bench-policy bridge protocol.

A benchmark runner and a policy server encode the same msgpack frame, so the
codec lives in the SDK, in one place. A frame is:

    {"type": str, "payload": dict, "seq": int, "timestamp": float, "lane": int}

`lane` is the per-frame correlation id for one of several concurrent rollouts
*within one connection*. The current synchronous loop runs a single lane per
connection and never sets it (it defaults to `DEFAULT_FRAME_LANE`); it is
reserved, additively, for a future vectorized serving path that multiplexes many
lanes over one connection and must route each frame to the right per-rollout
state. A frame written without it decodes with
the default, so the field is backward-compatible.

Inside `payload`, numpy arrays and images are encoded as marked dicts:

    {"__ndarray__": True, "data": <bytes>, "dtype": "<f4", "shape": [...]}
    {"__image__": True, "data": <bytes>, "format": "jpeg"|"png"|"raw", "shape": [...]}
"""

from __future__ import annotations

import math
import time
from typing import Any, Literal, cast

import msgpack
import numpy as np

# The image encodings carried in an `__image__` wrapper's `format` field.
ImageFormat = Literal["raw", "jpeg", "png"]

# The little-endian dtypes accepted off the wire. This is the gate: a dtype
# string from an untrusted frame decodes only if it is one of these.
_WIRE_DTYPES: dict[str, np.dtype[Any]] = {
    "<f2": np.dtype("<f2"),
    "<f4": np.dtype("<f4"),
    "<f8": np.dtype("<f8"),
    "<i4": np.dtype("<i4"),
    "<i8": np.dtype("<i8"),
    "<u4": np.dtype("<u4"),
}

# The single implicit lane every frame on the current synchronous loop carries.
# A future multi-lane serving path overrides it per frame; until then it is a
# constant, so reading code can treat its absence and this value identically.
DEFAULT_FRAME_LANE = 0


def _decoded_ndarray(value: Any) -> tuple[np.ndarray, list[int]] | None:
    """Validate a `__ndarray__` wrapper and return its flat array and declared shape.

    The shared validation prologue of `unpack_ndarray` and `unpack_ndarray_full`:
    checks the wrapper, the field types, and that the dtype is one accepted off the
    wire with a byte length that is a whole number of elements. The caller applies the
    shape-product check (the two differ on the empty-array case). Returns None on any
    malformed input, since this parses an untrusted frame.
    """
    if not isinstance(value, dict) or value.get("__ndarray__") is not True:
        return None
    raw = value.get("data")
    dtype_str = value.get("dtype")
    shape = value.get("shape")
    if not isinstance(raw, (bytes, bytearray)):
        return None
    if not isinstance(dtype_str, str):
        return None
    if not isinstance(shape, list) or not all(isinstance(dim, int) for dim in shape):
        return None
    dtype = _WIRE_DTYPES.get(dtype_str)
    if dtype is None:
        return None
    if len(raw) % dtype.itemsize != 0:
        return None
    # .copy() so the decoded array is writable: np.frombuffer returns a read-only view
    # over the temporary bytes, and a caller doing in-place work on it would otherwise hit
    # "assignment destination is read-only".
    return np.frombuffer(bytes(raw), dtype=dtype).copy(), shape


def _as_encoded_image(node: dict[str, Any]) -> tuple[bytes, str] | None:
    """Return (bytes, format) if `node` is a valid `__image__` wrapper, else None."""
    if node.get("__image__") is not True:
        return None
    data = node.get("data")
    fmt = node.get("format")
    if isinstance(data, (bytes, bytearray)) and isinstance(fmt, str):
        return bytes(data), fmt
    return None


def pack_frame(
    type_: str,
    payload: dict[str, Any],
    *,
    seq: int,
    timestamp: float | None = None,
    lane: int = DEFAULT_FRAME_LANE,
) -> bytes:
    """Encode one bridge frame as msgpack bytes.

    `lane` is the per-frame correlation id for one rollout within one connection.
    It is forward-looking: the current single-lane loop always leaves it at
    `DEFAULT_FRAME_LANE`, and a frame decoded without it falls back to that
    default, so the field is additive and does not change today's behaviour.
    """
    frame = {
        "type": type_,
        "payload": payload,
        "seq": seq,
        "timestamp": time.time() if timestamp is None else timestamp,
        "lane": lane,
    }
    # `msgpack` is untyped; packb returns bytes here. The cast says so to type
    # checkers that infer its return as `Unknown | None`.
    return cast(bytes, msgpack.packb(frame, use_bin_type=True))


def unpack_frame(buf: bytes) -> dict[str, Any] | None:
    """Decode a msgpack frame. Return None on empty input or any decode failure.

    A frame produced by an older peer (or the current single-lane loop) carries
    no explicit `lane`; it is filled with `DEFAULT_FRAME_LANE` here so every
    decoded frame presents the field uniformly and reading code never special-
    cases its absence.
    """
    if not buf:
        return None
    try:
        decoded = msgpack.unpackb(buf, raw=False)
    except Exception:
        return None
    if not isinstance(decoded, dict):
        return None
    decoded.setdefault("lane", DEFAULT_FRAME_LANE)
    return decoded


def pack_ndarray(
    values: np.ndarray | list[float] | tuple[float, ...],
    *,
    dtype: str = "<f4",
    shape: list[int] | None = None,
) -> dict[str, Any]:
    """Encode a flat sequence or array as a `__ndarray__` wrapper.

    An array is accepted alongside a list so a large payload never round-trips
    through Python floats. It must already be flat: `shape` declares the layout
    the reader restores.
    """
    np_dtype = _WIRE_DTYPES.get(dtype)
    if np_dtype is None:
        raise ValueError(f"unsupported dtype {dtype!r}")
    if shape is None:
        shape = [len(values)]
    data = np.asarray(values, dtype=np_dtype).tobytes()
    return {"__ndarray__": True, "data": data, "dtype": dtype, "shape": shape}


def unpack_ndarray(value: Any) -> list[float] | None:
    """Decode a `__ndarray__` wrapper into a flat list of floats.

    For a 2D array of shape (chunk_size, action_dim) only the first step is
    returned: actions are sent chunked on the wire, but one step is consumed at a
    time. Return None on any malformed input (see `_decoded_ndarray`).
    """
    decoded = _decoded_ndarray(value)
    if decoded is None:
        return None
    flat, shape = decoded
    if flat.size == 0:
        return []
    if shape and flat.size != math.prod(shape):
        return None
    # For a chunked action (chunk_size, ...), return one step: the product of the
    # trailing dimensions.
    if len(shape) >= 2:
        flat = flat[: math.prod(shape[1:])]
    return [float(v) for v in flat]


def unpack_ndarray_full(value: Any) -> np.ndarray | None:
    """Decode a `__ndarray__` wrapper into the complete array, reshaped to `shape`.

    The true inverse of `pack_ndarray`: unlike `unpack_ndarray`, this applies no
    action-chunk truncation and preserves the full N-D shape, so a state array
    round-trips losslessly. Return None on any malformed input (see `_decoded_ndarray`).
    """
    decoded = _decoded_ndarray(value)
    if decoded is None:
        return None
    flat, shape = decoded
    if shape and flat.size != math.prod(shape):
        return None
    return flat.reshape(shape) if shape else flat


def find_encoded_image(value: Any) -> tuple[bytes, str] | None:
    """Find the first `__image__`-marked dict and return (bytes, format).

    Benchmark payloads nest the image dict under benchmark-defined keys, so the
    search descends into nested dicts. Return None if none is found.
    """
    if not isinstance(value, dict):
        return None
    found = _as_encoded_image(value)
    if found is not None:
        return found
    for child in value.values():
        found = find_encoded_image(child)
        if found is not None:
            return found
    return None


def pack_encoded_image(
    data: bytes,
    *,
    format_: ImageFormat,
    shape: list[int] | None = None,
) -> dict[str, Any]:
    """Build an `__image__` wrapper dict, the inverse of `find_encoded_image`."""
    out: dict[str, Any] = {"__image__": True, "format": format_, "data": data}
    if shape is not None:
        out["shape"] = shape
    return out


__all__ = [
    "DEFAULT_FRAME_LANE",
    "ImageFormat",
    "find_encoded_image",
    "pack_encoded_image",
    "pack_frame",
    "pack_ndarray",
    "unpack_frame",
    "unpack_ndarray",
    "unpack_ndarray_full",
]
