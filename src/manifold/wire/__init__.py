"""The public bridge/wire protocol: the data-plane serialization boundary.

`codec.py` is the low-level msgpack frame codec; `bridge.py` sits on top of it
and maps the SDK's domain types (`Observation`, `Action`) onto the wire, defines
the frame types the two sides exchange, and handles length-prefixed stream
framing. A benchmark runner and a policy server both speak this protocol, so it
lives in the SDK in one place.
"""

from __future__ import annotations

from manifold.wire.bridge import (
    BRIDGE_PROTOCOL_VERSION,
    MAX_FRAME_BYTES,
    FrameChannel,
    FrameType,
    decode_action,
    decode_observation,
    decode_rtc_fields,
    encode_action,
    encode_observation,
    pack_stream_frame,
    read_stream_frame,
)
from manifold.wire.codec import (
    DEFAULT_FRAME_LANE,
    ImageFormat,
    find_encoded_image,
    pack_encoded_image,
    pack_frame,
    pack_ndarray,
    unpack_frame,
    unpack_ndarray,
    unpack_ndarray_full,
)

__all__ = [
    "BRIDGE_PROTOCOL_VERSION",
    "DEFAULT_FRAME_LANE",
    "MAX_FRAME_BYTES",
    "FrameChannel",
    "FrameType",
    "ImageFormat",
    "decode_action",
    "decode_observation",
    "decode_rtc_fields",
    "encode_action",
    "encode_observation",
    "find_encoded_image",
    "pack_encoded_image",
    "pack_frame",
    "pack_ndarray",
    "pack_stream_frame",
    "read_stream_frame",
    "unpack_frame",
    "unpack_ndarray",
    "unpack_ndarray_full",
]
