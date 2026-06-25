import contextlib
import io
import socket
import struct

import numpy as np
import pytest

from manifold.core.values import Action, Observation
from manifold.wire import (
    DEFAULT_FRAME_LANE,
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
from manifold.wire import codec as wire


def _reader(data: bytes):
    """A read_exactly closure over a buffer: returns exactly n bytes or None."""
    buffer = io.BytesIO(data)

    def read_exactly(n: int) -> bytes | None:
        chunk = buffer.read(n)
        return chunk if len(chunk) == n else None

    return read_exactly


def _sample_observation() -> Observation:
    return Observation(
        state={"ee_pose": np.arange(6, dtype=np.float32)},
        sensors={
            "agentview": np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3),
            "wrist": (np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3) % 7).astype(np.uint8),
        },
        instruction="pick up the cube",
    )


def test_observation_round_trips_raw() -> None:
    obs = _sample_observation()
    decoded = decode_observation(encode_observation(obs, image_format="raw"))

    assert np.allclose(decoded.state["ee_pose"], obs.state["ee_pose"])
    for name, array in obs.sensors.items():
        assert decoded.sensors[name].dtype == np.uint8
        assert decoded.sensors[name].shape == array.shape
        # Raw is lossless, so the bytes survive exactly.
        assert np.array_equal(decoded.sensors[name], array)
    assert decoded.instruction == "pick up the cube"


def test_decoded_arrays_are_writable_and_native_byte_order() -> None:
    # Covers raw (state + sensor) and compressed (png) paths in one shot:
    # any read-only view or wrong byte-order would raise or fail the dtype check.
    obs = _sample_observation()
    decoded_raw = decode_observation(encode_observation(obs, image_format="raw"))

    state = decoded_raw.state["ee_pose"]
    assert state.flags.writeable is True
    assert state.dtype == np.float32
    assert state.dtype.byteorder in ("=", "|")  # native (or not-applicable)
    state[0] = 99.0  # a read-only view would raise here

    sensor_raw = decoded_raw.sensors["agentview"]
    assert sensor_raw.flags.writeable is True
    assert sensor_raw.dtype == np.uint8
    assert sensor_raw.dtype.byteorder in ("=", "|")
    sensor_raw[0, 0, 0] = 7

    # Compressed path: png-decoded sensor must also be writable.
    decoded_png = decode_observation(encode_observation(obs, image_format="png"))
    sensor_png = decoded_png.sensors["agentview"]
    assert sensor_png.flags.writeable is True
    assert sensor_png.dtype == np.uint8
    sensor_png[0, 0, 0] = 7


def test_observation_round_trips_png_losslessly() -> None:
    obs = _sample_observation()
    decoded = decode_observation(encode_observation(obs, image_format="png"))

    for name, array in obs.sensors.items():
        assert decoded.sensors[name].shape == array.shape
        # PNG is lossless, so the pixels survive exactly.
        assert np.array_equal(decoded.sensors[name], array)


def test_observation_jpeg_preserves_shape_and_dtype() -> None:
    obs = _sample_observation()
    decoded = decode_observation(encode_observation(obs, image_format="jpeg"))

    # JPEG is lossy: only shape and dtype are guaranteed, not exact pixels.
    for name, array in obs.sensors.items():
        assert decoded.sensors[name].shape == array.shape
        assert decoded.sensors[name].dtype == np.uint8


def test_unsupported_image_format_raises() -> None:
    obs = Observation(sensors={"cam": np.zeros((4, 4, 3), dtype=np.uint8)})
    with pytest.raises(ValueError, match="unsupported image_format"):
        # "webp" is intentionally an invalid format to exercise the runtime guard.
        encode_observation(obs, image_format="webp")  # ty: ignore[invalid-argument-type]


def test_observation_without_instruction_round_trips() -> None:
    obs = Observation(state={"ee_pose": np.zeros(3, dtype=np.float32)})
    decoded = decode_observation(encode_observation(obs))
    assert decoded.instruction is None


def test_action_round_trips_1d() -> None:
    action = Action.from_array(np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 1.0], dtype=np.float32))
    decoded = decode_action(encode_action(action))
    assert np.allclose(decoded.values, action.values)


def test_chunked_action_decodes_to_first_step() -> None:
    chunk = np.arange(3 * 7, dtype=np.float32).reshape(3, 7)
    decoded = decode_action(encode_action(Action(values=chunk)))
    # unpack_ndarray returns only the first step of a chunk.
    assert np.allclose(decoded.values, chunk[0])


def test_observation_without_rtc_fields_decodes_to_none() -> None:
    # The current synchronous loop passes neither field, so the payload omits them
    # and a peer reads them back as None — the reserved fields have defaults.
    payload = encode_observation(_sample_observation())
    assert "action_prefix" not in payload
    assert "timestep" not in payload
    assert decode_rtc_fields(payload) == (None, None)


def test_observation_rtc_fields_round_trip() -> None:
    prefix = np.arange(6, dtype=np.float32).reshape(2, 3)
    payload = encode_observation(_sample_observation(), action_prefix=prefix, timestep=7)
    decoded_prefix, decoded_timestep = decode_rtc_fields(payload)
    assert decoded_prefix is not None
    assert decoded_prefix.shape == (2, 3)
    assert np.array_equal(decoded_prefix, prefix)
    assert decoded_timestep == 7
    # The observation itself still round-trips alongside the reserved fields.
    assert decode_observation(payload).instruction == "pick up the cube"


def test_action_rtc_fields_round_trip() -> None:
    action = Action.from_array(np.array([0.1, -0.2, 0.3], dtype=np.float32))
    prefix = np.array([1.0, 2.0], dtype=np.float32)
    payload = encode_action(action, action_prefix=prefix, timestep=11)
    decoded_prefix, decoded_timestep = decode_rtc_fields(payload)
    assert decoded_prefix is not None
    assert np.array_equal(decoded_prefix, prefix)
    assert decoded_timestep == 11
    # The action still decodes to its first step alongside the reserved fields.
    assert np.allclose(decode_action(payload).values, [0.1, -0.2, 0.3])


def test_action_without_rtc_fields_decodes_to_none() -> None:
    payload = encode_action(Action.from_array(np.array([0.0], dtype=np.float32)))
    assert "action_prefix" not in payload
    assert "timestep" not in payload
    assert decode_rtc_fields(payload) == (None, None)


def test_decode_rtc_fields_rejects_a_malformed_action_prefix() -> None:
    payload = {"action_prefix": {"__ndarray__": True, "data": b"xx", "dtype": "bad", "shape": [1]}}
    with pytest.raises(ValueError, match="action_prefix"):
        decode_rtc_fields(payload)


def test_decode_rtc_fields_rejects_a_non_integer_timestep() -> None:
    with pytest.raises(ValueError, match="timestep"):
        decode_rtc_fields({"timestep": "not-an-int"})


def test_stream_frame_carries_lane() -> None:
    framed = pack_stream_frame(FrameType.OBSERVATION, {}, seq=0, lane=5)
    frame = read_stream_frame(_reader(framed))
    assert frame is not None
    assert frame["lane"] == 5


def test_stream_frame_defaults_lane_when_unset() -> None:
    framed = pack_stream_frame(FrameType.OBSERVATION, {}, seq=0)
    frame = read_stream_frame(_reader(framed))
    assert frame is not None
    assert frame["lane"] == DEFAULT_FRAME_LANE


def test_decode_observation_rejects_sensor_missing_data() -> None:
    payload = {
        "state": {},
        "sensors": {"cam": {"__image__": True, "format": "raw", "shape": [4, 4, 3]}},
        "instruction": None,
    }
    with pytest.raises(ValueError, match="missing its image data"):
        decode_observation(payload)


def test_decode_observation_rejects_raw_image_missing_shape() -> None:
    payload = {
        "state": {},
        "sensors": {"cam": {"__image__": True, "format": "raw", "data": b"\x00\x00\x00"}},
        "instruction": None,
    }
    with pytest.raises(ValueError, match="missing a valid shape"):
        decode_observation(payload)


def test_decode_observation_rejects_bad_state_array() -> None:
    payload = {"state": {"ee_pose": {"not": "an ndarray"}}, "sensors": {}, "instruction": None}
    with pytest.raises(ValueError, match="not a decodable ndarray"):
        decode_observation(payload)


def test_decode_action_rejects_missing_action() -> None:
    with pytest.raises(ValueError, match="missing 'action'"):
        decode_action({})


def test_decode_action_rejects_undecodable_action() -> None:
    with pytest.raises(ValueError, match="not a decodable ndarray"):
        decode_action(
            {"action": {"__ndarray__": True, "data": b"xx", "dtype": "bad", "shape": [1]}}
        )


def test_stream_frame_round_trips() -> None:
    obs = _sample_observation()
    framed = pack_stream_frame(
        FrameType.OBSERVATION, encode_observation(obs), seq=42, timestamp=1.0
    )

    buffer = io.BytesIO(framed)

    def read_exactly(n: int) -> bytes | None:
        data = buffer.read(n)
        return data if len(data) == n else None

    frame = read_stream_frame(read_exactly)
    assert frame is not None
    assert frame["type"] == "observation"
    assert frame["seq"] == 42
    decoded = decode_observation(frame["payload"])
    assert decoded.instruction == "pick up the cube"


def test_stream_frame_accepts_a_plain_string_type() -> None:
    framed = pack_stream_frame("reset", {}, seq=0)
    buffer = io.BytesIO(framed)

    def read_exactly(n: int) -> bytes | None:
        data = buffer.read(n)
        return data if len(data) == n else None

    frame = read_stream_frame(read_exactly)
    assert frame is not None
    assert frame["type"] == "reset"


def test_read_stream_frame_returns_none_on_header_eof() -> None:
    # Covers both EOF (no bytes) and truncated-body: any short read returns None.
    assert read_stream_frame(lambda _: None) is None


def test_read_stream_frame_returns_none_on_truncated_body() -> None:
    body = wire.pack_frame("bye", {}, seq=1)
    # A length prefix that promises the full body, but the stream ends short.
    framed = struct.pack(">I", len(body)) + body[:-3]
    assert read_stream_frame(_reader(framed)) is None


def test_read_stream_frame_rejects_oversized_frame() -> None:
    # A length prefix that declares more than the cap, before any body is read.
    framed = struct.pack(">I", MAX_FRAME_BYTES + 1)
    with pytest.raises(ValueError, match="exceeds max_frame_bytes"):
        read_stream_frame(_reader(framed))


def test_read_stream_frame_honours_custom_max_frame_bytes() -> None:
    framed = pack_stream_frame("bye", {}, seq=1)
    with pytest.raises(ValueError, match="exceeds max_frame_bytes"):
        read_stream_frame(_reader(framed), max_frame_bytes=1)


def test_state_array_round_trips_with_full_shape() -> None:
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    obs = Observation(state={"grid": array})
    decoded = decode_observation(encode_observation(obs))
    assert decoded.state["grid"].shape == (2, 3)
    assert np.array_equal(decoded.state["grid"], array)


def test_raw_encoding_rejects_non_uint8_sensor() -> None:
    obs = Observation(sensors={"depth": np.zeros((4, 4), dtype=np.float32)})
    with pytest.raises(ValueError, match="raw image encoding requires a uint8"):
        encode_observation(obs, image_format="raw")


def test_empty_instruction_round_trips() -> None:
    obs = Observation(instruction="")
    decoded = decode_observation(encode_observation(obs))
    assert decoded.instruction == ""


def test_decode_observation_rejects_raw_image_with_wrong_byte_length() -> None:
    # shape (4,4,3) needs 48 bytes; supply 3 so the length check fires, not numpy.
    payload = {
        "state": {},
        "sensors": {
            "cam": {"__image__": True, "format": "raw", "shape": [4, 4, 3], "data": b"abc"}
        },
        "instruction": None,
    }
    with pytest.raises(ValueError, match="raw bytes but shape"):
        decode_observation(payload)


def _pipe() -> tuple[FrameChannel, FrameChannel]:
    """A pair of channels over a shared in-memory buffer: writes feed reads."""
    buffer = io.BytesIO()

    def make(read_pos: list[int]) -> FrameChannel:
        def read_exactly(n: int) -> bytes | None:
            buffer.seek(read_pos[0])
            chunk = buffer.read(n)
            read_pos[0] = buffer.tell()
            return chunk if len(chunk) == n else None

        def write(data: bytes) -> None:
            buffer.seek(0, io.SEEK_END)
            buffer.write(data)

        return FrameChannel(read_exactly, write)

    # Both channels share the buffer; the reader tracks its own cursor.
    cursor = [0]
    sender = make(cursor)
    receiver = make(cursor)
    return sender, receiver


def test_frame_channel_round_trips_several_frames() -> None:
    sender, receiver = _pipe()
    obs = _sample_observation()

    sender.send(FrameType.HELLO, {"contract": "demo"})
    sender.send(FrameType.OBSERVATION, encode_observation(obs))
    sender.send(FrameType.ACTION, encode_action(Action.from_array([0.0, 1.0, 2.0])))

    hello = receiver.recv()
    assert hello is not None
    assert hello["type"] == "hello"
    assert hello["payload"]["contract"] == "demo"

    observation = receiver.recv()
    assert observation is not None
    assert observation["type"] == "observation"
    assert decode_observation(observation["payload"]).instruction == "pick up the cube"

    action = receiver.recv()
    assert action is not None
    assert action["type"] == "action"
    assert np.allclose(decode_action(action["payload"]).values, [0.0, 1.0, 2.0])


def test_frame_channel_send_increments_seq() -> None:
    sender, receiver = _pipe()

    sender.send(FrameType.RESET, {})
    sender.send(FrameType.RESET, {})
    sender.send(FrameType.RESET, {})

    seqs = []
    for _ in range(3):
        frame = receiver.recv()
        assert frame is not None
        seqs.append(frame["seq"])
    assert seqs == [0, 1, 2]


def test_frame_channel_send_carries_lane() -> None:
    sender, receiver = _pipe()

    sender.send(FrameType.OBSERVATION, {}, lane=2)
    sender.send(FrameType.OBSERVATION, {})  # default lane

    first = receiver.recv()
    second = receiver.recv()
    assert first is not None and first["lane"] == 2
    assert second is not None and second["lane"] == DEFAULT_FRAME_LANE


def test_frame_channel_recv_returns_none_on_eof() -> None:
    _, receiver = _pipe()  # nothing was ever written
    assert receiver.recv() is None


def test_frame_channel_from_socket_round_trips() -> None:
    left, right = socket.socketpair()
    try:
        sender = FrameChannel.from_socket(left)
        receiver = FrameChannel.from_socket(right)

        sender.send(FrameType.HELLO, {"contract": "demo"})
        sender.send(FrameType.OBSERVATION, encode_observation(_sample_observation()))

        hello = receiver.recv()
        assert hello is not None
        assert hello["type"] == "hello"
        assert hello["seq"] == 0

        observation = receiver.recv()
        assert observation is not None
        assert observation["seq"] == 1
        assert decode_observation(observation["payload"]).instruction == "pick up the cube"
    finally:
        left.close()
        right.close()


def test_frame_channel_from_socket_recv_returns_none_on_close() -> None:
    left, right = socket.socketpair()
    try:
        receiver = FrameChannel.from_socket(right)
        left.close()  # peer closes without sending: recv-exactly sees EOF
        assert receiver.recv() is None
    finally:
        right.close()


def test_two_frame_channels_are_isolated_over_separate_socketpairs() -> None:
    # Wire-level isolation: two independent FrameChannel pairs over separate
    # socketpairs must decode each other's frames correctly and without
    # cross-contamination — a frame written on channel A must not appear on
    # channel B, and vice versa.  Each channel gets its own socketpair so their
    # byte streams are entirely separate; the seq counters also start
    # independently, confirming there is no shared state.
    a_left, a_right = socket.socketpair()
    b_left, b_right = socket.socketpair()
    try:
        chan_a_send = FrameChannel.from_socket(a_left)
        chan_a_recv = FrameChannel.from_socket(a_right)
        chan_b_send = FrameChannel.from_socket(b_left)
        chan_b_recv = FrameChannel.from_socket(b_right)

        obs_a = _sample_observation()
        action_b = Action.from_array(np.array([1.0, 2.0, 3.0], dtype=np.float32))

        # Send an observation on A and an action on B.
        chan_a_send.send(FrameType.OBSERVATION, encode_observation(obs_a))
        chan_b_send.send(FrameType.ACTION, encode_action(action_b))

        # Close the sending sides so recv() sees EOF after the one frame.
        a_left.close()
        b_left.close()

        # Channel A receives exactly what was sent on A — an observation.
        frame_a = chan_a_recv.recv()
        assert frame_a is not None
        assert frame_a["type"] == "observation"
        assert frame_a["seq"] == 0
        decoded_obs = decode_observation(frame_a["payload"])
        assert decoded_obs.instruction == obs_a.instruction
        assert np.array_equal(decoded_obs.state["ee_pose"], obs_a.state["ee_pose"])

        # No second frame on A (the sender closed).
        assert chan_a_recv.recv() is None

        # Channel B receives exactly what was sent on B — an action.
        frame_b = chan_b_recv.recv()
        assert frame_b is not None
        assert frame_b["type"] == "action"
        assert frame_b["seq"] == 0  # independent seq counter, not A's
        decoded_act = decode_action(frame_b["payload"])
        assert np.allclose(decoded_act.values, action_b.values)

        # No second frame on B either.
        assert chan_b_recv.recv() is None
    finally:
        # a_left / b_left were already closed above; ignore if already gone.
        with contextlib.suppress(OSError):
            a_left.close()
        with contextlib.suppress(OSError):
            b_left.close()
        a_right.close()
        b_right.close()
