import numpy as np

from manifold.wire import codec as wire


def test_ndarray_round_trips() -> None:
    packed = wire.pack_ndarray([0.5, 1.5, 2.5])
    assert wire.unpack_ndarray(packed) == [0.5, 1.5, 2.5]


def test_unpack_returns_only_the_first_step_of_a_chunk() -> None:
    packed = wire.pack_ndarray([1, 2, 3, 4, 5, 6, 7, 8], shape=[2, 4])
    assert wire.unpack_ndarray(packed) == [1.0, 2.0, 3.0, 4.0]


def test_unpack_rejects_an_unknown_dtype() -> None:
    assert (
        wire.unpack_ndarray({"__ndarray__": True, "data": b"xx", "dtype": "bad", "shape": [1]})
        is None
    )


def test_unpack_rejects_a_buffer_that_does_not_match_its_shape() -> None:
    short = wire.pack_ndarray([1, 2, 3])
    short["shape"] = [3, 4]  # claims 12 elements; only 3 are present
    assert wire.unpack_ndarray(short) is None


def test_frame_round_trips() -> None:
    frame = wire.unpack_frame(wire.pack_frame("obs", {"a": 1}, seq=7))
    assert frame is not None
    assert frame["type"] == "obs"
    assert frame["seq"] == 7


def test_frame_carries_lane_when_set() -> None:
    frame = wire.unpack_frame(wire.pack_frame("obs", {}, seq=0, lane=3))
    assert frame is not None
    assert frame["lane"] == 3


def test_frame_defaults_lane_when_unset() -> None:
    # The current single-lane loop never passes a lane; it round-trips as the
    # default so reading code never special-cases the field's absence.
    frame = wire.unpack_frame(wire.pack_frame("obs", {}, seq=0))
    assert frame is not None
    assert frame["lane"] == wire.DEFAULT_FRAME_LANE


def test_frame_from_an_older_peer_decodes_with_default_lane() -> None:
    # A frame packed by a peer that predates the `lane` field (a bare dict with no
    # "lane" key) still decodes, with the default filled in.
    import msgpack

    legacy = msgpack.packb(
        {"type": "obs", "payload": {}, "seq": 0, "timestamp": 0.0}, use_bin_type=True
    )
    frame = wire.unpack_frame(legacy)
    assert frame is not None
    assert frame["lane"] == wire.DEFAULT_FRAME_LANE


def test_find_encoded_image_descends_into_nested_payloads() -> None:
    image = wire.pack_encoded_image(b"JPEGDATA", format_="jpeg")
    assert wire.find_encoded_image({"images": {"agentview": image}}) == (b"JPEGDATA", "jpeg")


def test_unpack_ndarray_full_preserves_shape_and_all_values() -> None:
    packed = wire.pack_ndarray([1, 2, 3, 4, 5, 6], shape=[2, 3])
    full = wire.unpack_ndarray_full(packed)
    assert full is not None
    assert full.shape == (2, 3)
    assert np.array_equal(full, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))


def test_unpack_ndarray_full_rejects_a_buffer_that_does_not_match_its_shape() -> None:
    short = wire.pack_ndarray([1, 2, 3])
    short["shape"] = [3, 4]  # claims 12 elements; only 3 are present
    assert wire.unpack_ndarray_full(short) is None


def test_empty_array_divergence_between_unpack_ndarray_and_unpack_ndarray_full() -> None:
    # unpack_ndarray short-circuits on empty data and returns [] without applying
    # the shape-product check; unpack_ndarray_full applies the check (0 == prod([0,
    # 3])) and reshapes, preserving the declared shape in the returned array.
    empty = wire.pack_ndarray([], shape=[0, 3])
    assert wire.unpack_ndarray(empty) == []
    full = wire.unpack_ndarray_full(empty)
    assert full is not None
    assert full.shape == (0, 3)
