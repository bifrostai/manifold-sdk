"""The request/response half of the bridge codec: one frame per body.

`pack_stream_frame` and `read_stream_frame` are covered in `test_bridge.py`.
What is asserted here is that the HTTP twin carries the same frames, and that
a chunk's entries stay encoded on the way through.
"""

from __future__ import annotations

import numpy as np
import pytest

from manifold.core.values import Action
from manifold.wire import FrameType, bridge


def test_a_payload_survives_one_request_body():
    body = bridge.pack_http_frame(FrameType.OBSERVATION, {"step": 3})
    assert bridge.read_http_frame(body, FrameType.OBSERVATION) == {"step": 3}


def test_a_body_carrying_another_frame_is_refused_rather_than_dispatched_on():
    body = bridge.pack_http_frame(FrameType.ACTION, {"actions": []})
    with pytest.raises(ValueError, match="observation frame"):
        bridge.read_http_frame(body, FrameType.OBSERVATION)


def test_an_undecodable_body_is_refused():
    with pytest.raises(ValueError, match="does not decode"):
        bridge.read_http_frame(b"not a frame", FrameType.ACTION)


def test_a_chunk_entry_comes_back_as_the_action_frame_payload_it_went_in_as():
    # The caller sends an entry on as the ACTION frame of one step, so the bytes
    # a benchmark receives are the ones the one-step-at-a-time path produces.
    actions = [Action(values=np.arange(7, dtype=np.float32) + step) for step in range(3)]
    encoded = [bridge.encode_action(action) for action in actions]

    read = bridge.read_action_chunk(bridge.encode_action_chunk(actions))

    assert read == encoded
    assert bridge.decode_action(read[1]).values.tolist() == actions[1].values.tolist()


def test_a_payload_without_an_actions_list_is_refused():
    with pytest.raises(ValueError, match="missing 'actions'"):
        bridge.read_action_chunk({"steps": []})


def test_an_entry_that_is_not_an_action_payload_is_refused_rather_than_dropped():
    # Dropping it would shorten the chunk, and the caller would then refill its
    # queue at a different cadence from the one-step path.
    with pytest.raises(ValueError, match=r"entries \[1\]"):
        bridge.read_action_chunk({"actions": [{"values": []}, "not a payload"]})
