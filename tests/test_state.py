from manifold.core.state import DEFAULT_LANE, PipelineState


def test_slice_for_returns_a_persistent_per_lane_key_bag() -> None:
    state = PipelineState()

    bag = state.slice_for(lane=DEFAULT_LANE, key="frame_history")
    bag["agentview"] = [1, 2, 3]

    # The same (lane, key) hands back the same mutable object, so an adapter's
    # writes persist across steps.
    again = state.slice_for(lane=DEFAULT_LANE, key="frame_history")
    assert again is bag
    assert again["agentview"] == [1, 2, 3]


def test_slices_are_isolated_by_key_and_by_lane() -> None:
    state = PipelineState()

    same_lane_other_key = state.slice_for(lane=DEFAULT_LANE, key="other")
    other_lane_same_key = state.slice_for(lane=1, key="frame_history")
    target = state.slice_for(lane=DEFAULT_LANE, key="frame_history")

    assert same_lane_other_key is not target
    assert other_lane_same_key is not target


def test_reset_lane_drops_only_that_lanes_slices() -> None:
    state = PipelineState()
    lane0 = state.slice_for(lane=DEFAULT_LANE, key="frame_history")
    lane0["agentview"] = [1]
    lane1 = state.slice_for(lane=1, key="frame_history")
    lane1["agentview"] = [2]

    state.reset_lane(DEFAULT_LANE)

    # Lane 0 is cleared: a fresh slice, not the old populated one.
    fresh = state.slice_for(lane=DEFAULT_LANE, key="frame_history")
    assert fresh == {}
    assert fresh is not lane0
    # Lane 1 is untouched: same object, same contents.
    assert state.slice_for(lane=1, key="frame_history") is lane1
    assert lane1["agentview"] == [2]
