"""Tests for the Manifold events a replay step carries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from manifold.replay import (
    REPLAY_LOG_VERSION,
    Above,
    Behind,
    CenteredOn,
    Channel,
    ChannelKind,
    Closed,
    EpisodeEnded,
    EpisodeStarted,
    Grasped,
    GripperAction,
    GripperCommanded,
    GripperFullyClosed,
    GripperHit,
    GripperTouching,
    InFrontOf,
    Inside,
    LeftOf,
    ManifoldEventType,
    Note,
    ObjectDisplaced,
    ObjectLeftScene,
    ObjectTipped,
    On,
    Open,
    Outside,
    Pose,
    ReplayFrameType,
    ReplayLogWriter,
    RightOf,
    Severity,
    Stacked,
    StackOrder,
    Touching,
    TurnedOn,
    Upright,
    read_replay_log,
)
from manifold.wire.bridge import pack_stream_frame, read_stream_frame

if TYPE_CHECKING:
    from pathlib import Path

CHANNELS = (Channel(name="reward", kind=ChannelKind.SCALAR),)

EVERY_TYPE = (
    EpisodeStarted(
        instruction="put the bowl on the plate",
        attributes={"category": "camera", "difficulty": "3"},
    ),
    *(
        cls(object="bowl", is_active=True)
        for cls in (Grasped, GripperTouching, Upright, Open, Closed, TurnedOn)
    ),
    *(
        cls(object="bowl", reference="plate", is_active=False)
        for cls in (
            Inside,
            Outside,
            On,
            CenteredOn,
            Above,
            LeftOf,
            RightOf,
            InFrontOf,
            Behind,
            Touching,
        )
    ),
    Stacked(objects=("red_block", "blue_block"), is_active=True, order=StackOrder.BOTTOM_TO_TOP),
    GripperCommanded(action=GripperAction.CLOSE),
    *(cls(object="plate") for cls in (GripperHit, ObjectTipped, ObjectLeftScene)),
    ObjectDisplaced(object="ketchup", distance=0.07),
    GripperFullyClosed(),
    Note(text="joint limit reached", severity=Severity.WARN),
    EpisodeEnded(success=True),
)


def _pose() -> Pose:
    return Pose(position=np.zeros(3), orientation=np.array([0.0, 0.0, 0.0, 1.0]))


def _log_with_step(tmp_path: Path, events: list) -> Path:
    """A log whose only step carries `events` as stored, bypassing the writer."""
    path = tmp_path / "episode.replay"
    header = {"version": REPLAY_LOG_VERSION, "episode_idx": 0, "channels": []}
    step = {"index": 0, "poses": {}, "images": {}, "scalars": {}, "text": {}, "events": events}
    path.write_bytes(
        pack_stream_frame(str(ReplayFrameType.HEADER), header, seq=0)
        + pack_stream_frame(str(ReplayFrameType.STEP), step, seq=1)
    )
    return path


def test_every_type_has_one_event_class():
    types = [event.event_type for event in EVERY_TYPE]
    assert sorted(set(types)) == sorted(ManifoldEventType)
    assert len(types) == len(ManifoldEventType)


def test_every_type_of_event_round_trips(tmp_path):
    path = tmp_path / "episode.replay"
    with ReplayLogWriter(path, episode_idx=0, channels=CHANNELS) as log:
        log.write_step({"bowl": _pose()}, events=EVERY_TYPE)
    assert read_replay_log(path).steps[0].events == EVERY_TYPE


def test_events_stay_on_the_step_they_were_written_on(tmp_path):
    path = tmp_path / "episode.replay"
    grasp, release = Grasped(object="bowl", is_active=True), Grasped(object="bowl", is_active=False)
    with ReplayLogWriter(path, episode_idx=0, channels=CHANNELS) as log:
        log.write_step({"bowl": _pose()}, events=[grasp])
        log.write_step({"bowl": _pose()})
        log.write_step({"bowl": _pose()}, events=[release])
    assert [step.events for step in read_replay_log(path).steps] == [(grasp,), (), (release,)]


def test_a_step_without_events_stores_an_empty_list(tmp_path):
    path = tmp_path / "episode.replay"
    with ReplayLogWriter(path, episode_idx=0, channels=CHANNELS) as log:
        log.write_step({"bowl": _pose()})
    with path.open("rb") as handle:
        frames = iter(lambda: read_stream_frame(handle.read), None)
        step = next(frame for frame in frames if frame["type"] == ReplayFrameType.STEP)
    assert step["payload"]["events"] == []
    assert read_replay_log(path).steps[0].events == ()


def test_a_step_missing_its_events_is_refused(tmp_path):
    path = _log_with_step(tmp_path, [])
    header, step = _frames(path)
    del step["events"]
    _rewrite(path, header, step)
    with pytest.raises(ValueError, match="must carry 'events'"):
        read_replay_log(path)


def test_a_version_4_log_reads_with_no_events(tmp_path):
    path = _log_with_step(tmp_path, [])
    header, step = _frames(path)
    header["version"] = 4
    del step["events"]
    _rewrite(path, header, step)
    assert read_replay_log(path).steps[0].events == ()


def _frames(path: Path) -> tuple[dict, dict]:
    with path.open("rb") as handle:
        frames = list(iter(lambda: read_stream_frame(handle.read), None))
    return frames[0]["payload"], frames[1]["payload"]


def _rewrite(path: Path, header: dict, step: dict) -> None:
    path.write_bytes(
        pack_stream_frame(str(ReplayFrameType.HEADER), header, seq=0)
        + pack_stream_frame(str(ReplayFrameType.STEP), step, seq=1)
    )


def test_the_stored_type_is_the_facts_name(tmp_path):
    path = tmp_path / "episode.replay"
    with ReplayLogWriter(path, episode_idx=0, channels=CHANNELS) as log:
        log.write_step({}, events=[Inside(object="banana", reference="bin", is_active=True)])
    with path.open("rb") as handle:
        frames = iter(lambda: read_stream_frame(handle.read), None)
        step = next(frame for frame in frames if frame["type"] == ReplayFrameType.STEP)
    assert step["payload"]["events"] == [
        {"event_type": "inside", "object": "banana", "reference": "bin", "is_active": True}
    ]


def test_a_fact_about_an_object_needs_a_name():
    with pytest.raises(ValueError, match="non-empty"):
        Grasped(object=" ", is_active=True)


def test_a_relation_needs_a_named_reference():
    with pytest.raises(ValueError, match="non-empty"):
        On(object="bowl", reference="", is_active=True)


def test_a_displaced_object_must_have_moved():
    with pytest.raises(ValueError, match="greater than 0"):
        ObjectDisplaced(object="ketchup", distance=0.0)


def test_the_gripper_does_not_hit_itself():
    with pytest.raises(ValueError, match="itself"):
        GripperHit(object="gripper")


def test_a_stored_event_of_unknown_type_is_refused(tmp_path):
    path = _log_with_step(tmp_path, [{"event_type": "levitating", "object": "bowl"}])
    with pytest.raises(ValueError, match="does not match any of the expected tags"):
        read_replay_log(path)


def test_a_stored_fact_with_a_malformed_field_is_refused(tmp_path):
    path = _log_with_step(
        tmp_path, [{"event_type": "grasped", "object": "bowl", "is_active": "yes"}]
    )
    with pytest.raises(ValueError, match="is_active"):
        read_replay_log(path)


def test_a_stored_relation_without_a_reference_is_refused(tmp_path):
    path = _log_with_step(tmp_path, [{"event_type": "inside", "object": "bowl", "is_active": True}])
    with pytest.raises(ValueError, match="reference"):
        read_replay_log(path)


def test_an_unordered_stack_round_trips(tmp_path):
    path = tmp_path / "episode.replay"
    stack = Stacked(objects=("cube_1", "cube_2", "cube_3"), is_active=True)
    with ReplayLogWriter(path, episode_idx=0, channels=CHANNELS) as log:
        log.write_step({}, events=[stack])
    assert read_replay_log(path).steps[0].events == (stack,)


def test_a_stack_needs_two_objects():
    with pytest.raises(ValueError, match="at least 2 items"):
        Stacked(objects=("cube_1",), is_active=True)
