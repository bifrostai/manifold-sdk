"""Tests for how the Rerun viewer shows a replay's scalars and events."""

from __future__ import annotations

import pytest

from manifold.replay import (
    EpisodeEndedEvent,
    EpisodeStartedEvent,
    GraspedEvent,
    GripperAction,
    GripperCommandedEvent,
    NavigatingEvent,
    NoteEvent,
    OnEvent,
    RerunScalar,
    RerunTab,
    Severity,
    rerun_panel_line,
    rerun_scalars,
)


def test_a_vector_becomes_a_scalar_for_each_value_on_one_chart():
    assert rerun_scalars("joints", RerunTab.ROBOT, [0.5, 1.5]) == [
        RerunScalar("joints/0", RerunTab.ROBOT, 0.5, None),
        RerunScalar("joints/1", RerunTab.ROBOT, 1.5, None),
    ]


@pytest.mark.parametrize(
    ("event", "line"),
    [
        (EpisodeStartedEvent(instruction="put the bowl"), "episode started: put the bowl"),
        (
            EpisodeStartedEvent(instruction="put the bowl", attributes={"category": "camera"}),
            "episode started [category: camera]: put the bowl",
        ),
        (EpisodeEndedEvent(success=True), "episode ended at step 7: SUCCESS"),
        (
            GripperCommandedEvent(gripper="1", action=GripperAction.CLOSE),
            "gripper 1 close at step 7",
        ),
        (NavigatingEvent(is_active=True), "navigating started at step 7"),
        (NavigatingEvent(is_active=False), "navigating stopped at step 7"),
        (
            GraspedEvent(gripper="1", object="bowl", is_active=True),
            "grasp: bowl by gripper 1 at step 7",
        ),
        (
            GraspedEvent(gripper="1", object="bowl", is_active=False),
            "release: bowl by gripper 1 at step 7",
        ),
        (NoteEvent(text="sub-task passed", severity=Severity.WARN), "[WARN] sub-task passed"),
        (OnEvent(object="bowl", reference="plate", is_active=True), None),
    ],
)
def test_each_event_has_its_line_in_the_events_panel(event, line):
    assert rerun_panel_line(event, 7) == line


def test_a_declared_hand_reads_hand_in_the_events_panel():
    event = GraspedEvent(gripper="left", object="bowl", is_active=True)

    assert rerun_panel_line(event, 7, {"left": "hand"}) == "grasp: bowl by hand left at step 7"
