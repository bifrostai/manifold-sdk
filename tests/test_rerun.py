"""Tests for how the Rerun viewer shows a replay's scalars and events."""

from __future__ import annotations

import pytest

from manifold.replay import (
    EpisodeEndedEvent,
    EpisodeStartedEvent,
    GraspedEvent,
    GripperAction,
    GripperCommandedEvent,
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
        (GripperCommandedEvent(action=GripperAction.CLOSE), "gripper close at step 7"),
        (GraspedEvent(object="bowl", is_active=True), "grasp: bowl at step 7"),
        (GraspedEvent(object="bowl", is_active=False), "release at step 7"),
        (NoteEvent(text="sub-task passed", severity=Severity.WARN), "[WARN] sub-task passed"),
        (OnEvent(object="bowl", reference="plate", is_active=True), None),
    ],
)
def test_each_event_has_its_line_in_the_events_panel(event, line):
    assert rerun_panel_line(event, 7) == line
