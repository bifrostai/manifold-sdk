"""How the Rerun viewer shows a replay: its pages of charts and its events panel.

The runner turns each replay log into a Rerun recording. A scalar channel's
`group` is the viewer's page for its line chart, and its `row` is the line of the
page's grid. The `events` text channel is the viewer's events panel, which shows
a line for each event that tells the story of the episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from manifold.lib.compat import StrEnum
from manifold.replay.events import (
    EpisodeEndedEvent,
    EpisodeStartedEvent,
    GraspedEvent,
    GripperCommandedEvent,
    ManifoldEvent,
    NoteEvent,
)

# The rows of the `Action` page, which the viewer shows beside the 3D scene, top
# first: the gripper alone, then the arm, then the arm's rotation.
RERUN_GRIPPER_ROW = 0
RERUN_ARM_ROW = 1
RERUN_ROTATION_ROW = 2


class RerunTab(StrEnum):
    """The page of the Rerun viewer that shows a scalar as a line chart.

    A scalar channel's `group` is its page. The viewer shows the `ACTION` page
    beside the 3D scene and each other page on its own.
    """

    ACTION = "Action"
    ROBOT = "Robot"
    TASK = "Task"
    PHYSICS = "Physics"


@dataclass(frozen=True)
class RerunScalar:
    """A scalar's value at one step, and where the Rerun viewer shows it.

    `tab` is the page of the chart. Names that share a `/` prefix are drawn as the
    series of one chart, so `joints/0` to `joints/6` make one chart. `row` is the
    line of the page's grid that the chart is on, or None to let the viewer place it.
    """

    name: str
    tab: RerunTab
    value: float
    row: int | None


def rerun_scalars(name: str, tab: RerunTab, values: Any) -> list[RerunScalar]:
    """A Rerun scalar for each value, named `<name>/<i>` so that they share a chart."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return [
        RerunScalar(f"{name}/{index}", tab, float(value), None) for index, value in enumerate(flat)
    ]


def rerun_panel_line(event: ManifoldEvent, step: int) -> str | None:
    """The line that a replay's events panel shows for `event` on `step`, or None.

    A fact that is not about the gripper holding an object shows no line, because
    the panel tells the story of the episode and the facts are plotted.
    """
    if isinstance(event, EpisodeStartedEvent):
        labels = ", ".join(f"{name}: {value}" for name, value in event.attributes.items())
        return (
            f"episode started [{labels}]: {event.instruction}"
            if labels
            else (f"episode started: {event.instruction}")
        )
    if isinstance(event, EpisodeEndedEvent):
        return f"episode ended at step {step}: {'SUCCESS' if event.success else 'FAILURE'}"
    if isinstance(event, GripperCommandedEvent):
        return f"gripper {event.action.value} at step {step}"
    if isinstance(event, GraspedEvent):
        return (
            f"grasp: {event.object} at step {step}"
            if event.is_active
            else (f"release at step {step}")
        )
    if isinstance(event, NoteEvent):
        return f"[{event.severity.value.upper()}] {event.text}"
    return None


__all__ = [
    "RERUN_ARM_ROW",
    "RERUN_GRIPPER_ROW",
    "RERUN_ROTATION_ROW",
    "RerunScalar",
    "RerunTab",
    "rerun_panel_line",
    "rerun_scalars",
]
