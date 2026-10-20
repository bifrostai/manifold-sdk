"""The layout of a replay in the Rerun viewer: pages of charts and an events panel.

The runner converts each replay log into a Rerun recording. A scalar channel's
`group` is the page of its chart, and its `row` is the row of the page's grid. The
viewer shows the `events` text channel in an events panel, with a line for each
event.
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

# The rows of the `Action` page, from the top: the gripper, then the arm, then the
# arm's rotation. The viewer shows this page beside the 3D scene.
RERUN_GRIPPER_ROW = 0
RERUN_ARM_ROW = 1
RERUN_ROTATION_ROW = 2


class RerunTab(StrEnum):
    """A page of the Rerun viewer. Each page shows its scalars as line charts.

    A scalar channel's `group` is its page. The viewer shows the `ACTION` page
    beside the 3D scene, and the other pages alone.
    """

    ACTION = "Action"
    ROBOT = "Robot"
    TASK = "Task"
    PHYSICS = "Physics"


@dataclass(frozen=True)
class RerunScalar:
    """A scalar's value at one step, and where the Rerun viewer shows it.

    `tab` is the page of the chart. The viewer draws names with the same prefix before
    `/` as series of a single chart, such as `joints/0` to `joints/6`. `row` is the
    row of the page's grid for the chart, or None if the viewer places the chart.
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

    Other events show no line. The `Task` page charts the fact events.
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
