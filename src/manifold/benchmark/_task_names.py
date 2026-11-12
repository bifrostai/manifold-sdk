"""The scene bodies and sites that a hooks file names for the task, for one episode."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskBody:
    """A scene body that the task names, by the simulator's name for the body.

    The body takes the task's name in the replay's scene. When the body is a fixture, the
    bodies under it in the model's body tree are its parts, such as its doors, shelves and
    knobs. A part with a name of its own takes the task's name with the rest of its name,
    so `stove_1_knob_front_left` becomes `stove_knob_front_left` for the task name
    `stove`, and records `stove` as its parent. Another part takes the task's name.
    """

    body: str


@dataclass(frozen=True)
class TaskSite:
    """A site that the task names, such as a burner on a stove, by the simulator's name for it.

    A site has a shape but no body of its own, so the scene has no body for it. The
    replay adds one, named after the site, that moves with the site's body.
    """

    site: str


# What a task name refers to in the scene.
TaskTarget = TaskBody | TaskSite

__all__ = [
    "TaskBody",
    "TaskSite",
    "TaskTarget",
]
