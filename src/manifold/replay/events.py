"""The Manifold events a benchmark records on a replay step.

Every event is one named thing that happened on its step. A fact, such as an
object being grasped or inside another, is recorded when it starts or stops
holding. An occurrence, such as the gripper hitting an object, is recorded when
it happens. The step an event rides on is when it happened, so an event carries
no step of its own. Events are observations, not judgements: grading decides
what they mean from the rubric (ADR 0009).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from manifold.lib.compat import StrEnum

if TYPE_CHECKING:
    from collections.abc import Mapping

# The name every benchmark gives the robot's gripper.
_GRIPPER = "gripper"


def _require_name(value: str, what: str) -> None:
    if not value.strip():
        raise ValueError(f"{what} must be a non-empty name")


class ManifoldEventKind(StrEnum):
    """The tag a stored event carries, one per event class."""

    EPISODE_STARTED = "episode_started"
    EPISODE_ENDED = "episode_ended"
    GRASPED = "grasped"
    GRIPPER_TOUCHING = "gripper_touching"
    UPRIGHT = "upright"
    OPEN = "open"
    CLOSED = "closed"
    TURNED_ON = "turned_on"
    INSIDE = "inside"
    OUTSIDE = "outside"
    ON = "on"
    CENTERED_ON = "centered_on"
    ABOVE = "above"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    TOUCHING = "touching"
    STACKED = "stacked"
    GRIPPER_COMMANDED = "gripper_commanded"
    GRIPPER_HIT = "gripper_hit"
    GRIPPER_FULLY_CLOSED = "gripper_fully_closed"
    OBJECT_DISPLACED = "object_displaced"
    OBJECT_TIPPED = "object_tipped"
    OBJECT_LEFT_SCENE = "object_left_scene"
    NOTE = "note"


class GripperAction(StrEnum):
    """Which way the policy commanded the gripper."""

    OPEN = "open"
    CLOSE = "close"


class StackOrder(StrEnum):
    """Which way a stack's objects are listed."""

    BOTTOM_TO_TOP = "bottom_to_top"
    TOP_TO_BOTTOM = "top_to_bottom"


class Severity(StrEnum):
    """How a note is shown. A warning is flagged to the reader, not graded."""

    INFO = "info"
    WARN = "warn"


@dataclass(frozen=True)
class EpisodeStarted:
    """The episode began, with the instruction it was given.

    `attributes` names what varies between episodes of one task, such as a
    perturbation's category and difficulty.
    """

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.EPISODE_STARTED
    instruction: str
    attributes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeEnded:
    """The episode ended, and whether the benchmark counts it a success."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.EPISODE_ENDED
    success: bool


@dataclass(frozen=True)
class ObjectFact:
    """A fact about `object` started or stopped holding.

    The benchmark decides how its simulator evaluates each fact.
    """

    kind: ClassVar[ManifoldEventKind]
    object: str
    holds: bool

    def __post_init__(self) -> None:
        _require_name(self.object, f"a {self.kind} fact's object")


@dataclass(frozen=True)
class Grasped(ObjectFact):
    """The gripper holds `object`, closed on it."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.GRASPED


@dataclass(frozen=True)
class GripperTouching(ObjectFact):
    """The gripper is in contact with `object`, whether or not it is closed."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.GRIPPER_TOUCHING


@dataclass(frozen=True)
class Upright(ObjectFact):
    """`object` stands upright."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.UPRIGHT


@dataclass(frozen=True)
class Open(ObjectFact):
    """`object`, such as a drawer or a door, is open."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.OPEN


@dataclass(frozen=True)
class Closed(ObjectFact):
    """`object`, such as a drawer or a door, is closed."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.CLOSED


@dataclass(frozen=True)
class TurnedOn(ObjectFact):
    """`object`, such as a stove, is turned on."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.TURNED_ON


@dataclass(frozen=True)
class RelationFact:
    """A fact about `object` compared with `reference` started or stopped holding.

    The benchmark decides how its simulator evaluates each fact.
    """

    kind: ClassVar[ManifoldEventKind]
    object: str
    reference: str
    holds: bool

    def __post_init__(self) -> None:
        _require_name(self.object, f"a {self.kind} fact's object")
        _require_name(self.reference, f"a {self.kind} fact's reference")


@dataclass(frozen=True)
class Inside(RelationFact):
    """`object` is inside `reference`, such as a bin."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.INSIDE


@dataclass(frozen=True)
class Outside(RelationFact):
    """`object` is outside `reference`."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.OUTSIDE


@dataclass(frozen=True)
class On(RelationFact):
    """`object` rests on `reference`."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.ON


@dataclass(frozen=True)
class CenteredOn(RelationFact):
    """`object` rests on the centre of `reference`."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.CENTERED_ON


@dataclass(frozen=True)
class Above(RelationFact):
    """`object` is above `reference`."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.ABOVE


@dataclass(frozen=True)
class LeftOf(RelationFact):
    """`object` is left of `reference`."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.LEFT_OF


@dataclass(frozen=True)
class RightOf(RelationFact):
    """`object` is right of `reference`."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.RIGHT_OF


@dataclass(frozen=True)
class InFrontOf(RelationFact):
    """`object` is in front of `reference`."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.IN_FRONT_OF


@dataclass(frozen=True)
class Behind(RelationFact):
    """`object` is behind `reference`."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.BEHIND


@dataclass(frozen=True)
class Touching(RelationFact):
    """`object` is in contact with `reference`."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.TOUCHING


@dataclass(frozen=True)
class Stacked:
    """`objects` are stacked on one another.

    With an `order`, the objects are listed from the bottom or from the top and must
    stack that way. Without one, they may stack in any order.
    """

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.STACKED
    objects: tuple[str, ...]
    holds: bool
    order: StackOrder | None = None

    def __post_init__(self) -> None:
        if len(self.objects) < 2:
            raise ValueError("a stack needs at least two objects")
        for name in self.objects:
            _require_name(name, "a stacked object")


@dataclass(frozen=True)
class GripperCommanded:
    """The policy commanded the gripper open or closed."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.GRIPPER_COMMANDED
    action: GripperAction


@dataclass(frozen=True)
class ObjectOccurrence:
    """Something happened to `object`."""

    kind: ClassVar[ManifoldEventKind]
    object: str

    def __post_init__(self) -> None:
        _require_name(self.object, f"a {self.kind} event's object")


@dataclass(frozen=True)
class GripperHit(ObjectOccurrence):
    """The gripper struck `object`, such as the table or an object it did not grasp."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.GRIPPER_HIT

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.object == _GRIPPER:
            raise ValueError("the gripper cannot hit itself")


@dataclass(frozen=True)
class ObjectTipped(ObjectOccurrence):
    """`object` tipped over."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.OBJECT_TIPPED


@dataclass(frozen=True)
class ObjectLeftScene(ObjectOccurrence):
    """`object` left the workspace."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.OBJECT_LEFT_SCENE


@dataclass(frozen=True)
class ObjectDisplaced:
    """`object` moved without being grasped and came to rest `distance` metres away."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.OBJECT_DISPLACED
    object: str
    distance: float

    def __post_init__(self) -> None:
        _require_name(self.object, "a displaced object")
        if not self.distance > 0:
            raise ValueError("a displaced object must have moved")


@dataclass(frozen=True)
class GripperFullyClosed:
    """The gripper closed all the way, with nothing between its fingers."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.GRIPPER_FULLY_CLOSED


@dataclass(frozen=True)
class Note:
    """Free text for a reader. A grader never reads a note."""

    kind: ClassVar[ManifoldEventKind] = ManifoldEventKind.NOTE
    text: str
    severity: Severity = Severity.INFO


# Every event a replay step may carry.
ManifoldEvent = (
    EpisodeStarted
    | EpisodeEnded
    | ObjectFact
    | RelationFact
    | Stacked
    | GripperCommanded
    | ObjectOccurrence
    | ObjectDisplaced
    | GripperFullyClosed
    | Note
)

__all__ = [
    "Above",
    "Behind",
    "CenteredOn",
    "Closed",
    "EpisodeEnded",
    "EpisodeStarted",
    "Grasped",
    "GripperAction",
    "GripperCommanded",
    "GripperFullyClosed",
    "GripperHit",
    "GripperTouching",
    "InFrontOf",
    "Inside",
    "LeftOf",
    "ManifoldEvent",
    "ManifoldEventKind",
    "Note",
    "ObjectDisplaced",
    "ObjectFact",
    "ObjectLeftScene",
    "ObjectOccurrence",
    "ObjectTipped",
    "On",
    "Open",
    "Outside",
    "RelationFact",
    "RightOf",
    "Severity",
    "StackOrder",
    "Stacked",
    "Touching",
    "TurnedOn",
    "Upright",
]
