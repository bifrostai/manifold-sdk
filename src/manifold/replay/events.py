"""The Manifold events a benchmark records on a replay step.

Every event is one named thing that happened on its step. A fact, such as an
object being grasped or inside another, is recorded when it starts or stops
holding. An occurrence, such as the gripper hitting an object, is recorded when
it happens. The step an event rides on is when it happened, so an event carries
no step of its own. Events are observations, not judgements: grading decides
what they mean from the rubric (ADR 0009).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StrictBool, field_validator

from manifold.lib.compat import StrEnum

# The name every benchmark gives the robot's gripper.
_GRIPPER = "gripper"


def _non_empty(value: str) -> str:
    if not value.strip():
        raise ValueError("must be a non-empty name")
    return value


# A scene object's name, such as `bowl_1`.
_Name = Annotated[str, AfterValidator(_non_empty)]


class ManifoldEventType(StrEnum):
    """The type a stored event carries, one per event class."""

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


class _Event(BaseModel):
    model_config = ConfigDict(frozen=True)


class EpisodeStarted(_Event):
    """The episode began, with the instruction it was given.

    `attributes` names what varies between episodes of one task, such as a
    perturbation's category and difficulty.
    """

    event_type: Literal[ManifoldEventType.EPISODE_STARTED] = ManifoldEventType.EPISODE_STARTED
    instruction: str
    attributes: dict[str, str] = Field(default_factory=dict)


class EpisodeEnded(_Event):
    """The episode ended, and whether the benchmark counts it a success."""

    event_type: Literal[ManifoldEventType.EPISODE_ENDED] = ManifoldEventType.EPISODE_ENDED
    success: StrictBool


class ObjectFact(_Event):
    """A fact about `object` started or stopped holding, as `is_active` says.

    The benchmark decides how its simulator evaluates each fact.
    """

    object: _Name
    is_active: StrictBool


class Grasped(ObjectFact):
    """The gripper holds `object`, closed on it."""

    event_type: Literal[ManifoldEventType.GRASPED] = ManifoldEventType.GRASPED


class GripperTouching(ObjectFact):
    """The gripper is in contact with `object`, whether or not it is closed."""

    event_type: Literal[ManifoldEventType.GRIPPER_TOUCHING] = ManifoldEventType.GRIPPER_TOUCHING


class Upright(ObjectFact):
    """`object` stands upright."""

    event_type: Literal[ManifoldEventType.UPRIGHT] = ManifoldEventType.UPRIGHT


class Open(ObjectFact):
    """`object`, such as a drawer or a door, is open."""

    event_type: Literal[ManifoldEventType.OPEN] = ManifoldEventType.OPEN


class Closed(ObjectFact):
    """`object`, such as a drawer or a door, is closed."""

    event_type: Literal[ManifoldEventType.CLOSED] = ManifoldEventType.CLOSED


class TurnedOn(ObjectFact):
    """`object`, such as a stove, is turned on."""

    event_type: Literal[ManifoldEventType.TURNED_ON] = ManifoldEventType.TURNED_ON


class RelationFact(_Event):
    """A fact about `object` compared with `reference` started or stopped holding.

    `is_active` says which. The benchmark decides how its simulator evaluates each fact.
    """

    object: _Name
    reference: _Name
    is_active: StrictBool


class Inside(RelationFact):
    """`object` is inside `reference`, such as a bin."""

    event_type: Literal[ManifoldEventType.INSIDE] = ManifoldEventType.INSIDE


class Outside(RelationFact):
    """`object` is outside `reference`."""

    event_type: Literal[ManifoldEventType.OUTSIDE] = ManifoldEventType.OUTSIDE


class On(RelationFact):
    """`object` rests on `reference`."""

    event_type: Literal[ManifoldEventType.ON] = ManifoldEventType.ON


class CenteredOn(RelationFact):
    """`object` rests on the centre of `reference`."""

    event_type: Literal[ManifoldEventType.CENTERED_ON] = ManifoldEventType.CENTERED_ON


class Above(RelationFact):
    """`object` is above `reference`."""

    event_type: Literal[ManifoldEventType.ABOVE] = ManifoldEventType.ABOVE


class LeftOf(RelationFact):
    """`object` is left of `reference`."""

    event_type: Literal[ManifoldEventType.LEFT_OF] = ManifoldEventType.LEFT_OF


class RightOf(RelationFact):
    """`object` is right of `reference`."""

    event_type: Literal[ManifoldEventType.RIGHT_OF] = ManifoldEventType.RIGHT_OF


class InFrontOf(RelationFact):
    """`object` is in front of `reference`."""

    event_type: Literal[ManifoldEventType.IN_FRONT_OF] = ManifoldEventType.IN_FRONT_OF


class Behind(RelationFact):
    """`object` is behind `reference`."""

    event_type: Literal[ManifoldEventType.BEHIND] = ManifoldEventType.BEHIND


class Touching(RelationFact):
    """`object` is in contact with `reference`."""

    event_type: Literal[ManifoldEventType.TOUCHING] = ManifoldEventType.TOUCHING


class Stacked(_Event):
    """`objects` are stacked on one another.

    With an `order`, the objects are listed from the bottom or from the top and must
    stack that way. Without one, they may stack in any order.
    """

    event_type: Literal[ManifoldEventType.STACKED] = ManifoldEventType.STACKED
    objects: tuple[_Name, ...] = Field(min_length=2)
    is_active: StrictBool
    order: StackOrder | None = None


class GripperCommanded(_Event):
    """The policy commanded the gripper open or closed."""

    event_type: Literal[ManifoldEventType.GRIPPER_COMMANDED] = ManifoldEventType.GRIPPER_COMMANDED
    action: GripperAction


class ObjectOccurrence(_Event):
    """Something happened to `object`."""

    object: _Name


class GripperHit(ObjectOccurrence):
    """The gripper struck `object`, such as the table or an object it did not grasp."""

    event_type: Literal[ManifoldEventType.GRIPPER_HIT] = ManifoldEventType.GRIPPER_HIT

    @field_validator("object")
    @classmethod
    def _not_the_gripper(cls, value: str) -> str:
        if value == _GRIPPER:
            raise ValueError("the gripper cannot hit itself")
        return value


class ObjectTipped(ObjectOccurrence):
    """`object` tipped over."""

    event_type: Literal[ManifoldEventType.OBJECT_TIPPED] = ManifoldEventType.OBJECT_TIPPED


class ObjectLeftScene(ObjectOccurrence):
    """`object` left the workspace."""

    event_type: Literal[ManifoldEventType.OBJECT_LEFT_SCENE] = ManifoldEventType.OBJECT_LEFT_SCENE


class ObjectDisplaced(_Event):
    """`object` moved without being grasped and came to rest `distance` metres away."""

    event_type: Literal[ManifoldEventType.OBJECT_DISPLACED] = ManifoldEventType.OBJECT_DISPLACED
    object: _Name
    distance: float = Field(gt=0)


class GripperFullyClosed(_Event):
    """The gripper closed all the way, with nothing between its fingers."""

    event_type: Literal[ManifoldEventType.GRIPPER_FULLY_CLOSED] = (
        ManifoldEventType.GRIPPER_FULLY_CLOSED
    )


class Note(_Event):
    """Free text for a reader. A grader never reads a note."""

    event_type: Literal[ManifoldEventType.NOTE] = ManifoldEventType.NOTE
    text: str
    severity: Severity = Severity.INFO


# Every event a replay step may carry, told apart by `event_type`.
ManifoldEvent = Annotated[
    EpisodeStarted
    | EpisodeEnded
    | Grasped
    | GripperTouching
    | Upright
    | Open
    | Closed
    | TurnedOn
    | Inside
    | Outside
    | On
    | CenteredOn
    | Above
    | LeftOf
    | RightOf
    | InFrontOf
    | Behind
    | Touching
    | Stacked
    | GripperCommanded
    | GripperHit
    | ObjectTipped
    | ObjectLeftScene
    | ObjectDisplaced
    | GripperFullyClosed
    | Note,
    Field(discriminator="event_type"),
]
