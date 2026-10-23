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


class GripperAction(StrEnum):
    """Which way the policy commanded the gripper."""

    OPEN = "open"
    CLOSE = "close"


class StackOrder(StrEnum):
    """Which way a stack's objects are listed."""

    BOTTOM_TO_TOP = "bottom_to_top"
    TOP_TO_BOTTOM = "top_to_bottom"


class FailureType(StrEnum):
    """A type in the failure taxonomy (ADR 0009). No other type is a failure."""

    NO_CONTACT = "no_contact"
    GRASP_POSITION_OFFSET = "grasp_position_offset"
    GRASP_ORIENTATION_OFFSET = "grasp_orientation_offset"
    NO_GRASP = "no_grasp"
    PLACE_POSITION_OFFSET = "place_position_offset"
    PLACE_ORIENTATION_OFFSET = "place_orientation_offset"
    ACTUATION_INCOMPLETE = "actuation_incomplete"
    SLIP_DROP = "slip_drop"
    COLLISION_OBJECT = "collision_object"
    COLLISION_ENVIRONMENT = "collision_environment"
    OBJECT_MOVED_UNAIDED = "object_moved_unaided"
    ENVIRONMENT_OCCUPIED = "environment_occupied"
    ENVIRONMENT_BLOCKING = "environment_blocking"
    WRONG_DESTINATION = "wrong_destination"
    WRONG_OBJECT = "wrong_object"
    VIEW_OCCLUDED = "view_occluded"
    PREREQUISITE_SKIPPED = "prerequisite_skipped"
    STEP_MISSED = "step_missed"
    WRONG_ORDER = "wrong_order"
    AMBIGUOUS_INSTRUCTION = "ambiguous_instruction"


class Severity(StrEnum):
    """How a note is shown. A warning is flagged to the reader, not graded."""

    INFO = "info"
    WARN = "warn"


class _Event(BaseModel):
    model_config = ConfigDict(frozen=True)


class EpisodeStartedEvent(_Event):
    """The episode began, with the instruction it was given.

    `attributes` names what varies between episodes of one task, such as a
    perturbation's category and difficulty.
    """

    event_type: Literal["episode_started"] = "episode_started"
    instruction: str
    attributes: dict[str, str] = Field(default_factory=dict)


class EpisodeEndedEvent(_Event):
    """The episode ended, and whether the benchmark counts it a success."""

    event_type: Literal["episode_ended"] = "episode_ended"
    success: StrictBool


class FactEvent(_Event):
    """A fact started or stopped holding, as `is_active` says.

    A hooks file returns each fact's current `is_active` at every step. The replay
    writes the event only on the step where `is_active` changes. The benchmark
    decides how its simulator evaluates each fact.
    """

    is_active: StrictBool


class ObjectFactEvent(FactEvent):
    """A fact about `object` started or stopped holding."""

    object: _Name


class GraspedEvent(ObjectFactEvent):
    """The gripper holds `object`, closed on it."""

    event_type: Literal["grasped"] = "grasped"


class GripperTouchingEvent(ObjectFactEvent):
    """The gripper is in contact with `object`, whether or not it is closed."""

    event_type: Literal["gripper_touching"] = "gripper_touching"


class UprightEvent(ObjectFactEvent):
    """`object` stands upright."""

    event_type: Literal["upright"] = "upright"


class OpenEvent(ObjectFactEvent):
    """`object`, such as a drawer or a door, is open."""

    event_type: Literal["open"] = "open"


class ClosedEvent(ObjectFactEvent):
    """`object`, such as a drawer or a door, is closed."""

    event_type: Literal["closed"] = "closed"


class TurnedOnEvent(ObjectFactEvent):
    """`object`, such as a stove, is turned on."""

    event_type: Literal["turned_on"] = "turned_on"


class RelationFactEvent(FactEvent):
    """A fact about `object` compared with `reference` started or stopped holding."""

    object: _Name
    reference: _Name


class InsideEvent(RelationFactEvent):
    """`object` is inside `reference`, such as a bin."""

    event_type: Literal["inside"] = "inside"


class OutsideEvent(RelationFactEvent):
    """`object` is outside `reference`."""

    event_type: Literal["outside"] = "outside"


class OnEvent(RelationFactEvent):
    """`object` rests on `reference`."""

    event_type: Literal["on"] = "on"


class CenteredOnEvent(RelationFactEvent):
    """`object` rests on the centre of `reference`."""

    event_type: Literal["centered_on"] = "centered_on"


class AboveEvent(RelationFactEvent):
    """`object` is above `reference`."""

    event_type: Literal["above"] = "above"


class LeftOfEvent(RelationFactEvent):
    """`object` is left of `reference`."""

    event_type: Literal["left_of"] = "left_of"


class RightOfEvent(RelationFactEvent):
    """`object` is right of `reference`."""

    event_type: Literal["right_of"] = "right_of"


class InFrontOfEvent(RelationFactEvent):
    """`object` is in front of `reference`."""

    event_type: Literal["in_front_of"] = "in_front_of"


class BehindEvent(RelationFactEvent):
    """`object` is behind `reference`."""

    event_type: Literal["behind"] = "behind"


class TouchingEvent(RelationFactEvent):
    """`object` is in contact with `reference`."""

    event_type: Literal["touching"] = "touching"


class StackedEvent(FactEvent):
    """`objects` are stacked on one another.

    With an `order`, the objects are listed from the bottom or from the top and must
    stack that way. Without one, they may stack in any order.
    """

    event_type: Literal["stacked"] = "stacked"
    objects: tuple[_Name, ...] = Field(min_length=2)
    order: StackOrder | None = None


class GripperCommandedEvent(_Event):
    """The policy commanded the gripper open or closed."""

    event_type: Literal["gripper_commanded"] = "gripper_commanded"
    action: GripperAction


class ObjectOccurrenceEvent(_Event):
    """Something happened to `object`."""

    object: _Name


class GripperHitEvent(ObjectOccurrenceEvent):
    """The gripper struck `object`, such as the table or an object it did not grasp."""

    event_type: Literal["gripper_hit"] = "gripper_hit"

    @field_validator("object")
    @classmethod
    def _not_the_gripper(cls, value: str) -> str:
        if value == _GRIPPER:
            raise ValueError("the gripper cannot hit itself")
        return value


class ObjectTippedEvent(ObjectOccurrenceEvent):
    """`object` tipped over."""

    event_type: Literal["object_tipped"] = "object_tipped"


class ObjectLeftSceneEvent(ObjectOccurrenceEvent):
    """`object` left the workspace."""

    event_type: Literal["object_left_scene"] = "object_left_scene"


class ObjectDisplacedEvent(_Event):
    """`object` moved without being grasped and came to rest `distance` metres away."""

    event_type: Literal["object_displaced"] = "object_displaced"
    object: _Name
    distance: float = Field(gt=0)


class GripperFullyClosedEvent(_Event):
    """The gripper closed all the way, with nothing between its fingers."""

    event_type: Literal["gripper_fully_closed"] = "gripper_fully_closed"


class FailureEvent(_Event):
    """A failure of `failure_type` that the simulator detected, with its objects.

    `object` and `reference` name the objects that the failure involved. Both
    are optional: `object` is left out when the failure names no single object,
    such as `wrong_order` or `step_missed`. A benchmark records a failure only for
    a type that its log header lists (ADR 0009).
    """

    event_type: Literal["failure"] = "failure"
    failure_type: FailureType
    object: _Name | None = None
    reference: _Name | None = None


class NoteEvent(_Event):
    """Free text for a reader. A grader never reads a note."""

    event_type: Literal["note"] = "note"
    text: str
    severity: Severity = Severity.INFO


# Every event a replay step may carry, told apart by `event_type`.
ManifoldEvent = Annotated[
    EpisodeStartedEvent
    | EpisodeEndedEvent
    | GraspedEvent
    | GripperTouchingEvent
    | UprightEvent
    | OpenEvent
    | ClosedEvent
    | TurnedOnEvent
    | InsideEvent
    | OutsideEvent
    | OnEvent
    | CenteredOnEvent
    | AboveEvent
    | LeftOfEvent
    | RightOfEvent
    | InFrontOfEvent
    | BehindEvent
    | TouchingEvent
    | StackedEvent
    | GripperCommandedEvent
    | GripperHitEvent
    | ObjectTippedEvent
    | ObjectLeftSceneEvent
    | ObjectDisplacedEvent
    | GripperFullyClosedEvent
    | FailureEvent
    | NoteEvent,
    Field(discriminator="event_type"),
]
