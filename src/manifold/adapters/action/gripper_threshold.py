"""Binarize a continuous open-high gripper head into a target encoding.

Some policies emit the gripper as a continuous score in [0, 1] where the high end
means open, then leave a hard threshold to the serving glue. That threshold is a
convention step, so it belongs in an adapter: this one reads the head as a raw
open-high UNSIGNED score, thresholds it at `cutoff` into fully-open or
fully-closed, and emits the value the target encoding gives for that openness.

The target's polarity decides the emitted sign. The head is open-high, so a value
above the cutoff is open; `from_openness(1.0, target)` then maps "open" to whatever
the target calls open: `+1` for SIGNED, `-1` for SIGNED_OPEN_LOW. One adapter, two
declared outcomes — the benchmark's gripper format selects which.

Unlike `GripperPolarityAdapter` (a lossless remap of one continuous value to
another), this discards the continuous head's magnitude at the threshold, so it is
declared lossy: the caller opts in by supplying it.

`GripperThresholdAdapter` acts on a plain `EEActionSpace`. Its sibling
`UnifiedGripperThresholdAdapter` does the same job when the gripper lives inside a
whole-body `UnifiedActionSpace` payload (the RLDX/RoboCasa use case). The two are
declared with distinct `from_spec`s — `check_compatibility` matches an adapter by
exact `from_spec` type, so a single adapter cannot serve both forms through the
checked pipeline; they share their thresholding core via `_threshold_gripper` so
they cannot drift.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel

from manifold.core.adapter import ActionAdapter
from manifold.core.conventions import GripperFormat
from manifold.core.embodiment import EEActionSpace, UnifiedActionSpace
from manifold.lib.gripper import from_openness


def _threshold_gripper(
    out: list[float],
    gripper_offset: int,
    group_width: int,
    groups: int,
    target: GripperFormat,
    cutoff: float,
) -> None:
    """Threshold one gripper slot in each of `groups` back-to-back groups, in place.

    The slot sits at `gripper_offset` within each `group_width`-wide group. A group is
    whatever the caller strides by -- one arm of an end-effector action, or one step
    of a unified buffer -- so the helper knows nothing of either. The head is a raw
    open-high score (not yet in `target`'s range), so it is thresholded directly:
    above the cutoff is open (openness 1.0), at or below is closed.
    """
    for group in range(groups):
        gripper_index = group * group_width + gripper_offset
        openness = 1.0 if out[gripper_index] > cutoff else 0.0
        out[gripper_index] = from_openness(openness, target)


class GripperThresholdAdapter(ActionAdapter):
    """Threshold a continuous open-high gripper head into the `target` encoding."""

    from_spec: ClassVar[type[BaseModel]] = EEActionSpace
    to_spec: ClassVar[type[BaseModel]] = EEActionSpace
    lossless: ClassVar[bool] = False

    def __init__(self, target: GripperFormat, cutoff: float = 0.5) -> None:
        self.target = target
        self.cutoff = cutoff

    def applies(self, source: BaseModel) -> bool:
        """True when `source` carries a gripper that differs from the target."""
        return (
            isinstance(source, EEActionSpace)
            and source.gripper is not None
            and source.gripper != self.target
        )

    def produce(self, source: BaseModel) -> EEActionSpace:
        """The same action space with its gripper set to the target encoding."""
        if not isinstance(source, EEActionSpace):
            raise TypeError("GripperThresholdAdapter transforms EEActionSpace only")
        return source.model_copy(update={"gripper": self.target})

    def adapt(self, values: Any, *, source: BaseModel) -> list[float]:
        """Threshold every arm's gripper element into the target."""
        if not isinstance(source, EEActionSpace) or source.gripper is None:
            raise TypeError("GripperThresholdAdapter needs an EEActionSpace with a gripper")
        out = [float(v) for v in values]
        source.validate_value(out)  # the slice below assumes the declared length
        # The gripper is the last element of each ARM's values, and a bimanual step
        # holds `arm_count` of them. Treating the step as one arm would threshold
        # the trailing gripper and leave the others in the source encoding -- a robot
        # whose left arm obeys a different convention from its right.
        per_arm = source.per_arm_length()
        _threshold_gripper(
            out,
            gripper_offset=per_arm - 1,
            group_width=per_arm,
            groups=source.chunk_size * source.arm_count,
            target=self.target,
            cutoff=self.cutoff,
        )
        return out


class UnifiedGripperThresholdAdapter(ActionAdapter):
    """Threshold the gripper inside a whole-body `UnifiedActionSpace` payload.

    The whole-body counterpart of `GripperThresholdAdapter`: the RLDX/RoboCasa use
    case, where the policy emits a fixed-width buffer (e.g. 12-D) whose leading slots
    are an `EEActionSpace` payload carrying the grippers and the rest is padding. Each
    arm's gripper ends that arm's values within the payload, in every `width`-wide
    step, located via the payload's `per_arm_length()` — never hardcoded.

    It is declared with `from_spec = to_spec = UnifiedActionSpace` so it is reachable
    through `check_compatibility`, which matches an adapter by exact `from_spec` type.
    Polarity and lossy semantics are identical to the EE adapter: it shares
    `_threshold_gripper`, and both walk every arm of the payload.
    """

    from_spec: ClassVar[type[BaseModel]] = UnifiedActionSpace
    to_spec: ClassVar[type[BaseModel]] = UnifiedActionSpace
    lossless: ClassVar[bool] = False

    def __init__(self, target: GripperFormat, cutoff: float = 0.5) -> None:
        self.target = target
        self.cutoff = cutoff

    def applies(self, source: BaseModel) -> bool:
        """True when the unified payload is an EE space whose gripper differs from the target."""
        if not isinstance(source, UnifiedActionSpace):
            return False
        payload = source.payload
        return (
            isinstance(payload, EEActionSpace)
            and payload.gripper is not None
            and payload.gripper != self.target
        )

    def produce(self, source: BaseModel) -> UnifiedActionSpace:
        """The same buffer with the EE payload's gripper set to the target encoding."""
        if not isinstance(source, UnifiedActionSpace) or not isinstance(
            source.payload, EEActionSpace
        ):
            raise TypeError(
                "UnifiedGripperThresholdAdapter transforms a "
                "UnifiedActionSpace(payload=EEActionSpace) only"
            )
        new_payload = source.payload.model_copy(update={"gripper": self.target})
        return source.model_copy(update={"payload": new_payload})

    def adapt(self, values: Any, *, source: BaseModel) -> list[float]:
        """Threshold every arm's gripper inside each `width`-wide step of the buffer.

        Arm `n`'s gripper is the last dim of that arm's values in the payload, at
        `n * per_arm + per_arm - 1` within each `width`-wide step. The payload's
        per-step length must fit within `width`; if it does not the layout is
        malformed and indexing a gripper would read past the step, so raise rather
        than corrupt a neighbouring dim.
        """
        if not isinstance(source, UnifiedActionSpace) or not isinstance(
            source.payload, EEActionSpace
        ):
            raise TypeError(
                "UnifiedGripperThresholdAdapter needs a "
                "UnifiedActionSpace(payload=EEActionSpace) source"
            )
        payload = source.payload
        if payload.gripper is None:
            raise TypeError("UnifiedGripperThresholdAdapter needs a unified payload with a gripper")
        out = [float(v) for v in values]
        source.validate_value(out)
        per_step = payload.per_step_length()
        if per_step > source.width:
            raise ValueError(
                f"unified payload per-step length {per_step} does not fit within "
                f"the buffer width {source.width}: the gripper would index past the step"
            )
        # One pass per arm, each at that arm's gripper: offsetting by the whole payload
        # step would threshold the last arm alone and hand the rest on as raw scores.
        per_arm = payload.per_arm_length()
        for arm in range(payload.arm_count):
            _threshold_gripper(
                out,
                gripper_offset=arm * per_arm + per_arm - 1,
                group_width=source.width,
                groups=payload.chunk_size,
                target=self.target,
                cutoff=self.cutoff,
            )
        return out


__all__ = ["GripperThresholdAdapter", "UnifiedGripperThresholdAdapter"]
