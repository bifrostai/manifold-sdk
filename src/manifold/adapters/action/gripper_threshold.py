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
    gripper_index: int,
    target: GripperFormat,
    cutoff: float,
) -> None:
    """Threshold the gripper slot of an action in place.

    The head is a raw open-high score (not yet in `target`'s range), so threshold
    it directly: above the cutoff is open (openness 1.0), at or below is closed.
    Shared by both threshold adapters so the EE and unified paths cannot drift.
    """
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
        """Threshold the gripper element of the action into the target."""
        if not isinstance(source, EEActionSpace) or source.gripper is None:
            raise TypeError("GripperThresholdAdapter needs an EEActionSpace with a gripper")
        out = [float(v) for v in values]
        source.validate_value(out)  # the slice below assumes the declared length
        # The gripper is the last element of the action.
        _threshold_gripper(out, gripper_index=len(out) - 1, target=self.target, cutoff=self.cutoff)
        return out


class UnifiedGripperThresholdAdapter(ActionAdapter):
    """Threshold the gripper inside a whole-body `UnifiedActionSpace` payload.

    The whole-body counterpart of `GripperThresholdAdapter`: the RLDX/RoboCasa use
    case, where the policy emits a fixed-width buffer (e.g. 12-D) whose leading slots
    are an `EEActionSpace` payload carrying the gripper and the rest is padding. The
    gripper sits at the last slot of the payload within the `width`-wide action,
    located via the payload's length — never hardcoded.

    It is declared with `from_spec = to_spec = UnifiedActionSpace` so it is reachable
    through `check_compatibility`, which matches an adapter by exact `from_spec` type.
    Polarity and lossy semantics are identical to the EE adapter: it shares
    `_threshold_gripper`, so the two cannot drift.
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
        """Threshold the gripper dim inside the width-D action.

        The gripper sits at `payload.expected_length() - 1` (the last dim of the EE
        payload). The payload's length must fit within `width`; if it does not the
        layout is malformed and indexing the gripper would read into the padding, so
        raise rather than corrupt a neighbouring dim.
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
        arm = payload.expected_length()
        if arm > source.width:
            raise ValueError(
                f"unified payload length {arm} does not fit within "
                f"the buffer width {source.width}: the gripper would index past the action"
            )
        # The gripper is the last dim of the arm payload.
        _threshold_gripper(out, gripper_index=arm - 1, target=self.target, cutoff=self.cutoff)
        return out


__all__ = ["GripperThresholdAdapter", "UnifiedGripperThresholdAdapter"]
