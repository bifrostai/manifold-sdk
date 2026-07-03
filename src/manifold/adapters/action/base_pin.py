"""Widen a 7-D arm action to a unified whole-body buffer with the base pinned.

Some checkpoints predict only the manipulation half of a whole-body embodiment —
a 7-D end-effector arm action (pos3 + rotation + gripper1) — while the benchmark
env expects the full width (the arm DOFs plus the mobile-base / control-mode
DOFs). NVIDIA's Cosmos-Policy RoboCasa eval is one such case: the policy was
trained on 7-D arm actions, but the RoboCasa env's action is 12-D, so the eval
appends a constant `[0, 0, 0, 0, -1]` (the trailing 5 whole-body DOFs) to every
predicted step to pin the mobile base — the arm acts, the base stays put.

This adapter encodes exactly that widening as a typed action transform. It bridges
an `EEActionSpace` (the policy's 7-D arm payload) to the `UnifiedActionSpace` the
benchmark declares (`width=12`, the same `EEActionSpace` as its payload), so the
action chain reaches the benchmark's canonical action space and the pairing gates
COMPATIBLE_VIA_PIPELINE.

It is not lossless: the constant tail is invented, not a representation of the
policy's emitted action — the policy does not emit a base command at all, so the
widening cannot be inverted. A caller opts in by supplying it (the RoboCasa
pairing does), exactly as with a resize.

Unlike `UnifiedSliceAdapter` (the inverse — a base model that emits a wide buffer
and a slice drops the padding), here the policy emits the narrow arm action and
this adapter widens it to the full whole-body width by pinning the base.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel

from manifold.core.adapter import ActionAdapter
from manifold.core.embodiment import EEActionSpace, UnifiedActionSpace

# The trailing whole-body DOFs a 7-D arm action lacks, pinned to hold the mobile
# base still: 4 base-motion DOFs at 0 plus a control-mode value of -1. Appended to
# each per-step arm block to reach the env's 12-D whole-body action.
_BASE_PIN: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, -1.0)


class BasePinWiden(ActionAdapter):
    """Widen a 7-D arm `EEActionSpace` to a `width`-D unified buffer, pinning the base.

    Each per-step arm block is followed by a constant base-pin tail so the emitted
    action reaches the benchmark's whole-body width. `width` defaults to 12 (the
    RoboCasa PandaOmron whole-body action); the tail length is `width - per_step`.
    """

    from_spec: ClassVar[type[BaseModel]] = EEActionSpace
    to_spec: ClassVar[type[BaseModel]] = UnifiedActionSpace
    lossless: ClassVar[bool] = False

    def __init__(self, width: int = 12, pin: tuple[float, ...] = _BASE_PIN) -> None:
        self.width = width
        self.pin = tuple(float(v) for v in pin)

    def applies(self, source: BaseModel) -> bool:
        """True when `source` is an arm action narrower than the target width.

        The per-step arm length must leave room for the pin tail (and the pin must
        fill exactly that room), so the widening is well-defined.
        """
        if not isinstance(source, EEActionSpace):
            return False
        per_step = source.expected_length() // source.chunk_size
        return per_step < self.width and per_step + len(self.pin) == self.width

    def produce(self, source: BaseModel) -> UnifiedActionSpace:
        """The unified whole-body buffer carrying this arm action unchanged as its payload."""
        if not isinstance(source, EEActionSpace):
            raise TypeError("BasePinWiden transforms EEActionSpace only")
        return UnifiedActionSpace(width=self.width, payload=source)

    def adapt(self, values: Any, *, source: BaseModel) -> list[float]:
        """Append the base-pin tail to each per-step arm block, reaching `width` per step."""
        if not isinstance(source, EEActionSpace):
            raise TypeError("BasePinWiden needs an EEActionSpace source")
        buffer = [float(v) for v in values]
        source.validate_value(buffer)  # full per-step length * chunk, per the arm spec
        per_step = source.expected_length() // source.chunk_size
        out: list[float] = []
        for step in range(source.chunk_size):
            out.extend(buffer[step * per_step : (step + 1) * per_step])
            out.extend(self.pin)
        return out


__all__ = ["BasePinWiden"]
