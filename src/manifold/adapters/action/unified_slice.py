"""Slice a unified action buffer down to its end-effector payload.

A multi-embodiment base model (pi0.5 at width 32, RDT-1B at width 128) emits a
fixed-width buffer per step: the real action occupies the leading slots and the
rest is zero padding. This adapter drops the padding, producing the embodiment's
action. It is lossless — the padding contains nothing to preserve.

It bridges a `UnifiedActionSpace` whose payload is an `EEActionSpace`; a joint
payload would need its own sibling.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel

from manifold.core.adapter import ActionAdapter
from manifold.core.embodiment import EEActionSpace, UnifiedActionSpace


class UnifiedSliceAdapter(ActionAdapter):
    """Drop the padding of a unified buffer, leaving its end-effector payload."""

    from_spec: ClassVar[type[BaseModel]] = UnifiedActionSpace
    to_spec: ClassVar[type[BaseModel]] = EEActionSpace
    lossless: ClassVar[bool] = True

    def applies(self, source: BaseModel) -> bool:
        """True when `source` is a unified buffer carrying an end-effector payload."""
        return isinstance(source, UnifiedActionSpace) and isinstance(source.payload, EEActionSpace)

    def produce(self, source: BaseModel) -> EEActionSpace:
        """The end-effector payload the buffer carries."""
        return self._unified(source)[1]

    def adapt(self, values: Any, *, source: BaseModel) -> list[float]:
        """Keep the leading payload slots of each step; drop the padding."""
        unified, payload = self._unified(source)
        buffer = [float(v) for v in values]
        unified.validate_value(buffer)  # full width * chunk length, per the unified spec
        chunk = payload.chunk_size
        per_step = payload.expected_length() // chunk
        out: list[float] = []
        for step in range(chunk):
            start = step * unified.width
            out.extend(buffer[start : start + per_step])
        return out

    @staticmethod
    def _unified(source: BaseModel) -> tuple[UnifiedActionSpace, EEActionSpace]:
        if not isinstance(source, UnifiedActionSpace) or not isinstance(
            source.payload, EEActionSpace
        ):
            raise TypeError(
                "UnifiedSliceAdapter needs a UnifiedActionSpace with an EEActionSpace payload"
            )
        return source, source.payload


__all__ = ["UnifiedSliceAdapter"]
