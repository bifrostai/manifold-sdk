"""Binarize one dimension of a UnifiedActionSpace per-step block.

Some whole-body policies emit a control-mode or mode-select dimension whose
continuous output must be snapped to a discrete low/high value before the
benchmark's env can consume it. `DiscreteBinarize` applies that snap to a
single named dimension inside a `UnifiedActionSpace`'s per-step width-D vector.

The adapter is parameterized by `dim_index` (0-based within the width-D step),
`threshold` (default 0.5), and `low`/`high` (default -1.0/+1.0). A value
strictly above the threshold maps to `high`; at or below maps to `low`. All
other dimensions in the buffer pass through unchanged.

The mapping is lossy — continuous magnitude is discarded — so the adapter
declares `lossless = False` and the caller opts in explicitly, consistent with
`GripperThresholdAdapter`.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel

from manifold.core.adapter import ActionAdapter
from manifold.core.embodiment import UnifiedActionSpace


class DiscreteBinarize(ActionAdapter):
    """Binarize one dimension of a `UnifiedActionSpace` per-step vector.

    Value-only, single-pass, author-inserted: `produce` returns the source spec
    unchanged, so `applies` stays True after it runs and the adapter never reaches a
    spec fixpoint. It is therefore safe only under the single-pass `_check_action`
    walk (which visits each chain position once) and must be placed explicitly in the
    chain, not discovered by a re-applying resolver, which would loop here.

    `dim_index` is the 0-based index within the per-step width-D vector to binarize;
    it must satisfy ``0 <= dim_index < source.width``, or `applies` is False.
    `threshold` (default 0.5) is the decision boundary: values strictly above map to
    `high` (the "open"/"on" side, default +1.0); values at or below map to `low`
    (the "closed"/"off" side, default -1.0).
    """

    from_spec: ClassVar[type[BaseModel]] = UnifiedActionSpace
    to_spec: ClassVar[type[BaseModel]] = UnifiedActionSpace
    lossless: ClassVar[bool] = False

    def __init__(
        self,
        dim_index: int,
        threshold: float = 0.5,
        low: float = -1.0,
        high: float = 1.0,
    ) -> None:
        self.dim_index = dim_index
        self.threshold = threshold
        self.low = low
        self.high = high

    def applies(self, source: BaseModel) -> bool:
        """True when `source` is a `UnifiedActionSpace` and `dim_index` is in range.

        Stays True after this adapter runs (see the class docstring): place at most
        one `DiscreteBinarize` per dim per chain.
        """
        return isinstance(source, UnifiedActionSpace) and 0 <= self.dim_index < source.width

    def produce(self, source: BaseModel) -> UnifiedActionSpace:
        """The source spec unchanged — binarizing a dim does not alter the shape."""
        if not isinstance(source, UnifiedActionSpace):
            raise TypeError("DiscreteBinarize transforms UnifiedActionSpace only")
        return source

    def adapt(self, values: Any, *, source: BaseModel) -> list[float]:
        """Binarize `dim_index` of each per-step block; leave all other dims unchanged."""
        if not isinstance(source, UnifiedActionSpace):
            raise TypeError("DiscreteBinarize needs a UnifiedActionSpace source")
        out = [float(v) for v in values]
        source.validate_value(out)
        chunk_size = source.payload.chunk_size if source.payload is not None else 1
        for step in range(chunk_size):
            idx = step * source.width + self.dim_index
            out[idx] = self.high if out[idx] > self.threshold else self.low
        return out


__all__ = ["DiscreteBinarize"]
