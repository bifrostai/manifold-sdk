"""Read a checkpoint's raw output chunk back into an SDK `Action`.

`UnpackFromNativeLayout` is the action-side packing adapter — the mirror of
`PackToNativeLayout`. The model emits a raw chunk: either a single array indexed by
step (Cosmos's `chunk[step][:7]`) or a dict of per-key `(batch, time, dim)` tensors
indexed at `[0, step, :]` (RLDX's `action.x`/…). This adapter reads one step out of
that raw output through a declared output `NativeLayout` and assembles the flat SDK
`Action` vector — slice, dtype, and concatenation only. Convention math (gripper
thresholding, rotation re-encoding) is not here: it is a phase-one convention
adapter that runs after this, on the resulting `Action`, toward the benchmark form.

`from_spec` is the output `NativeLayout` (the model's wire output) and `to_spec` is
the policy's action space — so the rest of the action chain (the convention
adapters) continues from `to_spec` toward the benchmark's canonical action space, as
`check_compatibility`/`verify` expect. Like the pack side it is author-inserted, not
resolved.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel

from manifold.adapters.packing._ops import apply_entry
from manifold.core.adapter import ActionAdapter
from manifold.core.embodiment.spec import ValueSpec
from manifold.core.native_layout import NativeLayout, SourceKind
from manifold.core.values import Action


def _output_value(raw_output: Any, source: SourceKind, name: str | None, step: int) -> np.ndarray:
    """Select the step-`step` slice of one raw-output piece as a numpy array.

    A named entry (`SourceKind.STATE`, used here for "an output dict key") indexes
    `raw_output[name]`; an unnamed entry indexes the chunk array itself. In both
    cases the step axis is taken first: a `(batch, time, dim)` dict tensor reduces to
    its `[0, step]` row, a `(chunk, dim)` array to its `[step]` row — leaving the
    feature axis for the entry's `Slice`/`Assemble` ops to window.
    """
    if name is not None:
        tensor = np.asarray(raw_output[name])
        # Per-key tensors are (batch, time, dim); take batch 0 and the requested step.
        if tensor.ndim >= 3:
            return tensor[0, step]
        if tensor.ndim == 2:
            return tensor[step]
        return tensor
    array = np.asarray(raw_output)
    return array[step]


class UnpackFromNativeLayout(ActionAdapter):
    """Read one step of a raw model-output chunk into an SDK `Action`.

    `from_spec` is the output `NativeLayout` describing the model's raw chunk;
    `to_spec` is the policy's action space (a `ValueSpec` subclass), which `produce`
    returns so the convention adapters that follow can continue the chain. The
    output layout's entries identify the chunk's keys (a `SourceKind.STATE` entry is a
    dict key in the raw output; the instruction source does not apply on the action
    side) and the structural ops that extract each component; `adapt` concatenates
    them, in entry order, into the flat per-step action vector.

    It is lossless in the structural sense: it selects and concatenates the chunk's
    real predicted dims; any value the layout drops is a dim the policy's action
    space does not carry. Stateless — the chunk queue (which step to read) is
    serve-loop control flow, not pipeline state.
    """

    from_spec: ClassVar[type[BaseModel]] = NativeLayout
    to_spec: ClassVar[type[BaseModel]] = ValueSpec
    lossless: ClassVar[bool] = True

    def __init__(self, layout: NativeLayout, action_space: ValueSpec) -> None:
        self.layout = layout
        self.action_space = action_space

    def applies(self, source: BaseModel) -> bool:
        """True for the output `NativeLayout` this adapter was built to read."""
        return isinstance(source, NativeLayout) and source == self.layout

    def produce(self, source: BaseModel) -> ValueSpec:
        """The policy's action space — where the convention chain continues from."""
        if not isinstance(source, NativeLayout):
            raise TypeError("UnpackFromNativeLayout transforms a NativeLayout source only")
        return self.action_space

    def adapt(self, raw_output: Any, /, *, source: BaseModel, step: int = 0) -> Action:
        """Read step `step` of the raw chunk into a flat `Action`, per the layout.

        Each entry identifies a piece of the raw output (a dict-keyed tensor, or the chunk
        array itself); the per-entry results are concatenated in declaration order
        into the per-step action vector. `step` indexes the chunk's step axis.
        """
        pieces: list[np.ndarray] = []
        for entry in self.layout.entries:
            value = _output_value(raw_output, entry.source, entry.source_name, step)
            produced = apply_entry(value, entry)
            pieces.extend(np.asarray(part).reshape(-1) for part in produced.values())
        flat = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        # Validate the unpacked width against the policy's action space before
        # handing it on. The convention `action` chain that follows is often empty,
        # so a mis-windowed layout that produces the wrong number of dims would
        # otherwise flow through as a malformed Action and surface far downstream.
        # `validate_value` is a structural length check; fail loudly here instead.
        self.action_space.validate_value(flat)
        return Action.from_array(flat)


__all__ = ["UnpackFromNativeLayout"]
