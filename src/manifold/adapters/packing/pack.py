"""Pack an SDK `Observation` into a checkpoint's native tensor dict.

`PackToNativeLayout` is the observation-side packing adapter: the second phase of
the pipeline, where the seam crosses from the channel-typed `Observation` to the
flat `dict[str, np.ndarray]` a checkpoint consumes (ADR-0001). It applies a declared
`NativeLayout` — pulling each entry's source channel from the observation and
running the entry's structural ops (rename, cast, batch/time axis, slice, split,
assemble) — and does not apply convention math, by the codomain rule: rotation/gripper/
frame transforms are phase-one convention adapters that ran before this.

It is the generic adapter the layout drives, so it cannot drift from the layout: a
single implementation applies whatever a profile declares. It is author-inserted,
never resolved — a resolver walks the channel-spec graph, and this advances that
graph off its end (to a `NativeLayout`, not another `ObservationSpace`).
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel

from manifold.adapters.packing._ops import apply_entry
from manifold.core.adapter import ObservationAdapter
from manifold.core.native_layout import NativeLayout, SourceKind
from manifold.core.observation_space import ObservationSpace
from manifold.core.values import Observation


def _channel_value(observation: Observation, source: SourceKind, name: str | None) -> np.ndarray:
    """Pull one numeric channel (a camera or a proprioception value) as a numpy array.

    A camera comes from `obs.sensors[name]`, a proprioception value from
    `obs.state[name]`. The instruction is text, not a tensor, and is handled directly
    by the pack adapter, not here. Raises a clear error if the named channel is absent
    — a data-level mismatch `verify` surfaces as a failed packing check.
    """
    if name is None:
        raise ValueError(f"layout source {source} requires a source_name")
    if source is SourceKind.CAMERA:
        bag, label = observation.sensors, "camera"
    else:
        bag, label = observation.state, "state"
    if name not in bag:
        present = sorted(bag.keys())
        raise KeyError(
            f"the native layout requires {label} channel {name!r}, but the observation "
            f"does not carry it (present {label} channels: {present}). This channel is not "
            f"modeled in the policy's proprioception spec, so verify could only synthesize "
            f"it — the real runner must supply it under this exact key."
        )
    return np.asarray(bag[name])


class PackToNativeLayout(ObservationAdapter):
    """Apply a `NativeLayout` to an `Observation`, producing the model's tensor dict.

    `from_spec` is `ObservationSpace` (the policy's consumed channel form, the phase
    boundary) and `to_spec` is `NativeLayout` (the checkpoint's wire form). `produce`
    returns the declared layout — the produced spec does not depend on the source
    channels' values, only on the layout the profile fixed. `adapt` runs it.

    It is lossless in the structural sense the seam requires: it renames and
    reshapes, discarding no channel data the layout selects. (A `Slice` that drops a
    sub-vector the model never consumes is intentional selection, not loss — the
    dropped slots hold nothing the policy emitted, like a unified buffer's
    padding.) It is stateless: any per-episode buffering (frame history) is a
    phase-one stateful adapter that ran before packing.
    """

    from_spec: ClassVar[type[BaseModel]] = ObservationSpace
    to_spec: ClassVar[type[BaseModel]] = NativeLayout
    lossless: ClassVar[bool] = True

    def __init__(self, layout: NativeLayout) -> None:
        self.layout = layout

    def applies(self, source: BaseModel) -> bool:
        """True for any `ObservationSpace` — the layout declares the channels it needs.

        Packing is the terminal phase, so it applies to whatever channel form the
        chain reaches; whether the observation actually carries the channels the
        layout declares is a data property `verify` exercises, not a spec property the
        static check can decide.
        """
        return isinstance(source, ObservationSpace)

    def produce(self, source: BaseModel) -> NativeLayout:
        """The declared native layout — the model's wire form, fixed by the profile."""
        if not isinstance(source, ObservationSpace):
            raise TypeError("PackToNativeLayout transforms an ObservationSpace source only")
        return self.layout

    def adapt(self, observation: Any, /, *, source: BaseModel) -> dict[str, Any]:
        """Build the native tensor dict by applying each layout entry to its channel.

        The instruction is text, not a tensor: it is packed as a length-1 tuple of the
        string — the form the GR00T-lineage `annotation.*.task_description` key expects
        and that Cosmos unwraps via `str(x[0])`. Structural ops do not apply to it.
        """
        packed: dict[str, Any] = {}
        for entry in self.layout.entries:
            if entry.source is SourceKind.INSTRUCTION:
                if entry.ops:
                    raise ValueError(f"INSTRUCTION layout entry {entry.key!r} does not take ops")
                packed[entry.key] = (observation.instruction or "",)
                continue
            value = _channel_value(observation, entry.source, entry.source_name)
            packed.update(apply_entry(value, entry))
        return packed


__all__ = ["PackToNativeLayout"]
