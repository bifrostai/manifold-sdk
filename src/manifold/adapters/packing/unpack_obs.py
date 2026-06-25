"""Read a simulator's raw env dict into an SDK `Observation`.

`UnpackToObservation` is the benchmark-side mirror of `PackToNativeLayout` (ADR-0001, decision 4):
where pack scatters a channel-typed `Observation` out into a checkpoint's flat tensor
dict, this gathers a simulator's flat native env dict back into a channel-typed
`Observation`. It is the existing action-side `unpack` generalized — that adapter loops
a layout's entries, each naming a different native key, and concatenates them into one
`Action` vector (a multi-source gather); this loops the same entries but routes each
into the matching `Observation` channel, gathering the entries that target the same
state role into one state vector. No structural op is added; the only addition is
the per-entry `TargetChannel` tag identifying the destination channel.

`from_spec` is the env `NativeLayout` (the simulator's raw wire form) and `to_spec` is
`ObservationSpace` (the channel aggregate the convention chain continues from). Like
the other packing adapters it does not apply convention math (frame rebase, gripper
polarity, quat reorder are phase-one convention adapters that run after this on the
resulting `Observation`) and is author-inserted, never resolved.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel

from manifold.adapters.packing._ops import apply_entry
from manifold.core.adapter import ObservationAdapter
from manifold.core.native_layout import ChannelKind, LayoutEntry, NativeLayout
from manifold.core.observation_space import ObservationSpace
from manifold.core.values import Observation


def _env_value(env_dict: Any, entry: LayoutEntry) -> np.ndarray:
    """Read one entry's source env-dict key as a numpy array.

    `source_name` is the env-dict key (the inverse of pack, where it was an SDK
    channel). Raises a clear error if the key is absent — a data-level mismatch the
    benchmark-side `verify` surfaces, mirroring pack's missing-channel error.
    """
    name = entry.source_name
    if name is None:
        raise ValueError(f"env layout entry {entry.key!r} requires a source_name (the env key)")
    if name not in env_dict:
        present = sorted(env_dict.keys())
        raise KeyError(
            f"the env layout requires env-dict key {name!r}, but the raw observation does "
            f"not carry it (present keys: {present}). The runner must supply it under this "
            f"exact key."
        )
    return np.asarray(env_dict[name])


def _gather(parts: list[tuple[int, int, np.ndarray]]) -> np.ndarray:
    """Concatenate the per-channel pieces in (order, declaration-index) order.

    This is the multi-source gather at the crux of ADR-0001, decision 4: several env keys
    (`eef_pos`, `eef_quat`, `gripper_qpos`) concatenate into one state vector. A
    single-piece channel passes through as itself (reshaped to 1-D for a state role).
    """
    ordered = sorted(parts, key=lambda part: (part[0], part[1]))
    flats = [np.asarray(vector).reshape(-1) for _, _, vector in ordered]
    return np.concatenate(flats) if flats else np.zeros(0, dtype=np.float32)


def _require_name(name: str | None, entry: LayoutEntry, label: str) -> str:
    if name is None:
        raise ValueError(f"{label} target entry {entry.key!r} requires a target.name")
    return name


class UnpackToObservation(ObservationAdapter):
    """Apply an env `NativeLayout` to a raw env dict, producing an SDK `Observation`.

    Each layout entry must carry a `target` (`TargetChannel`): its `source`/
    `source_name` is the ENV-dict key to read, its `ops` reshape that value
    (rename/cast/slice/assemble — the same `_ops` applier pack and unpack share), and
    its `target` routes the result into the `Observation`:

    - `ChannelKind.CAMERA` ⇒ `obs.sensors[target.name]` — a renamed (and optionally
      recast) image key.
    - `ChannelKind.STATE` ⇒ `obs.state[target.name]` — entries targeting the same
      role are GATHERED (concatenated) in `target.order` then declaration order, so
      separate `eef_pos`/`eef_quat`/`gripper_qpos` env keys become one `ee_pose`.
    - `ChannelKind.INSTRUCTION` ⇒ `obs.instruction` — the language string, read
      verbatim from the env dict (no ops).

    `from_spec` is the env `NativeLayout`; `to_spec` is `ObservationSpace`, where the
    benchmark's convention chain continues. Stateless and structural only.
    """

    from_spec: ClassVar[type[BaseModel]] = NativeLayout
    to_spec: ClassVar[type[BaseModel]] = ObservationSpace
    lossless: ClassVar[bool] = True

    def __init__(self, layout: NativeLayout, observation_space: ObservationSpace) -> None:
        self.layout = layout
        self.observation_space = observation_space

    def applies(self, source: BaseModel) -> bool:
        """True for the env `NativeLayout` this adapter was built to read."""
        return isinstance(source, NativeLayout) and source == self.layout

    def produce(self, source: BaseModel) -> ObservationSpace:
        """The channel `ObservationSpace` the benchmark's convention chain continues from."""
        if not isinstance(source, NativeLayout):
            raise TypeError("UnpackToObservation transforms a NativeLayout source only")
        return self.observation_space

    def adapt(self, env_dict: Any, /, *, source: BaseModel) -> Observation:
        """Gather the raw env dict into a channel-typed `Observation` per the layout.

        State entries targeting the same role are concatenated in declared gather
        order into one state vector (matching the hand-coded `_observation`/`_ee_pose`
        this replaces); camera and instruction entries route to their own slots.
        """
        sensors: dict[str, np.ndarray] = {}
        # Role -> list of (order, declaration_index, vector) to gather into one channel.
        state_parts: dict[str, list[tuple[int, int, np.ndarray]]] = {}
        instruction: str | None = None

        for index, entry in enumerate(self.layout.entries):
            target = entry.target
            if target is None:
                raise ValueError(
                    f"UnpackToObservation requires every layout entry to declare a target "
                    f"channel; entry {entry.key!r} has none"
                )
            if target.kind is ChannelKind.INSTRUCTION:
                if entry.ops:
                    raise ValueError(f"INSTRUCTION target entry {entry.key!r} does not take ops")
                value = env_dict.get(entry.source_name) if entry.source_name else None
                instruction = "" if value is None else str(value)
                continue

            array = _env_value(env_dict, entry)
            emitted = apply_entry(array, entry)
            # An env->obs entry routes to exactly one Observation slot (one camera, or
            # one contribution to one gathered state role), so a fan-out op (`Split`,
            # which emits several keys) has no destination here — the gather is
            # many-keys-into-one, never one-into-many. Reject it with a clear message
            # rather than let the single-value unpack below raise an opaque ValueError.
            if len(emitted) != 1:
                raise ValueError(
                    f"env layout entry {entry.key!r} emits {len(emitted)} keys (a Split "
                    f"fan-out); an env->Observation entry must route to exactly one channel. "
                    f"Use separate entries gathered by target, not a Split."
                )
            (produced,) = emitted.values()
            if target.kind is ChannelKind.CAMERA:
                name = _require_name(target.name, entry, "camera")
                sensors[name] = produced
            else:  # ChannelKind.STATE
                name = _require_name(target.name, entry, "state")
                state_parts.setdefault(name, []).append((target.order, index, produced))

        state = {role: _gather(parts) for role, parts in state_parts.items()}
        return Observation(state=state, sensors=sensors, instruction=instruction)


__all__ = ["UnpackToObservation"]
