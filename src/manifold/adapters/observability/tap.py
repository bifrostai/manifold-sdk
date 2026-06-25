"""`tap` — an identity adapter that observes the data flowing through a seam.

After the shell idiom (Elixir's `tap`, `IO.inspect`, the Unix `tee`): a lossless
identity on *both* spec and data whose only effect is a side effect — it records a
per-field summary of what flows past (shape, dtype, range) and returns the data
completely unchanged. Because it is an adapter it composes: drop one before and
after an adapter to see its effect, or into a misbehaving pairing to read reality
at that seam (ADR-0001, decision 5).

A `tap` on the observation chain and one on the action chain give the *passive*
half of behavioural probing: the command-and-response stream logged per step, so
an inverted gripper is read directly off the finger positions rather than guessed
from a score.

A tap is **spec-identity**: `produce` returns the source spec unchanged. A
resolver's BFS over the spec graph therefore never selects it — it does not
advance the spec, so it does not add a node to explore (and de-duplicates straight
back to where it started). Like a stateful adapter, a tap is author-inserted, not
auto-discovered. It is lossless and `applies` is always True: it can observe any
spec.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel

from manifold.core.adapter import ActionAdapter, ObservationAdapter
from manifold.core.embodiment.spec import ValueSpec
from manifold.core.observation_space import ObservationSpace
from manifold.core.values import Observation

_logger = logging.getLogger("manifold.tap")

# A summary of one array field: its shape, dtype, and value range. The range is
# None for an empty array (no min/max/mean to report).
FieldSummary = dict[str, Any]
# A full tap reading: the label plus a per-field-name summary.
TapReading = dict[str, Any]
Sink = Callable[[TapReading], None]


def _summarize(array: Any) -> FieldSummary:
    """Shape, dtype, and (min, max, mean) of one array — the per-field record."""
    arr = np.asarray(array)
    summary: FieldSummary = {"shape": tuple(int(d) for d in arr.shape), "dtype": str(arr.dtype)}
    if arr.size:
        summary["min"] = float(arr.min())
        summary["max"] = float(arr.max())
        summary["mean"] = float(arr.mean())
    else:
        summary["min"] = summary["max"] = summary["mean"] = None
    return summary


def _default_sink(reading: TapReading) -> None:
    """Log the reading at INFO — the default observation behaviour."""
    _logger.info("tap %r: %s", reading["label"], reading["fields"])


class ObservationTap(ObservationAdapter):
    """Record a per-field summary of an observation, returning it unchanged.

    Summarizes each `state` entry (proprioception by role) and each `sensors`
    entry (camera by name). Spec-identity and data-identity: `produce` returns the
    source and `adapt` returns the same `Observation`. Parameterized by a `label`
    (which seam this taps) and an optional `sink` (defaults to a module logger).
    """

    from_spec: ClassVar[type[BaseModel]] = ObservationSpace
    to_spec: ClassVar[type[BaseModel]] = ObservationSpace
    lossless: ClassVar[bool] = True

    def __init__(self, label: str = "observation", sink: Sink | None = None) -> None:
        self.label = label
        self.sink = sink if sink is not None else _default_sink

    def applies(self, source: BaseModel) -> bool:
        """Always True: a tap can observe any observation spec."""
        return isinstance(source, ObservationSpace)

    def produce(self, source: BaseModel) -> ObservationSpace:
        """The source spec, unchanged — a tap is spec-identity."""
        if not isinstance(source, ObservationSpace):
            raise TypeError("ObservationTap taps an ObservationSpace source only")
        return source

    def adapt(self, observation: Any, *, source: BaseModel) -> Observation:
        """Record the per-field summary, then return the observation unchanged."""
        fields: dict[str, FieldSummary] = {}
        for role, value in observation.state.items():
            fields[f"state.{role}"] = _summarize(value)
        for name, value in observation.sensors.items():
            fields[f"sensors.{name}"] = _summarize(value)
        self.sink({"label": self.label, "fields": fields, "instruction": observation.instruction})
        return observation


class ActionTap(ActionAdapter):
    """Record a summary of an action's `values` array, returning it unchanged.

    Spec-identity and data-identity: `produce` returns the source action space and
    `adapt` returns the same `values`. Parameterized by a `label` and an optional
    `sink` (defaults to a module logger).

    `from_spec`/`to_spec` are the shared `ValueSpec` base, so the tap observes any
    action space (EE, joint, unified). This is why it is author-inserted into
    `apply_action`, never resolved: `_check_action` matches a chain link by *exact*
    type (`type(current) is adapter.from_spec`), which a universal base-typed tap
    would not satisfy mid-chain — but a tap never needs `check` to select it (it is
    spec-identity, so it advances nothing the BFS could discover). It runs inside
    `apply_action`, where dispatch is by `produce`/`adapt`, not by the exact-type
    gate.
    """

    from_spec: ClassVar[type[BaseModel]] = ValueSpec
    to_spec: ClassVar[type[BaseModel]] = ValueSpec
    lossless: ClassVar[bool] = True

    def __init__(self, label: str = "action", sink: Sink | None = None) -> None:
        self.label = label
        self.sink = sink if sink is not None else _default_sink

    def applies(self, source: BaseModel) -> bool:
        """Always True: a tap can observe any action spec."""
        return isinstance(source, ValueSpec)

    def produce(self, source: BaseModel) -> BaseModel:
        """The source spec, unchanged — a tap is spec-identity."""
        if not isinstance(source, ValueSpec):
            raise TypeError("ActionTap taps a ValueSpec (action space) source only")
        return source

    def adapt(self, values: Any, *, source: BaseModel) -> Any:
        """Record the summary of `values`, then return it unchanged."""
        self.sink({"label": self.label, "fields": {"values": _summarize(values)}})
        return values


__all__ = ["ActionTap", "ObservationTap", "Sink", "TapReading"]
