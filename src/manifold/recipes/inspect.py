"""Inspect and record the runtime stream at a seam — the empirical instruments.

This is the `recipes/`-level companion to the `tap` adapter (ADR-0001, decision
5). Where a tap observes *inline* in
a pipeline, these are offline tools over the same per-step values:

- `describe(observation_or_action)` returns a readable table of each field's
  shape, dtype, and value range — what a human reads at the seam.
- `Recorder` appends `Observation`/`Action` values across a run, and `dump`/
  `load` persist and replay that stream, so two runs can be diffed offline.

These are pure functions and a small container over plain values; they do not
decide strategy or hold adapters, so they belong in `recipes/`, not core.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from manifold.core.values import Action, Observation


def _field_row(name: str, array: Any) -> str:
    """One aligned table row for an array field: name, shape, dtype, range."""
    arr = np.asarray(array)
    shape = "x".join(str(int(d)) for d in arr.shape) if arr.ndim else "scalar"
    if arr.size:
        rng = f"[{float(arr.min()):.4g}, {float(arr.max()):.4g}] mean {float(arr.mean()):.4g}"
    else:
        rng = "empty"
    return f"  {name:<24} {shape:<16} {arr.dtype!s:<10} {rng}"


def describe(value: Observation | Action) -> str:
    """A readable, multi-line table of each field in an observation or action.

    For an `Observation`: each `state` entry (proprioception by role), each
    `sensors` entry (camera by name), and the instruction. For an `Action`: the
    `values` array. Each row is field name -> shape / dtype / range. Pure: it
    reads the value and returns a string; the caller prints it.
    """
    if isinstance(value, Observation):
        lines = ["Observation"]
        for role, arr in value.state.items():
            lines.append(_field_row(f"state.{role}", arr))
        for name, arr in value.sensors.items():
            lines.append(_field_row(f"sensors.{name}", arr))
        instruction = "None" if value.instruction is None else repr(value.instruction)
        lines.append(f"  {'instruction':<24} {instruction}")
        return "\n".join(lines)
    if isinstance(value, Action):
        return "Action\n" + _field_row("values", value.values)
    raise TypeError(f"describe expects an Observation or Action, got {type(value).__name__}")


@dataclass
class Recorder:
    """Append `Observation`/`Action` values across a run, for offline replay.

    Two parallel ordered logs — observations and actions — each replaying in its
    own append order (the obs/act interleaving across the two is not preserved,
    which the diff-two-runs use case does not need). `dump` writes the stream to a
    `.npz`; `load` reconstructs a Recorder from one. This is enough to diff two
    runs offline: record both, dump, and compare.
    """

    observations: list[Observation] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)

    def record_observation(self, observation: Observation) -> None:
        """Append one observation to the stream."""
        self.observations.append(observation)

    def record_action(self, action: Action) -> None:
        """Append one action to the stream."""
        self.actions.append(action)

    def dump(self, path: str | Path) -> None:
        """Persist the recorded stream to a `.npz` at `path`.

        Each array field is stored under a flat, indexed key (`obs{i}.state.{role}`,
        `obs{i}.sensors.{name}`, `act{i}.values`) alongside a JSON manifest that
        records the per-frame structure (which roles, which sensors, the
        instruction) so `load` can rebuild the exact `Observation`/`Action` objects.
        """
        arrays: dict[str, np.ndarray] = {}
        obs_manifest: list[dict[str, Any]] = []
        for i, obs in enumerate(self.observations):
            entry: dict[str, Any] = {"state": [], "sensors": [], "instruction": obs.instruction}
            for role, arr in obs.state.items():
                arrays[f"obs{i}.state.{role}"] = np.asarray(arr)
                entry["state"].append(role)
            for name, arr in obs.sensors.items():
                arrays[f"obs{i}.sensors.{name}"] = np.asarray(arr)
                entry["sensors"].append(name)
            obs_manifest.append(entry)
        for i, act in enumerate(self.actions):
            arrays[f"act{i}.values"] = np.asarray(act.values)
        manifest = {"observations": obs_manifest, "n_actions": len(self.actions)}
        arrays["_manifest"] = np.frombuffer(json.dumps(manifest).encode(), dtype=np.uint8)
        # `ty` matches a `**dict[str, ndarray]` against savez's typed `allow_pickle`
        # keyword (a stub imprecision); the keys are real array fields, never that.
        np.savez(Path(path), **arrays)  # ty: ignore[invalid-argument-type]

    @classmethod
    def load(cls, path: str | Path) -> Recorder:
        """Reconstruct a Recorder from a `.npz` written by `dump`."""
        with np.load(Path(path), allow_pickle=False) as data:
            manifest = json.loads(bytes(data["_manifest"]).decode())
            observations: list[Observation] = []
            for i, entry in enumerate(manifest["observations"]):
                state = {role: data[f"obs{i}.state.{role}"] for role in entry["state"]}
                sensors = {name: data[f"obs{i}.sensors.{name}"] for name in entry["sensors"]}
                observations.append(
                    Observation(state=state, sensors=sensors, instruction=entry["instruction"])
                )
            actions = [Action(values=data[f"act{i}.values"]) for i in range(manifest["n_actions"])]
        return cls(observations=observations, actions=actions)


def dump(recorder: Recorder, path: str | Path) -> None:
    """Free-function form of `Recorder.dump`."""
    recorder.dump(path)


def load(path: str | Path) -> Recorder:
    """Free-function form of `Recorder.load`."""
    return Recorder.load(path)


__all__ = ["Recorder", "describe", "dump", "load"]
