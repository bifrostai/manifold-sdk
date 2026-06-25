"""Per-episode runtime state, threaded beside the frozen pipeline value.

Most adaptation is stateless: an adapter maps one observation to another with no
memory. Some is not — a frame-history buffer that stacks the last N camera frames
into a clip carries state across steps within an episode. That state must not
live on the `Pipeline`, which is frozen config a caller assembles once and may
share. So a separate `PipelineState` is threaded into the `apply_*` operations,
and a stateful adapter is handed only its own private slice of it.

State is keyed per `(lane, slice)`: a **lane** is one independently-resettable
rollout (vectorized envs and parallel workers each carry their own), and a
**slice** is an adapter's `state_key` label naming an adapter-private mutable
bag. See `LaneKey` and `slice_for` for the precise semantics.

A fresh `PipelineState` lives with a pairing, not the shared policy server:
buffering frames on a server shared across concurrent workers would interleave
their histories and corrupt them silently. The caller (the serving loop, or
`verify`) owns the object and resets a lane on that lane's episode boundary.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any

# A lane identifies one independently-resettable rollout. It collapses the
# (connection, lane) pair into one opaque key; in-process it is a plain id.
LaneKey = Hashable

# The single implicit lane an in-process pairing uses.
DEFAULT_LANE: LaneKey = 0


@dataclass
class PipelineState:
    """Mutable per-episode state, keyed per `(lane, slice)`.

    A slice is owned by whichever adapter declares the matching `state_key`; its
    value is an adapter-private mutable dict the adapter reads and writes across
    steps. The state object itself is caller-owned and threaded into `apply_*`;
    the pipeline never holds it, so the pipeline value stays frozen.
    """

    _slots: dict[tuple[LaneKey, str], dict[str, Any]] = field(default_factory=dict)

    def slice_for(self, *, lane: LaneKey, key: str) -> dict[str, Any]:
        """The adapter-private bag for `(lane, key)`, created empty on first use.

        Returns the same dict object on every call for a given `(lane, key)`, so
        an adapter's mutations to it persist across steps until the lane is reset.
        """
        return self._slots.setdefault((lane, key), {})

    def reset_lane(self, lane: LaneKey = DEFAULT_LANE) -> None:
        """Drop every slice belonging to `lane`, leaving other lanes untouched.

        Called at a lane's episode boundary: the next `slice_for` on that lane
        starts from an empty bag, so a frame-history buffer rebuilds from scratch
        (repeat-padding its first frame again) rather than carrying the previous
        episode's tail into the new one.
        """
        self._slots = {k: v for k, v in self._slots.items() if k[0] != lane}


__all__ = ["DEFAULT_LANE", "LaneKey", "PipelineState"]
