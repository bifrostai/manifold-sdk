"""What the episode loop tells a benchmark about the episode it is driving.

The loop owns the lifecycle: which episode began, which action produced the
observation now current, and - the part a caller cannot see - whether a step
was the episode's last, since an episode ends either because the driver reported
it or because `max_steps` ran out. An `EpisodeRecorder` is where a benchmark hangs
its own per-step work off those points, instead of re-deriving them inside the
`reset` and `step` callables it passes in.

The replay log is what this exists for: a benchmark's recorder opens an episode's
log, appends a frame per step, and closes it (ADR 0003). Nothing here depends on
that, so a recorder that tallies steps or writes a CSV is as valid.

A recorder reads its own simulator. The loop passes the action and how the step
went, not an observation: what a replay needs is the sim state behind the
observation - the model and data handles, the goal predicates - and only the
benchmark process can reach those.

Not to be confused with `recipes.inspect.Recorder`, which accumulates
observations and actions in memory so two runs can be diffed offline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from manifold.core.values import Action


class EpisodeRecorder(Protocol):
    """Per-episode hooks the benchmark loop calls, in this order.

    `begin`, then `record` per step, then `end`, once per episode. A recorder is
    reused across the episodes of a shard, so `begin` is where per-episode state
    is set up and `end` is where it is let go.
    """

    def begin(self, episode_id: int) -> None:
        """One episode is starting, seeded by its global id.

        Called after `reset` returns and before the first observation reaches the
        policy, so the simulator is at the episode's initial state: what a recorder
        reads here is the episode's opening frame.

        `episode_id` is the id the run assigned, not this shard's position in its
        own list - two shards writing a file per episode would otherwise collide.
        """

    def record(self, action: Action, *, success: bool, done: bool) -> None:
        """One step happened: `action` produced the state the simulator is in now.

        `done` is true on the episode's last step whatever ended it - the driver
        reporting success, the driver reporting done, or the step budget running
        out. A recorder is notified that the episode ended rather than watching
        `max_steps` itself, since the budget is not otherwise visible to it.
        """

    def end(self) -> None:
        """The episode is over; release what `begin` set up.

        Called on every exit path, a raising `step` or `begin` included, so whatever
        `begin` opened is closed exactly once per episode. `begin` raising part-way
        is why this runs even when no episode got under way.
        """


class _NullRecorder:
    """The default recorder: every hook is a no-op."""

    def begin(self, episode_id: int) -> None:
        """Ignore the episode."""

    def record(self, action: Action, *, success: bool, done: bool) -> None:
        """Ignore the step."""

    def end(self) -> None:
        """Nothing was opened."""


# The default for every loop that takes a recorder. One shared instance, since
# `_NullRecorder` is stateless.
NO_RECORDER: EpisodeRecorder = _NullRecorder()


__all__ = [
    "NO_RECORDER",
    "EpisodeRecorder",
]
