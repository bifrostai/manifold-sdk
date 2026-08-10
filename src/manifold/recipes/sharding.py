"""Episode-sharding and cursor helpers for benchmark orchestration.

The episode-id sequencing and disjoint shard partition are run-orchestration
concerns — the same layer as `run_benchmark` — so they live here in
`manifold.recipes`, not in a benchmark runner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def shard_episode_ids(episodes: int, num_shards: int, shard_index: int) -> list[int]:
    """Return the global episode ids this shard owns, validating the shard flags.

    Partitions the global episode set ``[0, episodes)`` disjointly across shards:
    shard ``i`` owns every global index where ``idx % num_shards == shard_index``.
    This is a perfect disjoint cover — every index belongs to exactly one shard, and
    the union of all shards is the full range — so several shards against one server
    drive non-overlapping, reproducible subsets of the eval (each id seeds its env).

    Raises:
        ValueError: If `num_shards < 1` or `shard_index` is outside
            ``[0, num_shards)``. A catchable misuse, consistent with the rest of the
            recipes layer (a CLI wrapper translates it to its own exit code) rather
            than a `SystemExit` that a library caller cannot intercept.
    """
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not (0 <= shard_index < num_shards):
        raise ValueError(f"shard_index must be in [0, {num_shards}); got {shard_index}")
    return [idx for idx in range(episodes) if idx % num_shards == shard_index]


class EpisodeCursor:
    """Yields the global episode id for each ``reset``, used by benchmark runners.

    Every runner seeds its env with a global episode id so shards drive disjoint,
    reproducible subsets. The id sequencing is identical across runners, so it lives
    here in the recipes layer: it walks the caller's ordered ids, handing out one per
    reset. What each runner does with the id (seed numpy, seed the env, pick a grid
    cell) stays in the runner.
    """

    def __init__(self, episode_ids: Sequence[int]) -> None:
        # The ordered global ids this runner owns, one handed out per reset.
        self._episode_ids = list(episode_ids)
        self._cursor = 0

    def next_id(self) -> int:
        """The global episode id for this reset, advancing the cursor."""
        global_id = self._episode_ids[self._cursor]
        self._cursor += 1
        return global_id


__all__ = [
    "EpisodeCursor",
    "shard_episode_ids",
]
