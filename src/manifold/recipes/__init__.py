"""Opinionated, swappable pairing strategies built on the core (which imports nothing here)."""

from __future__ import annotations

from manifold.recipes.dispatch import assert_shared_profile, multi_pairing_pipeline
from manifold.recipes.inspect import Recorder, describe, dump, load
from manifold.recipes.lerobot import SignatureSuggestion, from_lerobot_checkpoint
from manifold.recipes.pairing import Pairing, read_pairing
from manifold.recipes.resolve import resolve
from manifold.recipes.serving import (
    BenchmarkResult,
    ChunkEndpoint,
    EpisodeRecord,
    OpenLoopChunkQueue,
    PairingRejected,
    PolicyProfile,
    StepResult,
    evaluate,
    launch_server,
    run_benchmark,
    run_episodes,
    run_sharded_benchmark,
    serve,
    write_rollup,
)
from manifold.recipes.sharding import EpisodeCursor, shard_episode_ids

__all__ = [
    "BenchmarkResult",
    "ChunkEndpoint",
    "EpisodeCursor",
    "EpisodeRecord",
    "OpenLoopChunkQueue",
    "Pairing",
    "PairingRejected",
    "PolicyProfile",
    "Recorder",
    "SignatureSuggestion",
    "StepResult",
    "assert_shared_profile",
    "describe",
    "dump",
    "evaluate",
    "from_lerobot_checkpoint",
    "launch_server",
    "load",
    "multi_pairing_pipeline",
    "read_pairing",
    "resolve",
    "run_benchmark",
    "run_episodes",
    "run_sharded_benchmark",
    "serve",
    "shard_episode_ids",
    "write_rollup",
]
