"""Opinionated, swappable pairing strategies built on the core (which imports nothing here)."""

from __future__ import annotations

from manifold.recipes.dispatch import assert_shared_profile, multi_pairing_pipeline
from manifold.recipes.inspect import Recorder, describe, dump, load
from manifold.recipes.lerobot import SignatureSuggestion, from_lerobot_checkpoint
from manifold.recipes.local import serve as serve_function
from manifold.recipes.pairing import Pairing, read_pairing
from manifold.recipes.resolve import resolve
from manifold.recipes.serving import (
    ActionQueue,
    BenchmarkResult,
    PolicyProfile,
    evaluate,
    launch_server,
    serve,
)

__all__ = [
    "ActionQueue",
    "BenchmarkResult",
    "Pairing",
    "PolicyProfile",
    "Recorder",
    "SignatureSuggestion",
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
    "serve",
    "serve_function",
]
