"""Observability adapters: identity taps that observe the data at a seam.

A `tap` is a lossless identity on spec and data whose only effect is to record a
per-field summary of what flows past. Use it to inspect the data during the
empirical make-it-work loop (ADR-0001, decision 5): drop one on each side of
a pairing to log the command-and-response stream per step.
"""

from __future__ import annotations

from manifold.adapters.observability.tap import (
    ActionTap,
    ObservationTap,
    Sink,
    TapReading,
)

__all__ = ["ActionTap", "ObservationTap", "Sink", "TapReading"]
