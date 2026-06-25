"""Catalog of adapters, one adapter per file.

Grouped by what they convert: `action/` holds action-space adapters,
`observation/` holds observation-form adapters, and `observability/` holds the
identity taps that observe a seam without changing it. Each adapter is a class a
caller instantiates (usually parameterized by its target) and collects into a
`Pipeline`. There is no `ALL` tuple here, unlike the embodiment and benchmark
catalogs: those ship instances, whereas an adapter is aimed at a
target only when constructed.
"""

from __future__ import annotations

from manifold.adapters.observability import ActionTap, ObservationTap
from manifold.adapters.packing import PackToNativeLayout, UnpackFromNativeLayout

__all__ = [
    "ActionTap",
    "ObservationTap",
    "PackToNativeLayout",
    "UnpackFromNativeLayout",
]
