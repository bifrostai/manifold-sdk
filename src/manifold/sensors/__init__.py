"""Catalog of sensors.

A sensor's name and mount are the shared vocabulary a policy and a benchmark
agree on; its resolution is benchmark-specific. So these are constructors keyed
on the canonical name and mount, and each benchmark calls them with its own
shape, rather than fixed instances. (So there is no `ALL` enumeration here,
unlike the embodiment and benchmark catalogs, which ship instances.)
"""

from __future__ import annotations

from manifold.sensors.cameras import agentview, wrist

__all__ = ["agentview", "wrist"]
