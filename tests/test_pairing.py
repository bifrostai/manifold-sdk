"""Tests for the pairing contract reader in recipes.pairing.

`read_pairing` reads the PROFILE/BENCHMARK/PIPELINE trio off any module into a typed
`Pairing` — namespace-agnostic, so a fake module stands in for a real finetune module.
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

from manifold.recipes import Pairing, read_pairing


def _module(**attrs: object) -> ModuleType:
    """A stand-in module: SimpleNamespace exposes attrs via hasattr/getattr like a module."""
    return cast(ModuleType, SimpleNamespace(__name__="fake.pairing.module", **attrs))


def test_read_pairing_reads_the_trio_into_a_typed_pairing():
    pairing = read_pairing(_module(PROFILE="P", BENCHMARK="B", PIPELINE="PIPE"))
    assert isinstance(pairing, Pairing)
    assert (pairing.profile, pairing.benchmark, pairing.pipeline) == ("P", "B", "PIPE")


def test_read_pairing_names_every_missing_export():
    with pytest.raises(AttributeError) as exc:
        read_pairing(_module(PROFILE="P"))  # missing BENCHMARK and PIPELINE
    message = str(exc.value)
    assert "fake.pairing.module" in message
    assert "missing BENCHMARK, PIPELINE" in message


def test_read_pairing_checks_presence_not_type():
    # The reader only checks the names exist — typing is the gate's job downstream.
    pairing = read_pairing(_module(PROFILE=None, BENCHMARK=0, PIPELINE=[]))
    assert pairing.profile is None
    assert pairing.benchmark == 0
    assert pairing.pipeline == []
