"""Serve a Python prediction function."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from manifold.core.policy import PolicySignature
from manifold.core.values import Action, Observation


class FunctionSession:
    """One session that calls a prediction function."""

    def __init__(self, predict: Callable[[Observation], Action]) -> None:
        self._predict = predict

    def advance(self, native: dict[str, Any], /) -> tuple[Any, int]:
        """Reject a packed observation.

        A prediction function takes an `Observation`, so no pipeline over this
        endpoint packs. `serve` calls `infer` instead, and reaches this only
        where somebody paired the endpoint with a packing pipeline.
        """
        raise NotImplementedError("a prediction function takes no packed observation")

    def infer(self, observation: Observation, /) -> Action:
        """Return an action for one observation."""
        return self._predict(observation)

    def reset(self) -> None:
        """Reset the session."""

    def close(self) -> None:
        """Close the session."""


class FunctionEndpoint:
    """Create sessions for one prediction function."""

    def __init__(
        self,
        predict: Callable[[Observation], Action],
        signature: PolicySignature,
    ) -> None:
        self._predict = predict
        self.signature = signature

    def session(self) -> FunctionSession:
        """Create one policy session."""
        return FunctionSession(self._predict)

    def forward(self, _native: dict[str, object], /) -> object:
        """Reject native model requests."""
        raise NotImplementedError("FunctionEndpoint does not use native model inputs")


__all__ = [
    "FunctionEndpoint",
    "FunctionSession",
]
