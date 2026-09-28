"""Serve a Python prediction function.

A call to `predict` returns the actions that the model predicts. When the
signature declares `chunk_size > 1`, `predict` returns a 2-D array with
`chunk_size` rows. Each row is an action laid out per `action_space`. The
session stores the rows and calls `predict` again after `execution_steps` of
them have run. With the default `chunk_size=1`, `predict` returns a single
action as a flat vector or a single row.
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Callable

import numpy as np

from manifold.core.policy import PolicySignature
from manifold.core.values import Action, Observation
from manifold.recipes.serving import ActionQueue

# Guards the one-per-process warning for a multi-row result under the default
# chunk fields; sessions run on separate worker threads.
_legacy_rows_lock = threading.Lock()
_legacy_rows_warned = False


def _warn_legacy_rows(rows: int) -> None:
    """Warn once per process that only the first of `rows` actions runs."""
    global _legacy_rows_warned
    with _legacy_rows_lock:
        if _legacy_rows_warned:
            return
        _legacy_rows_warned = True
    warnings.warn(
        f"`predict` returned {rows} actions but the signature declares the default "
        "`chunk_size=1`, so only the first action runs and the model is called every "
        "step. Declare `PolicySignature(chunk_size=..., execution_steps=...)` to run "
        "the rest of the chunk.",
        UserWarning,
        stacklevel=3,
    )


def check_chunk(action: Action, signature: PolicySignature) -> list[Action]:
    """Check a `predict` result against `signature` and return its actions.

    The result must be a 2-D array of `chunk_size` rows by
    `action_space.expected_length()` columns. When `chunk_size` is 1, a flat
    vector of a single action also passes and comes back unchanged. A multi-row result
    under the default `chunk_size=1` keeps the behaviour from before the chunk
    fields existed: the first row runs, with a warning once per process.
    Raises `ValueError` naming both the returned and the declared shape.
    """
    values = np.asarray(action.values)
    width = signature.action_space.expected_length()
    chunk_size = signature.chunk_size
    if values.ndim == 1:
        if chunk_size != 1:
            raise ValueError(
                f"`predict` returned a flat array of {values.shape[0]} values; the "
                f"signature declares `chunk_size={chunk_size}`, so `predict` must return "
                f"{chunk_size} rows of {width} values"
            )
        if values.shape[0] != width:
            raise ValueError(
                f"`predict` returned 1 action of {values.shape[0]} values; the signature's "
                f"action space declares {width} values for each action"
            )
        return [action]
    if values.ndim != 2:
        raise ValueError(
            f"`predict` returned an array of shape {values.shape}; the signature declares "
            f"`chunk_size={chunk_size}` actions of {width} values, shape ({chunk_size}, {width})"
        )
    rows, columns = values.shape
    if columns != width or (rows != chunk_size and chunk_size != 1):
        raise ValueError(
            f"`predict` returned {rows} actions of {columns} values; the signature declares "
            f"`chunk_size={chunk_size}` actions of {width} values"
        )
    if rows != chunk_size:
        # Only reachable under the default chunk fields (a `chunk_size` of 1 forces
        # `execution_steps` to 1): an older script that slices its chunk itself.
        _warn_legacy_rows(rows)
        return [Action(values=values[0])]
    return [Action(values=row) for row in values]


class FunctionSession(ActionQueue):
    """A shard's session over a prediction function.

    `ActionQueue` stores the actions from a `predict` call. The session calls
    `predict` again after `signature.execution_steps` of them have run. Each
    shard gets its own session, so shards keep separate actions.
    """

    _endpoint: FunctionEndpoint

    def _forward(self, observation: Observation, /) -> list[Action]:
        """Call `predict` and return its actions, checked against the signature."""
        return check_chunk(self._endpoint.predict(observation), self._endpoint.signature)

    def infer(self, observation: Observation, /) -> Action:
        """Return the next action, and call `predict` after the stored actions have run."""
        actions, step = self.advance(observation)
        return actions[step]


class FunctionEndpoint:
    """Create sessions for one prediction function."""

    def __init__(
        self,
        predict: Callable[[Observation], Action],
        signature: PolicySignature,
    ) -> None:
        self.predict = predict
        self.signature = signature

    def session(self) -> FunctionSession:
        """Create one policy session."""
        return FunctionSession(self)

    def forward(self, _native: dict[str, object], /) -> object:
        """Reject native model requests."""
        raise NotImplementedError("FunctionEndpoint does not use native model inputs")


__all__ = [
    "FunctionEndpoint",
    "FunctionSession",
    "check_chunk",
]
