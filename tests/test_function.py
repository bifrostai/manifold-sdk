import warnings

import numpy as np
import pytest

import manifold.recipes.function as function
from manifold.core.policy import PolicySignature
from manifold.core.values import Action, Observation
from manifold.embodiments.franka_ee_delta import FRANKA_EE_DELTA
from manifold.recipes.function import FunctionEndpoint, check_chunk

# A FRANKA_EE_DELTA action holds 7 values.
_WIDTH = 7


def _signature(chunk_size: int = 1, execution_steps: int = 1) -> PolicySignature:
    return PolicySignature(
        action_space=FRANKA_EE_DELTA.action,
        proprioception=FRANKA_EE_DELTA.proprioception,
        chunk_size=chunk_size,
        execution_steps=execution_steps,
    )


class _CountingPredict:
    """A predict that returns `rows` distinct actions per call and counts its calls.

    Row `i` of call `n` is filled with `100 * n + i`, so a served action shows the
    call and the row that it came from.
    """

    def __init__(self, rows: int) -> None:
        self.rows = rows
        self.calls = 0

    def __call__(self, _observation: Observation) -> Action:
        self.calls += 1
        actions = np.repeat(100.0 * self.calls + np.arange(self.rows), _WIDTH)
        return Action.from_array(actions.reshape(self.rows, _WIDTH))


def test_function_endpoint_returns_the_prediction() -> None:
    action = Action.from_array(np.zeros(_WIDTH, dtype=np.float32))

    def predict(_observation: Observation) -> Action:
        return action

    endpoint = FunctionEndpoint(predict, _signature())
    assert endpoint.session().infer(Observation()) is action


@pytest.mark.parametrize(
    ("chunk_size", "execution_steps", "expected_calls"),
    [(10, 5, 4), (10, 10, 2), (1, 1, 20)],
)
def test_a_session_calls_predict_once_per_execution_steps(
    chunk_size: int, execution_steps: int, expected_calls: int
) -> None:
    predict = _CountingPredict(rows=chunk_size)
    session = FunctionEndpoint(predict, _signature(chunk_size, execution_steps)).session()

    served = [float(session.infer(Observation()).values[0]) for _ in range(20)]

    assert predict.calls == expected_calls
    # Each call serves its first `execution_steps` rows, in order.
    expected = [
        100.0 * (step // execution_steps + 1) + step % execution_steps for step in range(20)
    ]
    assert served == expected


def test_reset_drops_the_buffered_moves() -> None:
    predict = _CountingPredict(rows=10)
    session = FunctionEndpoint(predict, _signature(10, 5)).session()
    session.infer(Observation())
    session.infer(Observation())
    assert predict.calls == 1

    session.reset()

    assert float(session.infer(Observation()).values[0]) == 200.0
    assert predict.calls == 2


def test_each_session_keeps_its_own_buffer() -> None:
    predict = _CountingPredict(rows=10)
    endpoint = FunctionEndpoint(predict, _signature(10, 5))
    first, second = endpoint.session(), endpoint.session()

    first.infer(Observation())
    second.infer(Observation())

    assert predict.calls == 2


def test_a_session_rejects_a_chunk_of_the_wrong_length() -> None:
    session = FunctionEndpoint(_CountingPredict(rows=8), _signature(10, 5)).session()
    with pytest.raises(ValueError, match=r"returned 8 actions of 7 values.*`chunk_size=10`"):
        session.infer(Observation())


def test_multiple_rows_under_the_default_fields_run_the_first_and_warn_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(function, "_legacy_rows_warned", False)
    predict = _CountingPredict(rows=5)
    session = FunctionEndpoint(predict, _signature()).session()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        served = [float(session.infer(Observation()).values[0]) for _ in range(3)]
        FunctionEndpoint(predict, _signature()).session().infer(Observation())

    # Row 0 of each call runs, and the model is called every step.
    assert served == [100.0, 200.0, 300.0]
    assert predict.calls == 4
    assert len(caught) == 1
    assert "chunk_size" in str(caught[0].message)
    assert "execution_steps" in str(caught[0].message)


def test_check_chunk_passes_a_flat_move_under_the_default_fields() -> None:
    action = Action.from_array(np.zeros(_WIDTH))
    assert check_chunk(action, _signature()) == [action]


def test_check_chunk_rejects_a_flat_move_of_the_wrong_width() -> None:
    with pytest.raises(ValueError, match=r"1 action of 6 values.*declares 7 values"):
        check_chunk(Action.from_array(np.zeros(6)), _signature())


def test_check_chunk_rejects_a_three_dimensional_result() -> None:
    with pytest.raises(ValueError, match=r"shape \(1, 10, 7\)"):
        check_chunk(Action.from_array(np.zeros((1, 10, _WIDTH))), _signature(10, 5))
