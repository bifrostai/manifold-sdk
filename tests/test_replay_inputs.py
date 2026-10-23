"""Tests that the replay never fails an episode on a value that it cannot read.

A benchmark's hooks and a policy can send values that the replay cannot chart: a
rotation of zeros, NaN, infinity, an array of the wrong length, or a state that is
missing on some frames. Each case here records an episode with such a value, for each
embodiment that the parity benchmarks use, and checks that the log is written and
read back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from manifold.benchmark import BenchmarkConfig
from manifold.benchmark._replay import _ReplayRecorder
from manifold.core.values import Action, Observation
from manifold.embodiments import DROID_JOINT_ABSOLUTE, FRANKA_EE_DELTA, WIDOWX_EE_DELTA
from manifold.replay import read_replay_log, replay_log_path
from tests.test_benchmark_runtime import _raw_config

if TYPE_CHECKING:
    from pathlib import Path

    from manifold.core.embodiment import Embodiment

_EMBODIMENTS = {
    "droid_joint": DROID_JOINT_ABSOLUTE,
    "franka_ee": FRANKA_EE_DELTA,
    "widowx_ee": WIDOWX_EE_DELTA,
}

# Each kind of value that the replay cannot chart, as a function of a valid array.
_BAD_VALUES = {
    "zeros": np.zeros_like,
    "nan": lambda values: np.full_like(values, np.nan),
    "inf": lambda values: np.full_like(values, np.inf),
    "tiny": lambda values: np.full_like(values, 1e-300),
    "short": lambda values: values[: len(values) // 2],
    "long": lambda values: np.concatenate([values, values]),
    "empty": lambda values: values[:0],
}


def _recorder(embodiment: Embodiment) -> _ReplayRecorder:
    raw = _raw_config({"type": "single"})
    raw["spec"] = {**raw["spec"], "embodiment": embodiment.model_dump(mode="json")}
    return _ReplayRecorder(BenchmarkConfig.model_validate(raw), None, 0)


def _state(embodiment: Embodiment) -> dict[str, np.ndarray]:
    """A valid state for each proprioception field that `embodiment` observes."""
    proprioception = embodiment.proprioception
    state: dict[str, np.ndarray] = {}
    if proprioception.ee_pose is not None:
        # A position, the identity quaternion in `xyzw`, and any gripper values.
        gripper = proprioception.ee_pose.gripper
        state["ee_pose"] = np.array(
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, *([0.04] * (gripper.dim if gripper else 0))]
        )
    if proprioception.joint_pos is not None:
        state["joint_pos"] = np.full(proprioception.joint_pos.dof + 1, 0.5)
    return state


def _observation(state: dict[str, Any]) -> Observation:
    return Observation(
        sensors={"agentview": np.zeros((4, 4, 3), dtype=np.uint8)},
        state=state,
        instruction="lift the toy",
    )


def _action(embodiment: Embodiment) -> np.ndarray:
    return np.full(embodiment.action.expected_length(), 0.5)


def _record(
    tmp_path: Path,
    embodiment: Embodiment,
    states: list[dict[str, Any]],
    actions: list[np.ndarray | None],
) -> None:
    """Record an episode with a first frame and a frame for each action, and read it back."""
    recorder = _recorder(embodiment)
    recorder.begin(tmp_path, 0, _observation(states[0]), {}, ())
    for step, (state, action) in enumerate(zip(states[1:], actions, strict=True)):
        recorder.record(
            None if action is None else Action(values=action),
            _observation(state),
            (),
            success=False,
            done=step == len(actions) - 1,
        )
    recorder.end()
    assert len(read_replay_log(replay_log_path(tmp_path, 0)).steps) == len(states)


@pytest.mark.parametrize("bad", sorted(_BAD_VALUES))
@pytest.mark.parametrize("field", ["ee_pose", "joint_pos"])
@pytest.mark.parametrize("name", sorted(_EMBODIMENTS))
def test_a_bad_state_value_on_every_frame_is_charted_without_failing(
    tmp_path: Path, name: str, field: str, bad: str
) -> None:
    embodiment = _EMBODIMENTS[name]
    state = _state(embodiment)
    if field not in state:
        pytest.skip(f"{name} observes no {field}")
    state[field] = _BAD_VALUES[bad](state[field])

    _record(tmp_path, embodiment, [state, state, state], [_action(embodiment)] * 2)


@pytest.mark.parametrize("bad", sorted(_BAD_VALUES))
@pytest.mark.parametrize("field", ["ee_pose", "joint_pos"])
@pytest.mark.parametrize("name", sorted(_EMBODIMENTS))
def test_a_bad_state_value_after_the_first_frame_is_charted_without_failing(
    tmp_path: Path, name: str, field: str, bad: str
) -> None:
    embodiment = _EMBODIMENTS[name]
    good = _state(embodiment)
    if field not in good:
        pytest.skip(f"{name} observes no {field}")
    bad_state = {**good, field: _BAD_VALUES[bad](good[field])}

    _record(tmp_path, embodiment, [good, bad_state, good], [_action(embodiment)] * 2)


@pytest.mark.parametrize("field", ["ee_pose", "joint_pos"])
@pytest.mark.parametrize("name", sorted(_EMBODIMENTS))
def test_a_state_missing_on_some_frames_is_charted_without_failing(
    tmp_path: Path, name: str, field: str
) -> None:
    embodiment = _EMBODIMENTS[name]
    good = _state(embodiment)
    if field not in good:
        pytest.skip(f"{name} observes no {field}")
    missing = {key: value for key, value in good.items() if key != field}

    _record(tmp_path, embodiment, [good, missing, good], [_action(embodiment)] * 2)
    _record(tmp_path, embodiment, [missing, good, missing], [_action(embodiment)] * 2)


@pytest.mark.parametrize("bad", sorted(_BAD_VALUES))
@pytest.mark.parametrize("name", sorted(_EMBODIMENTS))
def test_a_bad_action_is_charted_without_failing(tmp_path: Path, name: str, bad: str) -> None:
    embodiment = _EMBODIMENTS[name]
    state = _state(embodiment)
    good = _action(embodiment)

    _record(tmp_path, embodiment, [state, state, state], [_BAD_VALUES[bad](good), good])
