"""The state values a robot reports that no typed field holds: proprioception extras."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from manifold.adapters.packing import PackToNativeLayout, UnpackFromNativeLayout
from manifold.core import (
    Benchmark,
    Camera,
    EEActionSpace,
    EEObservationSpec,
    Embodiment,
    ExtraObservationSpec,
    GripperFormat,
    ObservationAdapter,
    ObservationSpace,
    Pipeline,
    PolicySignature,
    Proprioception,
    RotationFormat,
    check_compatibility,
    verify,
)
from manifold.core.names import EmbodimentName
from manifold.core.native_layout import BatchAxis, LayoutEntry, NativeLayout, Slice, SourceKind
from manifold.core.verify import probe_observation
from manifold.sensors import CameraName

_CAMERA = Camera(name=CameraName.AGENTVIEW, shape=(8, 8, 3))
_ACTION = EEActionSpace(
    rotation=RotationFormat.AXIS_ANGLE, gripper=GripperFormat.SIGNED_OPEN_LOW, delta=True
)
_FINGER_VEL = ExtraObservationSpec(name="gripper_qvel", dim=2, meaning="finger joint velocities")
_ARM_VEL = ExtraObservationSpec(name="joint_vel", dim=7, meaning="arm joint velocities")


def _proprio(*extras: ExtraObservationSpec) -> Proprioception:
    return Proprioception(
        ee_pose=EEObservationSpec(rotation=RotationFormat.QUATERNION), extras=extras
    )


def _benchmark(proprio: Proprioception) -> Benchmark:
    return Benchmark(
        name="suite",
        embodiment=Embodiment(
            name=EmbodimentName.PANDA_OMRON_WHOLE_BODY, action=_ACTION, proprioception=proprio
        ),
        sensors=[_CAMERA],
        instruction=False,
    )


def _policy(proprio: Proprioception) -> PolicySignature:
    return PolicySignature(
        action_space=_ACTION, proprioception=proprio, cameras=[_CAMERA], instruction=False
    )


def _check(report: Any, name: str):  # type: ignore[no-untyped-def]
    matches = [c for c in report.checks if c.name == name]
    assert matches, f"no check named {name!r}; have {[c.name for c in report.checks]}"
    return matches[0]


class _ZeroFingerVelocity(ObservationAdapter):
    """A bug: zeroes the finger velocities."""

    from_spec = ObservationSpace
    to_spec = ObservationSpace
    lossless = True

    def applies(self, source: BaseModel) -> bool:
        return isinstance(source, ObservationSpace)

    def produce(self, source: Any) -> ObservationSpace:
        return source

    def adapt(self, observation: Any, /, *, source: Any) -> Any:
        zeros = np.zeros_like(observation.state["gripper_qvel"])
        return replace(observation, state={**observation.state, "gripper_qvel": zeros})


@pytest.mark.parametrize(
    "fields",
    [
        {"name": "", "dim": 2, "meaning": "finger joint velocities"},
        {"name": "gripper_qvel", "dim": 0, "meaning": "finger joint velocities"},
        {"name": "gripper_qvel", "dim": 2, "meaning": ""},
    ],
)
def test_an_extra_needs_a_name_a_width_and_a_meaning(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ExtraObservationSpec(**fields)


def test_an_extra_named_twice_is_refused() -> None:
    with pytest.raises(ValidationError, match="declared twice"):
        _proprio(_FINGER_VEL, _FINGER_VEL)


def test_an_extra_named_after_a_typed_field_is_refused() -> None:
    with pytest.raises(ValidationError, match="is a typed field"):
        _proprio(ExtraObservationSpec(name="base_pose", dim=7, meaning="the base pose"))


def test_no_extras_are_written_as_an_empty_list() -> None:
    assert Proprioception().model_dump(mode="json")["extras"] == []


def test_a_proprioception_written_without_extras_still_parses() -> None:
    proprio = _proprio()
    written = proprio.model_dump(mode="json")
    del written["extras"]

    assert Proprioception.model_validate(written) == proprio


def test_extras_keep_proprioception_hashable() -> None:
    assert hash(_proprio(_FINGER_VEL)) == hash(_proprio(_FINGER_VEL))


def test_a_policy_that_needs_an_extra_is_refused_without_it() -> None:
    report = check_compatibility(_policy(_proprio(_FINGER_VEL)), _benchmark(_proprio()))

    assert report.status.value == "incompatible"
    assert any("extras.gripper_qvel" in reason for reason in report.reasons), report.reasons


def test_an_extra_of_another_width_is_refused() -> None:
    wider = _FINGER_VEL.model_copy(update={"dim": 3})

    report = check_compatibility(_policy(_proprio(wider)), _benchmark(_proprio(_FINGER_VEL)))

    assert report.status.value == "incompatible"


def test_an_extra_is_met_whatever_its_meaning_says() -> None:
    reworded = _FINGER_VEL.model_copy(update={"meaning": "how fast the fingers move"})

    report = check_compatibility(_policy(_proprio(reworded)), _benchmark(_proprio(_FINGER_VEL)))

    assert report.status.value == "compatible"


def test_a_benchmark_may_offer_extras_a_policy_ignores() -> None:
    report = check_compatibility(_policy(_proprio()), _benchmark(_proprio(_FINGER_VEL, _ARM_VEL)))

    assert report.status.value == "compatible"


def test_the_probe_gives_every_extra_slot_a_value_of_its_own() -> None:
    wide = [ExtraObservationSpec(name=f"wide{i}", dim=12, meaning="wide") for i in range(3)]
    state = probe_observation(_benchmark(_proprio(*wide)).observation_space).state

    values = [v for extra in wide for v in state[extra.name].tolist()]
    assert len(set(values)) == 36
    assert not set(values) & set(state["ee_pose"].tolist())


def test_verify_compares_an_extra_the_policy_consumes() -> None:
    proprio = _proprio(_FINGER_VEL)

    report = verify(_policy(proprio), _benchmark(proprio), Pipeline())

    assert report.ok, report.reasons
    assert _check(report, "observation.extras.gripper_qvel").passed
    assert any("meaning is free text" in reason for reason in report.not_checked)


def test_verify_catches_a_zeroed_extra() -> None:
    proprio = _proprio(_FINGER_VEL)
    pipeline = Pipeline(observation=[_ZeroFingerVelocity()])

    report = verify(_policy(proprio), _benchmark(proprio), pipeline)

    assert not report.ok
    assert not _check(report, "observation.extras.gripper_qvel").passed


def test_an_extra_the_benchmark_does_not_report_is_not_checked() -> None:
    class AddFingerVelocity(_ZeroFingerVelocity):
        def produce(self, source: Any) -> ObservationSpace:
            proprio = source.proprioception
            extras = (*proprio.extras, _FINGER_VEL)
            return source.with_proprioception(proprio.model_copy(update={"extras": extras}))

        def adapt(self, observation: Any, /, *, source: Any) -> Any:
            zeros = np.zeros(2, dtype=np.float32)
            return replace(observation, state={**observation.state, "gripper_qvel": zeros})

    report = verify(
        _policy(_proprio(_FINGER_VEL)),
        _benchmark(_proprio()),
        Pipeline(observation=[AddFingerVelocity()]),
    )

    assert report.ok, report.reasons
    assert not any(c.name == "observation.extras.gripper_qvel" for c in report.checks)
    assert any("extras.gripper_qvel" in reason for reason in report.not_checked)


def test_a_layout_reading_a_declared_extra_is_packed_from_the_spec() -> None:
    layout = NativeLayout(
        entries=(
            LayoutEntry(
                key="state.gripper_qvel",
                source=SourceKind.STATE,
                source_name="gripper_qvel",
                ops=(Slice(start=0, stop=2), BatchAxis(batch=True, time=True)),
            ),
        )
    )
    output = NativeLayout(
        entries=tuple(
            LayoutEntry(
                key=k, source=SourceKind.STATE, source_name=k, ops=(Slice(start=0, stop=1),)
            )
            for k in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
        )
    )
    pipeline = Pipeline(
        pack=PackToNativeLayout(layout), unpack=UnpackFromNativeLayout(output, _ACTION)
    )
    proprio = _proprio(_FINGER_VEL)

    report = verify(_policy(proprio), _benchmark(proprio), pipeline)

    packing = _check(report, "packing.observation")
    assert packing.passed, packing.detail
    assert "1 contract-modeled native key" in packing.detail
    assert not any("synthesized" in reason for reason in report.not_checked), report.not_checked
