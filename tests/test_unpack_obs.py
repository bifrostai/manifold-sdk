"""Env-dict -> SDK `Observation` unpacking (ADR-0001, decision 4).

These exercise `UnpackToObservation` on a SYNTHETIC raw env dict shaped like the
LIBERO runner's (`_observation`/`_ee_pose`): separate `eef_pos`/`eef_quat`/
`gripper_qpos` state keys, renamed image keys, and a language instruction. They
confirm the crux of the design — the multi-source GATHER, where several env keys
concatenate into one `state["ee_pose"]` channel — plus camera renaming and the
instruction, all via the same `_ops` applier and the per-entry `TargetChannel` tag.
"""

from __future__ import annotations

import numpy as np

from manifold.adapters.packing import UnpackToObservation
from manifold.core.embodiment import Proprioception
from manifold.core.native_layout import (
    Assemble,
    ChannelKind,
    Component,
    DtypeCast,
    LayoutEntry,
    NativeLayout,
    Slice,
    SourceKind,
    Split,
    TargetChannel,
)
from manifold.core.observation_space import ObservationSpace
from manifold.core.sensor import Camera
from manifold.core.values import Observation

# --- synthetic fixtures -----------------------------------------------------


def _env_layout() -> NativeLayout:
    """A LIBERO-shaped env layout: gather ee_pose from 3 env keys, rename 2 cameras.

    The three state entries all target the same `ee_pose` role, in gather order
    0/1/2, so the adapter concatenates pos(3) + axis_angle(3) + gripper(2) -> 8-D.
    The two image entries rename `pixels_image`/`pixels_image2` to agentview/wrist
    (one carries a dtype cast to exercise the shared `_ops` applier). The instruction
    entry reads the env's `task_description` key.
    """
    return NativeLayout(
        entries=(
            LayoutEntry(
                key="agentview",
                source=SourceKind.CAMERA,
                source_name="pixels_image",
                ops=(DtypeCast(dtype="uint8", contiguous=True),),
                target=TargetChannel(kind=ChannelKind.CAMERA, name="agentview"),
            ),
            LayoutEntry(
                key="wrist",
                source=SourceKind.CAMERA,
                source_name="pixels_image2",
                target=TargetChannel(kind=ChannelKind.CAMERA, name="wrist"),
            ),
            LayoutEntry(
                key="ee_pos",
                source=SourceKind.STATE,
                source_name="eef_pos",
                target=TargetChannel(kind=ChannelKind.STATE, name="ee_pose", order=0),
            ),
            LayoutEntry(
                key="ee_rot",
                source=SourceKind.STATE,
                source_name="eef_quat",
                target=TargetChannel(kind=ChannelKind.STATE, name="ee_pose", order=1),
            ),
            LayoutEntry(
                key="ee_grip",
                source=SourceKind.STATE,
                source_name="gripper_qpos",
                target=TargetChannel(kind=ChannelKind.STATE, name="ee_pose", order=2),
            ),
            LayoutEntry(
                key="task",
                source=SourceKind.INSTRUCTION,
                source_name="task_description",
                target=TargetChannel(kind=ChannelKind.INSTRUCTION),
            ),
        )
    )


def _observation_space() -> ObservationSpace:
    return ObservationSpace(
        proprioception=Proprioception(),
        cameras=(
            Camera(name="agentview", shape=(4, 4, 3)),
            Camera(name="wrist", shape=(4, 4, 3)),
        ),
        instruction=True,
    )


def _env_dict() -> dict[str, np.ndarray | str]:
    return {
        "pixels_image": np.full((4, 4, 3), 1, dtype=np.uint8),
        "pixels_image2": np.full((4, 4, 3), 2, dtype=np.uint8),
        "eef_pos": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        "eef_quat": np.array([3.0, 4.0, 5.0], dtype=np.float32),
        "gripper_qpos": np.array([6.0, 7.0], dtype=np.float32),
        "task_description": "pick up the bowl",
    }


# --- the adapter contract ---------------------------------------------------


def test_applies_and_produces() -> None:
    layout = _env_layout()
    space = _observation_space()
    adapter = UnpackToObservation(layout, space)
    assert adapter.applies(layout)
    assert not adapter.applies(NativeLayout())  # a different layout
    assert adapter.produce(layout) is space


def test_unpack_gathers_state_renames_cameras_sets_instruction() -> None:
    adapter = UnpackToObservation(_env_layout(), _observation_space())
    obs = adapter.adapt(_env_dict(), source=_env_layout())
    assert isinstance(obs, Observation)

    # THE CRUX: ee_pose is the concatenation of three SEPARATE env keys, in gather
    # order — pos(0,1,2) + axis_angle(3,4,5) + gripper(6,7) = the 8-D ee_pose.
    assert list(obs.state["ee_pose"]) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert obs.state["ee_pose"].shape == (8,)
    assert set(obs.state) == {"ee_pose"}

    # Cameras renamed/placed under their sensor names, the cast honoured.
    assert set(obs.sensors) == {"agentview", "wrist"}
    assert obs.sensors["agentview"].shape == (4, 4, 3)
    assert obs.sensors["agentview"].dtype == np.uint8
    assert obs.sensors["agentview"].flags["C_CONTIGUOUS"]
    assert int(obs.sensors["agentview"].reshape(-1)[0]) == 1
    assert int(obs.sensors["wrist"].reshape(-1)[0]) == 2

    # Instruction set from the env's task_description key.
    assert obs.instruction == "pick up the bowl"


def test_gather_respects_declared_order_not_dict_order() -> None:
    # Reverse the gather order on the layout entries: the same env dict must now
    # concatenate gripper + axis_angle + pos, proving order comes from target.order,
    # not declaration or env-dict order.
    layout = NativeLayout(
        entries=(
            LayoutEntry(
                key="g",
                source=SourceKind.STATE,
                source_name="gripper_qpos",
                target=TargetChannel(kind=ChannelKind.STATE, name="ee_pose", order=0),
            ),
            LayoutEntry(
                key="p",
                source=SourceKind.STATE,
                source_name="eef_pos",
                target=TargetChannel(kind=ChannelKind.STATE, name="ee_pose", order=1),
            ),
        )
    )
    adapter = UnpackToObservation(layout, _observation_space())
    obs = adapter.adapt(_env_dict(), source=layout)
    assert list(obs.state["ee_pose"]) == [6.0, 7.0, 0.0, 1.0, 2.0]


def test_missing_env_key_raises() -> None:
    adapter = UnpackToObservation(_env_layout(), _observation_space())
    env = _env_dict()
    del env["eef_quat"]
    try:
        adapter.adapt(env, source=_env_layout())
    except KeyError:
        return
    raise AssertionError("a missing env-dict key should raise")


def test_split_fanout_entry_rejected_at_construction() -> None:
    # A `Split` fans one source into MANY keys; an env->obs entry (target set) routes to
    # exactly one channel slot, so a Split has no destination. The LayoutEntry validator
    # rejects this at construction — a clear error where the layout is written, rather
    # than an opaque single-value-unpack failure mid-rollout.
    try:
        LayoutEntry(
            key="grp",
            source=SourceKind.STATE,
            source_name="eef_pos",
            ops=(
                Split(
                    components=(
                        Component(key="a", start=0, stop=1),
                        Component(key="b", start=1, stop=3),
                    )
                ),
            ),
            target=TargetChannel(kind=ChannelKind.STATE, name="ee_pose"),
        )
    except ValueError:
        return
    raise AssertionError("a Split in a target (env->obs) entry should be rejected at construction")


def test_entry_without_target_raises() -> None:
    # A pack/unpack-style entry (target=None) is meaningless on the env->obs side.
    layout = NativeLayout(
        entries=(LayoutEntry(key="x", source=SourceKind.STATE, source_name="eef_pos"),)
    )
    adapter = UnpackToObservation(layout, _observation_space())
    try:
        adapter.adapt(_env_dict(), source=layout)
    except ValueError:
        return
    raise AssertionError("an entry without a target channel should raise")


# --- the env->obs builders + validators -------------------------------------


def test_builders_are_byte_equivalent_to_the_long_hand_form() -> None:
    # The from_camera/from_state/from_instruction sugar must produce the same
    # Observation as the long-hand entries: the builders only author the entries more
    # legibly (their `key` differs, but `key` never reaches the env->obs Observation).
    sugared = NativeLayout(
        entries=(
            LayoutEntry.from_camera("pixels_image", channel="agentview"),
            LayoutEntry.from_state("ee_pose", "eef_pos", dim=3, order=0),
            LayoutEntry.from_state("ee_pose", "eef_quat", dim=3, order=1),
            LayoutEntry.from_state("ee_pose", "gripper_qpos", dim=2, order=2),
            LayoutEntry.from_instruction("task_description"),
        )
    )
    explicit = NativeLayout(
        entries=(
            LayoutEntry(
                key="x",
                source=SourceKind.CAMERA,
                source_name="pixels_image",
                ops=(DtypeCast(dtype="uint8", contiguous=True),),
                target=TargetChannel(kind=ChannelKind.CAMERA, name="agentview"),
            ),
            LayoutEntry(
                key="p",
                source=SourceKind.STATE,
                source_name="eef_pos",
                ops=(DtypeCast(dtype="float32"), Slice(start=0, stop=3)),
                target=TargetChannel(kind=ChannelKind.STATE, name="ee_pose", order=0),
            ),
            LayoutEntry(
                key="q",
                source=SourceKind.STATE,
                source_name="eef_quat",
                ops=(DtypeCast(dtype="float32"), Slice(start=0, stop=3)),
                target=TargetChannel(kind=ChannelKind.STATE, name="ee_pose", order=1),
            ),
            LayoutEntry(
                key="g",
                source=SourceKind.STATE,
                source_name="gripper_qpos",
                ops=(DtypeCast(dtype="float32"), Slice(start=0, stop=2)),
                target=TargetChannel(kind=ChannelKind.STATE, name="ee_pose", order=2),
            ),
            LayoutEntry(
                key="i",
                source=SourceKind.INSTRUCTION,
                source_name="task_description",
                target=TargetChannel(kind=ChannelKind.INSTRUCTION),
            ),
        )
    )
    space = _observation_space()
    a = UnpackToObservation(sugared, space).adapt(_env_dict(), source=sugared)
    b = UnpackToObservation(explicit, space).adapt(_env_dict(), source=explicit)
    assert list(a.state["ee_pose"]) == list(b.state["ee_pose"])
    assert a.instruction == b.instruction
    assert set(a.sensors) == set(b.sensors)
    assert a.sensors["agentview"].dtype == b.sensors["agentview"].dtype
    np.testing.assert_array_equal(a.sensors["agentview"], b.sensors["agentview"])


def test_from_state_builds_cast_then_slice_targeting_the_role() -> None:
    entry = LayoutEntry.from_state("ee_pose", "eef_pos", dim=3, order=2)
    assert entry.source is SourceKind.STATE
    assert entry.source_name == "eef_pos"
    assert entry.target == TargetChannel(kind=ChannelKind.STATE, name="ee_pose", order=2)
    assert [type(op).__name__ for op in entry.ops] == ["DtypeCast", "Slice"]
    # dim=None omits the Slice (the whole vector gathers, as SIMPLER's 8-D ee_pose does).
    assert [type(op).__name__ for op in LayoutEntry.from_state("ee_pose", "ee_pose").ops] == [
        "DtypeCast"
    ]


def test_instruction_entry_with_ops_rejected_at_construction() -> None:
    # Both directions: a pack entry whose SOURCE is the instruction...
    try:
        LayoutEntry(key="i", source=SourceKind.INSTRUCTION, ops=(DtypeCast(dtype="float32"),))
    except ValueError:
        pass
    else:
        raise AssertionError("an INSTRUCTION-source entry carrying ops should be rejected")
    # ...and an env->obs entry whose TARGET is the instruction (matches unpack_obs's guard).
    try:
        LayoutEntry(
            key="i",
            source=SourceKind.STATE,
            source_name="x",
            ops=(DtypeCast(dtype="float32"),),
            target=TargetChannel(kind=ChannelKind.INSTRUCTION),
        )
    except ValueError:
        return
    raise AssertionError("an INSTRUCTION-target entry carrying ops should be rejected")


def test_split_followed_by_fanout_rejected_at_construction() -> None:
    # The applier windows a single working array after a fan-out, so a Split cannot be
    # followed by another Split/Assemble. The validator turns that runtime ValueError
    # into a construction-time one.
    try:
        LayoutEntry(
            key="grp",
            source=SourceKind.STATE,
            source_name="ee_pose",
            ops=(
                Split(components=(Component(key="a", start=0, stop=4),)),
                Assemble(components=(Component(key="x", start=0, stop=2),)),
            ),
        )
    except ValueError:
        return
    raise AssertionError("a Split followed by an Assemble should be rejected at construction")
