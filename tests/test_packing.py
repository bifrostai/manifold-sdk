"""Native-packing phase: NativeLayout + Pack/Unpack adapters + verify coverage.

These exercise the packing machinery on SYNTHETIC specs and layouts, confirming:

- `NativeLayout` expresses the structural algebra a checkpoint's native layout
  needs — rename, dtype cast, batch/time axis, slice, split, assemble (gripper-first).
- `PackToNativeLayout.adapt` turns an `Observation` into the declared tensor dict.
- `UnpackFromNativeLayout.adapt` reads a raw model-output chunk into an `Action`.
- `verify` exercises the packing phase when present and is unchanged when absent.
- `check_compatibility` stays at the channel seam (packing never folds into it).
"""

from __future__ import annotations

import numpy as np

from manifold.adapters.packing import PackToNativeLayout, UnpackFromNativeLayout
from manifold.core.benchmark import Benchmark
from manifold.core.check import check_compatibility
from manifold.core.conventions import GripperFormat, RotationFormat
from manifold.core.embodiment import (
    EEActionSpace,
    EEObservationSpec,
    Embodiment,
    GripperObservationSpec,
    Proprioception,
)
from manifold.core.native_layout import (
    Assemble,
    BatchAxis,
    Component,
    DtypeCast,
    LayoutEntry,
    NativeLayout,
    Slice,
    SourceKind,
    Split,
)
from manifold.core.observation_space import ObservationSpace
from manifold.core.pipeline import NoPackingPhase, Pipeline
from manifold.core.policy import PolicySignature
from manifold.core.sensor import Camera
from manifold.core.values import Action, Observation
from manifold.core.verify import verify

# --- synthetic fixtures -----------------------------------------------------


def _ee(rotation: RotationFormat = RotationFormat.AXIS_ANGLE) -> EEActionSpace:
    # 3 pos + 3 axis-angle + 1 gripper = 7 per step, chunk 1.
    return EEActionSpace(rotation=rotation, gripper=GripperFormat.SIGNED, delta=True)


def _proprio() -> Proprioception:
    # ee_pose = 3 pos + 3 axis-angle + 2 gripper qpos = 8.
    return Proprioception(
        ee_pose=EEObservationSpec(
            rotation=RotationFormat.AXIS_ANGLE, gripper=GripperObservationSpec(dim=2)
        )
    )


def _observation_space() -> ObservationSpace:
    return ObservationSpace(
        proprioception=_proprio(),
        cameras=(Camera(name="agentview", shape=(8, 8, 3)),),
        instruction=True,
    )


def _benchmark() -> Benchmark:
    return Benchmark(
        name="suite",
        embodiment=Embodiment(name="arm", action=_ee(), proprioception=_proprio()),
        sensors=[Camera(name="agentview", shape=(8, 8, 3))],
        instruction=True,
    )


def _policy() -> PolicySignature:
    obs = _observation_space()
    return PolicySignature(
        action_space=_ee(),
        proprioception=obs.proprioception,
        cameras=list(obs.cameras),
        instruction=obs.instruction,
    )


def _observation() -> Observation:
    # ee_pose laid out [pos3, axisangle3, gripper_qpos2]; a distinct value per slot.
    ee = np.arange(8, dtype=np.float32)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    return Observation(state={"ee_pose": ee}, sensors={"agentview": frame}, instruction="pick")


# An input layout exercising every structural op:
#  - rename + cast + batch axis on the camera (video.image);
#  - SPLIT the 8-D ee_pose into per-axis scalar keys + a 2-D gripper slice, each
#    cast and given a (batch, time) prefix (the RLDX flat-key shape);
#  - ASSEMBLE a gripper-FIRST proprio vector concat(gripper_qpos(2), pos(3)) on a
#    second copy of the same source (the Cosmos gripper-first shape).
def _input_layout() -> NativeLayout:
    return NativeLayout(
        entries=(
            LayoutEntry(
                key="video.image",
                source=SourceKind.CAMERA,
                source_name="agentview",
                ops=(DtypeCast(dtype="uint8", contiguous=True), BatchAxis(batch=True)),
            ),
            LayoutEntry(
                key="state",  # group key, dropped; the split components are emitted.
                source=SourceKind.STATE,
                source_name="ee_pose",
                ops=(
                    Split(
                        components=(
                            Component(key="state.x", start=0, stop=1),
                            Component(key="state.y", start=1, stop=2),
                            Component(key="state.z", start=2, stop=3),
                            Component(key="state.gripper", start=6, stop=8),
                        )
                    ),
                    DtypeCast(dtype="float32"),
                    BatchAxis(batch=True, time=True),
                ),
            ),
            LayoutEntry(
                key="proprio",
                source=SourceKind.STATE,
                source_name="ee_pose",
                ops=(
                    Assemble(
                        components=(
                            Component(key="ee_pose", start=6, stop=8),  # gripper FIRST
                            Component(key="ee_pose", start=0, stop=3),  # then position
                        )
                    ),
                ),
            ),
            LayoutEntry(key="task", source=SourceKind.INSTRUCTION),
        )
    )


# An OUTPUT layout: a per-axis action dict (batch, time, 1) the model emits, read
# into the 7-D EE action [x, y, z, roll, pitch, yaw, gripper] via slices.
def _output_layout() -> NativeLayout:
    keys = (
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    )
    return NativeLayout(
        entries=tuple(
            LayoutEntry(
                key=k, source=SourceKind.STATE, source_name=k, ops=(Slice(start=0, stop=1),)
            )
            for k in keys
        )
    )


# --- NativeLayout schema ----------------------------------------------------


def test_layout_keys_expands_splits() -> None:
    layout = _input_layout()
    # The split's group key "state" is not emitted; its components are. Camera,
    # proprio, and task keys pass through.
    assert layout.keys() == (
        "video.image",
        "state.x",
        "state.y",
        "state.z",
        "state.gripper",
        "proprio",
        "task",
    )


def test_layout_is_frozen() -> None:
    layout = _input_layout()
    try:
        layout.entries = ()  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("NativeLayout should be frozen")


# --- PackToNativeLayout ------------------------------------------------------


def test_pack_applies_all_ops() -> None:
    pack = PackToNativeLayout(_input_layout())
    space = _observation_space()
    assert pack.applies(space)
    assert pack.produce(space) is _input_layout() or pack.produce(space) == _input_layout()

    packed = pack.adapt(_observation(), source=space)

    # Keys match the layout exactly (split expanded, group key gone).
    assert set(packed) == set(_input_layout().keys())

    # Camera: cast to uint8, C-contiguous, batch axis prepended -> (1, 8, 8, 3).
    cam = packed["video.image"]
    assert cam.shape == (1, 8, 8, 3)
    assert cam.dtype == np.uint8
    assert cam.flags["C_CONTIGUOUS"]

    # Split per-axis scalars: (1, 1, 1) each, value = the source slot.
    assert packed["state.x"].shape == (1, 1, 1)
    assert packed["state.x"].reshape(-1)[0] == 0.0
    assert packed["state.z"].reshape(-1)[0] == 2.0
    # The gripper split keeps a 2-vector window -> (1, 1, 2) = [6, 7].
    assert packed["state.gripper"].shape == (1, 1, 2)
    assert list(packed["state.gripper"].reshape(-1)) == [6.0, 7.0]

    # Assemble: gripper-FIRST concat(gripper_qpos(6:8), pos(0:3)) = [6,7,0,1,2].
    assert list(packed["proprio"].reshape(-1)) == [6.0, 7.0, 0.0, 1.0, 2.0]

    # Instruction packed as a length-1 tuple of the string (text, not a tensor) —
    # byte-identical to the old to_obs, the form GR00T's annotation key expects.
    assert packed["task"] == ("pick",)


def test_pack_missing_channel_raises() -> None:
    pack = PackToNativeLayout(_input_layout())
    obs = Observation(
        state={}, sensors={"agentview": np.zeros((8, 8, 3), np.uint8)}, instruction="x"
    )
    try:
        pack.adapt(obs, source=_observation_space())
    except KeyError:
        return
    raise AssertionError("packing a missing state channel should raise")


# --- UnpackFromNativeLayout --------------------------------------------------


def test_unpack_reads_chunk_into_action() -> None:
    layout = _output_layout()
    unpack = UnpackFromNativeLayout(layout, _ee())
    assert unpack.applies(layout)
    assert unpack.produce(layout) == _ee()

    # A per-axis raw output dict, each (batch=1, time=2, dim=1); step 1 distinct.
    raw = {
        "action.x": np.array([[[10.0], [11.0]]], dtype=np.float32),
        "action.y": np.array([[[20.0], [21.0]]], dtype=np.float32),
        "action.z": np.array([[[30.0], [31.0]]], dtype=np.float32),
        "action.roll": np.array([[[40.0], [41.0]]], dtype=np.float32),
        "action.pitch": np.array([[[50.0], [51.0]]], dtype=np.float32),
        "action.yaw": np.array([[[60.0], [61.0]]], dtype=np.float32),
        "action.gripper": np.array([[[1.0], [0.0]]], dtype=np.float32),
    }
    action = unpack.adapt(raw, source=layout, step=0)
    assert isinstance(action, Action)
    assert list(action.values) == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 1.0]

    action1 = unpack.adapt(raw, source=layout, step=1)
    assert list(action1.values) == [11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 0.0]
    # Length matches the policy's 7-D EE action space.
    _ee().validate_value(action1.values)


def test_unpack_wrong_width_raises() -> None:
    # The layout reads only 3 dims, but the policy's EE action space is 7-D. An empty
    # convention chain would pass a malformed 3-D Action straight through, so the
    # unpack must validate the width against its action space and fail loudly.
    layout = NativeLayout(
        entries=(
            LayoutEntry(
                key="action", source=SourceKind.STATE, source_name=None, ops=(Slice(stop=3),)
            ),
        )
    )
    unpack = UnpackFromNativeLayout(layout, _ee())  # _ee() is 7-D
    chunk = np.arange(2 * 9, dtype=np.float32).reshape(2, 9)
    try:
        unpack.adapt(chunk, source=layout, step=0)
    except Exception:
        return
    raise AssertionError("a wrong-width unpack should raise against the action space")


def test_unpack_single_array_chunk() -> None:
    # An output layout reading a single (chunk, dim) array (Cosmos's chunk[step][:7]).
    # source_name None -> the entry indexes the chunk array itself.
    layout = NativeLayout(
        entries=(
            LayoutEntry(
                key="action", source=SourceKind.STATE, source_name=None, ops=(Slice(stop=7),)
            ),
        )
    )
    unpack = UnpackFromNativeLayout(layout, _ee())
    chunk = np.arange(2 * 9, dtype=np.float32).reshape(2, 9)  # 2 steps, 9 wide
    action = unpack.adapt(chunk, source=layout, step=1)
    # Step 1 row is [9..17]; first 7 sliced.
    assert list(action.values) == [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]


# --- Pipeline carries the optional phase -------------------------------------


def test_pipeline_packing_fields_default_none() -> None:
    p = Pipeline()
    assert p.pack is None and p.unpack is None


def test_apply_pack_unpack_round_through_pipeline() -> None:
    pack = PackToNativeLayout(_input_layout())
    unpack = UnpackFromNativeLayout(_output_layout(), _ee())
    p = Pipeline(pack=pack, unpack=unpack)
    packed = p.apply_pack(_observation(), source=_observation_space())
    assert set(packed) == set(_input_layout().keys())
    raw = {
        k: np.array([[[float(i)]]], dtype=np.float32) for i, k in enumerate(_output_layout().keys())
    }
    action = p.apply_unpack(raw, source=_output_layout(), step=0)
    _ee().validate_value(action.values)


def test_apply_pack_without_phase_raises() -> None:
    try:
        Pipeline().apply_pack(_observation(), source=_observation_space())
    except NoPackingPhase:
        pass
    else:
        raise AssertionError("apply_pack with no pack should raise NoPackingPhase")
    try:
        Pipeline().apply_unpack({}, source=_output_layout())
    except NoPackingPhase:
        return
    raise AssertionError("apply_unpack with no unpack should raise NoPackingPhase")


# --- verify covers packing when present, unchanged when absent ----------------


def test_check_compatibility_ignores_packing() -> None:
    # A pipeline with packing but an empty channel seam is still a clean match: the
    # packing phase is not folded into matching (it stays at the channel seam).
    pack = PackToNativeLayout(_input_layout())
    unpack = UnpackFromNativeLayout(_output_layout(), _ee())
    report = check_compatibility(_policy(), _benchmark(), Pipeline(pack=pack, unpack=unpack))
    assert report.ok
    # No pipeline -> identical verdict (packing did not change matching).
    assert report.status == check_compatibility(_policy(), _benchmark(), Pipeline()).status


def test_pipeline_rejects_one_sided_packing() -> None:
    # The packing phase is a round trip: a pipeline with exactly one of pack/unpack
    # would pass verify (which probes `pack or unpack`) yet fall through to
    # session.infer at serve time. Construction must reject the one-sided case.
    pack = PackToNativeLayout(_input_layout())
    unpack = UnpackFromNativeLayout(_output_layout(), _ee())
    try:
        Pipeline(pack=pack)
    except ValueError:
        pass
    else:
        raise AssertionError("a pack-only pipeline should raise ValueError")
    try:
        Pipeline(unpack=unpack)
    except ValueError:
        pass
    else:
        raise AssertionError("an unpack-only pipeline should raise ValueError")
    # Both-or-neither both pass.
    Pipeline()
    Pipeline(pack=pack, unpack=unpack)


def test_verify_exercises_packing() -> None:
    pack = PackToNativeLayout(_input_layout())
    unpack = UnpackFromNativeLayout(_output_layout(), _ee())
    report = verify(_policy(), _benchmark(), Pipeline(pack=pack, unpack=unpack))
    assert report.ok, report.reasons
    names = {c.name for c in report.checks}
    assert "packing.observation" in names
    assert "packing.action" in names
    obs_check = next(c for c in report.checks if c.name == "packing.observation")
    assert obs_check.passed, obs_check.detail
    act_check = next(c for c in report.checks if c.name == "packing.action")
    assert act_check.passed, act_check.detail
    # The irreducible declaration-vs-checkpoint residual is surfaced as not_checked.
    assert any("checkpoint metadata" in n for n in report.not_checked)


def test_verify_unchanged_without_packing() -> None:
    report = verify(_policy(), _benchmark(), Pipeline())
    assert report.ok
    names = {c.name for c in report.checks}
    assert "packing.observation" not in names
    assert "packing.action" not in names
    assert not any("checkpoint metadata" in n for n in report.not_checked)


def test_verify_surfaces_bad_layout() -> None:
    # An input layout naming a CAMERA the observation lacks fails the pack probe.
    # Cameras are enumerated in the consumed ObservationSpace, so a missing one is a
    # genuine, detectable structural mismatch (unlike an unmodeled STATE channel,
    # which the probe synthesizes — see test_verify_synthesizes_unmodeled_state).
    bad = NativeLayout(
        entries=(LayoutEntry(key="missing", source=SourceKind.CAMERA, source_name="nope"),)
    )
    pack = PackToNativeLayout(bad)
    unpack = UnpackFromNativeLayout(_output_layout(), _ee())
    report = verify(_policy(), _benchmark(), Pipeline(pack=pack, unpack=unpack))
    assert not report.ok
    obs_check = next(c for c in report.checks if c.name == "packing.observation")
    assert not obs_check.passed


def test_verify_synthesizes_unmodeled_state() -> None:
    # A layout may pull a STATE channel the proprioception spec does not model — a
    # mobile-base pose, a relative-EE quaternion the checkpoint consumes alongside the
    # declared ee_pose (the RLDX RoboCasa case). The channel seam leaves
    # these unmodeled, so the probe can't synthesize them from the spec; verify instead
    # fills a length-correct value for each layout-named state channel, sized from the
    # entry's declared windows, so the structural pack probe is exercised end to end.
    #
    # A key sourced only from a synthesized channel cannot be verified against the
    # declared contract, so it must not count toward a green "packing checked"; it is
    # reported under not_checked instead. The pack still runs, so report.ok holds, but
    # the synthesized key is not blessed.
    layout = NativeLayout(
        entries=(
            LayoutEntry(
                key="state.base_position",
                source=SourceKind.STATE,
                source_name="base_position",
                ops=(Slice(start=0, stop=3), BatchAxis(batch=True, time=True)),
            ),
        )
    )
    pack = PackToNativeLayout(layout)
    unpack = UnpackFromNativeLayout(_output_layout(), _ee())
    report = verify(_policy(), _benchmark(), Pipeline(pack=pack, unpack=unpack))
    assert report.ok, report.reasons
    obs_check = next(c for c in report.checks if c.name == "packing.observation")
    assert obs_check.passed, obs_check.detail
    # The synthesized channel is reported as not-checked, not blessed as packed.
    assert any("base_position" in n for n in report.not_checked), report.not_checked
    # The check's detail must not over-claim the synthesized key as checked.
    assert "state.base_position" not in obs_check.detail or "not checked" in obs_check.detail
    # With only a synthesized-sourced key, nothing contract-modeled was checked.
    assert "0 contract-modeled native key" in obs_check.detail


def _single_array_output_layout() -> NativeLayout:
    # A single-array output (Cosmos's chunk[step][:7]): one entry with source_name
    # None reads the chunk array itself, sliced to the 7-D EE action.
    return NativeLayout(
        entries=(
            LayoutEntry(
                key="action", source=SourceKind.STATE, source_name=None, ops=(Slice(stop=7),)
            ),
        )
    )


def test_verify_exercises_single_array_unpack() -> None:
    # The single-array unpack path (source_name None) was unreachable through verify:
    # the synthetic-output builder skipped None-named entries, yielding {} and an
    # IndexError when the unpack adapter indexed an empty array. verify must now
    # synthesise a real (chunk, dim) array and exercise the path.
    unpack = UnpackFromNativeLayout(_single_array_output_layout(), _ee())
    pack = PackToNativeLayout(_input_layout())
    report = verify(_policy(), _benchmark(), Pipeline(pack=pack, unpack=unpack))
    assert report.ok, report.reasons
    act_check = next(c for c in report.checks if c.name == "packing.action")
    assert act_check.passed, act_check.detail


def test_verify_shape_dtype_mismatch_is_caught() -> None:
    # A layout that declares uint8 for a key the source is float32: the pack
    # check derives the declared dtype and compares it, so a drift between layout and
    # applier is caught rather than blessed by a run that merely did not raise. Here
    # the applier honours the cast, so this confirms the shape/dtype assertion is
    # exercised and passes for a correct layout.
    layout = NativeLayout(
        entries=(
            LayoutEntry(
                key="video.image",
                source=SourceKind.CAMERA,
                source_name="agentview",
                ops=(DtypeCast(dtype="uint8", contiguous=True), BatchAxis(batch=True)),
            ),
        )
    )
    unpack = UnpackFromNativeLayout(_output_layout(), _ee())
    report = verify(
        _policy(), _benchmark(), Pipeline(pack=PackToNativeLayout(layout), unpack=unpack)
    )
    assert report.ok, report.reasons
    obs_check = next(c for c in report.checks if c.name == "packing.observation")
    assert obs_check.passed
    assert "shapes and dtypes" in obs_check.detail


def test_verify_single_array_slice_assemble_layout() -> None:
    # A single-array output where a Slice precedes an Assemble: the Assemble must
    # window the SLICED working array, not the original. verify's independent expected
    # values must track the same left-to-right op order, so a correct layout passes.
    # chunk[step][:7] then assemble [gripper(6:7), pos(0:3), rot(3:6)] = 7-D reordered.
    layout = NativeLayout(
        entries=(
            LayoutEntry(
                key="action",
                source=SourceKind.STATE,
                source_name=None,
                ops=(
                    Slice(stop=7),
                    Assemble(
                        components=(
                            Component(key="g", start=6, stop=7),
                            Component(key="p", start=0, stop=3),
                            Component(key="r", start=3, stop=6),
                        )
                    ),
                ),
            ),
        )
    )
    unpack = UnpackFromNativeLayout(layout, _ee())
    pack = PackToNativeLayout(_input_layout())
    report = verify(_policy(), _benchmark(), Pipeline(pack=pack, unpack=unpack))
    assert report.ok, report.reasons
    act_check = next(c for c in report.checks if c.name == "packing.action")
    assert act_check.passed, act_check.detail


def test_verify_output_value_check_rejects_unhandled_op() -> None:
    # verify's output value-check only models Slice/Assemble. An output entry using a
    # different op (here DtypeCast) would silently compute a WRONG expected value, so
    # verify must refuse rather than bless. The check surfaces as a failed packing.action.
    layout = NativeLayout(
        entries=(
            LayoutEntry(
                key="action",
                source=SourceKind.STATE,
                source_name=None,
                ops=(Slice(stop=7), DtypeCast(dtype="float32")),
            ),
        )
    )
    unpack = UnpackFromNativeLayout(layout, _ee())
    pack = PackToNativeLayout(_input_layout())
    report = verify(_policy(), _benchmark(), Pipeline(pack=pack, unpack=unpack))
    act_check = next(c for c in report.checks if c.name == "packing.action")
    assert not act_check.passed
    assert "DtypeCast" in act_check.detail or "does not interpret" in act_check.detail


def test_verify_catches_swapped_output_keys() -> None:
    # A buggy unpack adapter that reads the keys in the WRONG order. The synthetic
    # output gives each key a per-key-distinct band, so a reorder lands the wrong
    # magnitudes at each offset — caught by the per-offset comparison, which a uniform
    # ramp could not distinguish.
    class SwappingUnpack(UnpackFromNativeLayout):
        def adapt(self, raw_output, /, *, source, step=0):  # type: ignore[no-untyped-def]
            action = super().adapt(raw_output, source=source, step=step)
            vals = list(action.values)
            vals[0], vals[1] = vals[1], vals[0]  # swap x and y — same length, wrong order
            return Action.from_array(vals)

    unpack = SwappingUnpack(_output_layout(), _ee())
    pack = PackToNativeLayout(_input_layout())
    report = verify(_policy(), _benchmark(), Pipeline(pack=pack, unpack=unpack))
    assert not report.ok
    act_check = next(c for c in report.checks if c.name == "packing.action")
    assert not act_check.passed
    assert "swapped or mis-windowed" in act_check.detail
