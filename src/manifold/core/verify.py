"""Confirm a pairing by running synthetic probe data through its pipeline.

Where `check_compatibility` proves the specs line up, `verify` pushes *asymmetric*
probe data with known semantics (a non-origin position, a non-identity rotation, an
open gripper, per-pixel distinct frames) through the pipeline and checks each
value's *meaning* survived. An all-zeros probe cannot reveal a polarity flip or an
element reorder; an asymmetric one can.

It returns a coverage manifest (`VerifyReport`), not a verdict: each exercised
thing is a passed/failed `VerifyCheck`, and what it could not check is listed under
`not_checked`. Only meaning-preserving conversions (rotation re-encode, gripper
polarity) can be asserted; anything that intentionally changes meaning or lacks a
synthetic ground truth — a camera flip/channel swap, a resize, a frame rebase,
normalization, real env/policy behaviour — is reported as not_checked. Shape and
dtype are still exercised on every channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from manifold.core.benchmark import Benchmark
from manifold.core.check import check_compatibility
from manifold.core.conventions import (
    GripperFormat,
    RotationFormat,
    ee_step_layout,
)
from manifold.core.embodiment import (
    EEActionSpace,
    EEObservationSpec,
    Proprioception,
    ValueSpec,
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
from manifold.core.pipeline import Pipeline
from manifold.core.policy import PolicySignature
from manifold.core.state import PipelineState
from manifold.core.values import Action, Observation
from manifold.lib.gripper import from_openness, openness

if TYPE_CHECKING:
    from manifold.core.pipeline import Pack, Unpack

# NOTE: manifold.lib.rotation is not imported at module level to avoid the circular
# import:  core/__init__ -> verify -> lib.rotation -> core.conventions -> core/__init__.
# Each function that needs it uses a lazy local import instead.


# Tolerances for the meaning-preserving comparisons. Rotations are compared by
# geodesic angle in radians (~0.06 degrees), positions and gripper openness by
# absolute value. They sit well above float32 round-trip noise yet far below any
# real semantic error (a flip is pi radians; an inverted gripper is openness 1.0).
_ROTATION_TOL_RAD = 1e-3
_POSITION_TOL = 1e-4
_GRIPPER_TOL = 1e-3

# The asymmetric probe values. A non-origin position and a non-identity rotation,
# built canonically so they decode to a known meaning on the far side. The
# rotation vector is a generic non-axis-aligned, sub-pi rotation: it
# avoids the gimbal-lock and 2pi-wrap degeneracies a probe near identity or near a
# singularity would mask, so a real reorder shows up as a large geodesic angle.
_PROBE_POSITION = (0.11, -0.22, 0.33)
_PROBE_ROTVEC = (0.3, -0.5, 1.2)


# The standing not-checked entries, present on every dynamic run: the failure
# classes outside an adapter's and a synthetic probe's reach (ADR-0001, decision
# 5). Normalization is policy-internal; live behaviour needs a real rollout.
_STANDING_NOT_CHECKED = (
    "normalization: policy-internal, out of reach of an offline probe",
    "env/policy behaviour: requires a live rollout",
)


def _fail(checks: list[VerifyCheck], name: str, message: str, exc: Exception) -> None:
    """Record a failed check for a probe that raised: ``"{message} ({exc})"``."""
    checks.append(VerifyCheck(name, False, f"{message} ({exc})"))


def _run_packing(
    policy: PolicySignature,
    pipeline: Pipeline,
    checks: list[VerifyCheck],
    not_checked: list[str],
) -> None:
    """Exercise the native-packing phase on probe data, recording coverage.

    Packing is structural only (no convention math), so there is no meaning-preserving
    invariant to assert; `verify` proves the declared layout runs and produces the
    declared keys, shapes, dtypes, and offsets. Whether those match the checkpoint's
    true metadata is irreducible offline and listed under `not_checked`.
    """
    if pipeline.pack is not None:
        _run_pack(policy, pipeline.pack, checks, not_checked)
    if pipeline.unpack is not None:
        _run_unpack(policy, pipeline.unpack, checks)
    not_checked.append(
        "native packing vs checkpoint metadata: the declared layout's keys, shapes, "
        "and dtypes are checked, but whether they are the names/forms the real "
        "checkpoint consumes can only be confirmed by running the real model"
    )


def _run_pack(
    policy: PolicySignature,
    pack: Pack,
    checks: list[VerifyCheck],
    not_checked: list[str],
) -> None:
    """Fold a probe observation into the model's tensor dict; assert keys, shapes, dtypes.

    The probe carries per-slot distinct values; expected shapes/dtypes are derived
    independently from the layout (`_declared_key_specs`), so the check tests
    structure rather than only that the run did not raise. A layout may pull a STATE channel
    the proprioception spec does not model (a mobile-base pose, a relative-EE quaternion):
    `_augment_probe_for_layout` synthesizes a length-correct value, but keys sourced
    only from such a channel cannot be independently confirmed, so they are reported
    under `not_checked` and excluded from the passed tally.
    """
    consumed = policy.observation_space
    observation = _probe_observation(consumed)
    layout = getattr(pack, "layout", None)
    if layout is not None:
        # Augment the probe with a length-correct value for each layout-named STATE
        # channel the proprio spec omits, so the structural pack can run.
        observation, synthesized = _augment_probe_for_layout(observation, layout)
    else:
        synthesized = set()

    try:
        packed = pack.adapt(observation, source=consumed)
    except Exception as exc:  # the layout's structural map failed on probe data
        _fail(checks, "packing.observation", "pack failed on probe data", exc)
        return
    if layout is None:
        checks.append(
            VerifyCheck("packing.observation", True, f"packed {len(packed)} native key(s)")
        )
        return
    if not _check_packed_keys(layout, packed, checks):
        return
    _check_packed_specs(observation, layout, packed, synthesized, checks, not_checked)


def _check_packed_keys(
    layout: NativeLayout, packed: dict[str, object], checks: list[VerifyCheck]
) -> bool:
    """Assert the packed keys match the layout's declared keys; record a check if not."""
    expected = set(layout.keys())
    actual = set(packed)
    if expected != actual:
        checks.append(
            VerifyCheck(
                "packing.observation",
                False,
                f"packed keys {sorted(actual)} do not match the declared layout {sorted(expected)}",
            )
        )
        return False
    return True


def _check_packed_specs(
    observation: Observation,
    layout: NativeLayout,
    packed: dict[str, object],
    synthesized: set[str],
    checks: list[VerifyCheck],
    not_checked: list[str],
) -> None:
    """Compare each packed tensor's shape/dtype (or the instruction container) to the layout.

    Keys sourced only from a synthesized channel are exercised structurally but
    reported under `not_checked` rather than counted as checked.
    """
    expected = set(layout.keys())
    synthesized_keys = _keys_from_synthesized(layout, synthesized)
    instruction_keys = {e.key for e in layout.entries if e.source is SourceKind.INSTRUCTION}
    checked_keys = expected - synthesized_keys - instruction_keys

    declared = _declared_key_specs(observation, layout)
    # Accumulate every mismatched key rather than bailing on the first, so one run
    # reports all the broken keys instead of one per run.
    failed = False
    for key in sorted(checked_keys):
        arr = np.asarray(packed[key])
        want_shape, want_dtype = declared[key]
        if arr.shape != want_shape or np.dtype(arr.dtype) != want_dtype:
            failed = True
            checks.append(
                VerifyCheck(
                    f"packing.observation.{key}",
                    False,
                    f"key {key!r} packed to shape {arr.shape} dtype {arr.dtype}; "
                    f"the layout declares shape {want_shape} dtype {want_dtype}",
                )
            )
    for key in sorted(instruction_keys):
        want = (observation.instruction or "",)
        if packed[key] != want:
            failed = True
            checks.append(
                VerifyCheck(
                    f"packing.observation.{key}",
                    False,
                    f"instruction key {key!r} packed to {packed[key]!r}; expected {want!r}",
                )
            )
    for channel in sorted(synthesized):
        not_checked.append(
            f"native packing of layout channel {channel!r}: not in the policy's declared "
            f"proprioception spec, so its probe value was synthesized from the layout's own "
            f"declared width — verify cannot confirm it, and PackToNativeLayout.adapt raises "
            f"if it is absent from the real observation"
        )
    if not failed:
        checks.append(
            VerifyCheck(
                "packing.observation",
                True,
                f"layout packed {len(checked_keys)} contract-modeled native key(s) with the "
                f"declared shapes and dtypes: {sorted(checked_keys)}"
                + (
                    f" ({len(synthesized_keys)} key(s) from synthesized channels not checked: "
                    f"{sorted(synthesized_keys)})"
                    if synthesized_keys
                    else ""
                ),
            )
        )


def _run_unpack(
    policy: PolicySignature,
    unpack: Unpack,
    checks: list[VerifyCheck],
) -> None:
    """Read a synthetic model-output chunk into an Action; assert length and offsets.

    The synthetic output gives each source key a per-key-distinct base value, so the
    unpacked action's per-component values reveal a swapped or mis-windowed key as a
    wrong magnitude, not just a length that happens to fit.
    """
    layout = getattr(unpack, "layout", None)
    if layout is None:
        checks.append(VerifyCheck("packing.action", False, "unpack adapter has no layout to probe"))
        return
    step = 0
    raw = _synthetic_model_output(layout, step=step)
    try:
        action = unpack.adapt(raw, source=layout, step=step)
        policy.action_space.validate_value(action.values)
    except Exception as exc:  # the output layout's read failed, or produced a wrong length
        _fail(checks, "packing.action", "unpack failed on synthetic output", exc)
        return

    try:
        expected = _expected_unpacked_values(layout, step=step)
    except NotImplementedError as exc:
        # The output layout uses an op verify's value-check cannot model;
        # surface it as a failed check rather than guessing (or crashing verify).
        checks.append(VerifyCheck("packing.action", False, str(exc)))
        return
    actual = [float(v) for v in action.values]
    if len(actual) != len(expected):
        checks.append(
            VerifyCheck(
                "packing.action",
                False,
                f"unpacked {len(actual)} values; the layout declares {len(expected)}",
            )
        )
        return
    if not np.allclose(actual, expected, atol=_POSITION_TOL):
        checks.append(
            VerifyCheck(
                "packing.action",
                False,
                f"unpacked values do not match the declared offsets "
                f"(got {actual}, expected {expected}): a swapped or mis-windowed key",
            )
        )
        return
    checks.append(
        VerifyCheck(
            "packing.action",
            True,
            "output layout read into an Action of the policy's action length, "
            "with each key landing at its declared offset",
        )
    )


def _synthetic_model_output(
    layout: NativeLayout, *, step: int = 0
) -> dict[str, np.ndarray] | np.ndarray:
    """A raw model-output chunk shaped to an output `NativeLayout`'s entries.

    Two output shapes occur: a per-key dict of
    `(1, n_steps, width)` float32 tensors when every entry declares its source, or a
    single `(n_steps, width)` array when any entry reads the chunk itself
    (`source_name is None`, e.g. Cosmos's `chunk[step][:7]`). Values are a
    deterministic ramp offset per source key so a wrong key/offset read shows as a
    wrong magnitude; each row adds its index so the unpacked step is distinguishable.
    """
    n_steps = step + 2  # at least two steps, so a wrong-step read is visible.
    if any(entry.source_name is None for entry in layout.entries):
        width = max((_entry_width(entry) for entry in layout.entries), default=1)
        rows = [_key_ramp(0, width) + float(s) for s in range(n_steps)]
        return np.stack(rows).astype(np.float32)
    bands = _source_bands(layout)
    raw: dict[str, np.ndarray] = {}
    for entry in layout.entries:
        assert entry.source_name is not None
        # Size the source tensor to the widest entry reading it, so two entries that
        # window the same source key (a shared tensor) agree rather than clobber.
        width = max(_entry_width(e) for e in layout.entries if e.source_name == entry.source_name)
        base = _key_ramp(bands[entry.source_name], width)
        rows = [base + float(s) for s in range(n_steps)]
        raw[entry.source_name] = np.stack(rows).astype(np.float32).reshape(1, n_steps, width)
    return raw


def _source_bands(layout: NativeLayout) -> dict[str, int]:
    """A distinct ramp band per distinct source key, in first-appearance order.

    Banding by source identity (not entry position) means two entries reading windows
    of the same source tensor share its band, matching the applier; a swapped or
    mis-windowed read still lands in the wrong band and is caught.
    """
    bands: dict[str, int] = {}
    for entry in layout.entries:
        assert entry.source_name is not None
        if entry.source_name not in bands:
            bands[entry.source_name] = len(bands)
    return bands


def _key_ramp(band_index: int, width: int) -> np.ndarray:
    """The per-source base ramp ``[100*band_index, 100*band_index + width)``.

    Bands are spaced 100 apart so no two windows overlap for any layout in the
    catalog; a read of the wrong source lands far from the expected value.
    """
    return np.arange(width, dtype=np.float32) + 100.0 * band_index


def _expected_unpacked_values(layout: NativeLayout, *, step: int) -> list[float]:
    """The flat per-step values the unpack adapter must produce from the synthetic chunk.

    Independently recomputes, in declaration order, the windows each entry reads,
    mirroring `_synthetic_model_output`, so a swapped or mis-windowed key shows as a
    mismatch. Only `Slice` and `Assemble` are interpreted (the only ops the catalog's
    outputs use); an output entry using any other op raises, since silently ignoring
    it (a rounding `DtypeCast`, an axis-adding `BatchAxis`, a fan-out `Split`) would
    compute a wrong expectation and bless a real bug.
    """
    single_array = any(entry.source_name is None for entry in layout.entries)
    bands = {} if single_array else _source_bands(layout)
    out: list[float] = []
    for entry in layout.entries:
        if single_array:
            width = _entry_width(entry)
            band_index = 0
        else:
            assert entry.source_name is not None
            width = max(
                _entry_width(e) for e in layout.entries if e.source_name == entry.source_name
            )
            band_index = bands[entry.source_name]
        row = _key_ramp(band_index, width) + float(step)
        # Track the current working window the same way the applier does (ops apply
        # left to right on a single working array), so a Slice that precedes an
        # Assemble narrows what the Assemble then windows.
        windows: list[np.ndarray] | None = None
        working = row
        for op in entry.ops:
            if isinstance(op, Slice):
                working = working[op.start : op.stop]
            elif isinstance(op, Assemble):
                windows = [working[c.start : c.stop] for c in op.components]
            else:
                # An output op the value-check does not model. Refuse to guess rather
                # than compute a wrong expectation that would bless a packing bug.
                raise NotImplementedError(
                    f"verify's output value-check does not interpret {type(op).__name__} on an "
                    f"unpack layout entry (key {entry.key!r}); it handles only Slice and Assemble. "
                    f"Extend _expected_unpacked_values and _synthetic_model_output before using it."
                )
        if windows is not None:
            out.extend(float(v) for w in windows for v in w)
        else:
            out.extend(float(v) for v in working)
    return out


def _entry_width(entry: LayoutEntry) -> int:
    """A feature width wide enough to span every window an output entry reads.

    Reads the largest `stop` across the entry's slice/assemble/split windows; a None
    `stop` (to-the-end) falls back to a small default. Sizes the probe tensor only,
    never the real shape.
    """
    stops: list[int] = [1]
    for op in entry.ops:
        if isinstance(op, Slice):
            stops.append(op.stop if op.stop is not None else op.start + 1)
        elif isinstance(op, (Assemble, Split)):
            for component in op.components:
                assert isinstance(component, Component)
                stops.append(component.stop if component.stop is not None else component.start + 1)
    return max(stops)


def _declared_key_specs(
    observation: Observation, layout: NativeLayout
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    """The shape and dtype the layout declares for every key it emits.

    Derived independently of the structural applier, so comparing it to what the pack
    adapter produced catches a drift between the layout and the applier rather than
    merely a run that did not raise.
    """
    specs: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
    for entry in layout.entries:
        source = _declared_source_array(observation, entry.source, entry.source_name)
        specs.update(_declared_entry_specs(source, entry))
    return specs


def _declared_entry_specs(
    source: np.ndarray, entry: LayoutEntry
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    """The shape and dtype of every key one layout entry emits, derived from `source`.

    Ops apply left to right on the single working array, exactly as the applier does:
    a `Slice`/`BatchAxis` before a `Split`/`Assemble` is reflected in the shape that
    fan-out then windows; `dtype` is shared across the entry's keys (a `DtypeCast`
    applies uniformly). A later fan-out after a `Split` is unsupported (see `LayoutOp`).
    """
    shapes: dict[str, tuple[int, ...]] = {entry.key: source.shape}
    dtype = np.dtype(source.dtype)
    for op in entry.ops:
        if isinstance(op, Slice):
            shapes = {k: _slice_last_axis(s, op.start, op.stop) for k, s in shapes.items()}
        elif isinstance(op, DtypeCast):
            dtype = np.dtype(op.dtype)
        elif isinstance(op, BatchAxis):
            prefix = (1,) * (int(op.batch) + int(op.time))
            shapes = {k: prefix + s for k, s in shapes.items()}
        elif isinstance(op, Assemble):
            (current,) = shapes.values()  # fan-out operates on the single working array
            width = sum(_window_len(current[-1], c.start, c.stop) for c in op.components)
            shapes = {entry.key: (*current[:-1], width)}
        elif isinstance(op, Split):
            (current,) = shapes.values()
            shapes = {c.key: _slice_last_axis(current, c.start, c.stop) for c in op.components}
    return {key: (shape, dtype) for key, shape in shapes.items()}


def _declared_source_array(
    observation: Observation, source: SourceKind, name: str | None
) -> np.ndarray:
    """The raw channel array the layout pulls — the starting point for shape derivation."""
    if source is SourceKind.INSTRUCTION:
        return np.asarray([observation.instruction or ""], dtype=object)
    assert name is not None
    bag = observation.sensors if source is SourceKind.CAMERA else observation.state
    return np.asarray(bag[name])


def _slice_last_axis(shape: tuple[int, ...], start: int, stop: int | None) -> tuple[int, ...]:
    """`shape` with its last axis windowed to ``[start:stop)``."""
    return (*shape[:-1], _window_len(shape[-1], start, stop))


def _window_len(length: int, start: int, stop: int | None) -> int:
    """The length of a ``[start:stop)`` window of an axis of length `length`."""
    return len(range(*slice(start, stop).indices(length)))


def _run_action(
    policy_action: ValueSpec,
    bench_action: ValueSpec,
    pipeline: Pipeline,
    checks: list[VerifyCheck],
) -> None:
    # Fold an asymmetric probe action through the action chain and compare the
    # output's meaning against the input's. A fresh state per call exercises any
    # stateful adapter (which raises without it).
    try:
        values = _probe_action_values(policy_action)
        result = pipeline.apply_action(
            Action.from_array(values), source=policy_action, state=PipelineState()
        )
        bench_action.validate_value(result.values)
    except Exception as exc:  # an adapter's data map failed on probe data
        _fail(checks, "action", "action pipeline failed on probe data", exc)
        return
    checks.append(VerifyCheck("action.length", True, "output length fits the benchmark spec"))

    # Meaning comparison is only valid for end-effector pairs (the rotation/gripper
    # conventions live on EEActionSpace). A joint or unified target is length-only.
    if not (isinstance(policy_action, EEActionSpace) and isinstance(bench_action, EEActionSpace)):
        return
    out = [float(v) for v in result.values]

    # Compare every chunk step (each carries a distinct position and gripper openness),
    # so a step-specific corruption is caught; the check names carry the step index.
    src_pos, src_rot, src_grip = ee_step_layout(policy_action.rotation, policy_action.gripper)
    tgt_pos, tgt_rot, tgt_grip = ee_step_layout(bench_action.rotation, bench_action.gripper)
    src_step = src_pos + src_rot + src_grip
    tgt_step = tgt_pos + tgt_rot + tgt_grip
    for i in range(policy_action.chunk_size):
        suffix = "" if policy_action.chunk_size == 1 else f".step{i}"
        in_block = values[i * src_step : (i + 1) * src_step]
        out_block = out[i * tgt_step : (i + 1) * tgt_step]
        _compare_position(
            f"action.position{suffix}", in_block[:src_pos], out_block[:tgt_pos], checks
        )
        _compare_rotation(
            f"action.rotation{suffix}",
            in_block[src_pos : src_pos + src_rot],
            policy_action.rotation,
            out_block[tgt_pos : tgt_pos + tgt_rot],
            bench_action.rotation,
            checks,
        )
        if policy_action.gripper is not None and bench_action.gripper is not None:
            _compare_gripper(
                f"action.gripper{suffix}",
                in_block[src_pos + src_rot],
                policy_action.gripper,
                out_block[tgt_pos + tgt_rot],
                bench_action.gripper,
                checks,
            )

    # A single-step probe only sees the open gripper, so a separate closed-gripper
    # probe gives the check both ends of the range (covering the chunk_size-1 case).
    if policy_action.gripper is not None and bench_action.gripper is not None:
        _check_closed_gripper(policy_action, bench_action, pipeline, checks)


def _check_closed_gripper(
    policy_action: EEActionSpace,
    bench_action: EEActionSpace,
    pipeline: Pipeline,
    checks: list[VerifyCheck],
) -> None:
    """Run a closed-gripper probe and compare the first step's gripper openness."""
    from manifold.lib.rotation import convert  # lazy — avoids circular import at module load

    assert policy_action.gripper is not None and bench_action.gripper is not None
    src_pos, src_rot, _ = ee_step_layout(policy_action.rotation, policy_action.gripper)
    tgt_pos, tgt_rot, _ = ee_step_layout(bench_action.rotation, bench_action.gripper)
    rotation = convert(_PROBE_ROTVEC, RotationFormat.AXIS_ANGLE, policy_action.rotation)
    closed = from_openness(0.0, policy_action.gripper)
    step = [*_PROBE_POSITION[:src_pos], *rotation, closed]
    try:
        result = pipeline.apply_action(
            Action.from_array(step * policy_action.chunk_size),
            source=policy_action,
            state=PipelineState(),
        )
    except Exception as exc:
        _fail(checks, "action.gripper.closed", "closed-gripper probe failed", exc)
        return
    out = [float(v) for v in result.values]
    _compare_gripper(
        "action.gripper.closed",
        step[src_pos + src_rot],
        policy_action.gripper,
        out[tgt_pos + tgt_rot],
        bench_action.gripper,
        checks,
    )


def _run_observation(
    bench_obs: ObservationSpace,
    policy_consumes: ObservationSpace,
    pipeline: Pipeline,
    checks: list[VerifyCheck],
    not_checked: list[str],
) -> None:
    observation = _probe_observation(bench_obs)
    in_ee = bench_obs.proprioception.ee_pose
    in_ee_values = (
        list(observation.state["ee_pose"])
        if in_ee is not None and "ee_pose" in observation.state
        else None
    )
    try:
        # A fresh state per call exercises any stateful adapter (e.g. a
        # frame-history buffer): the single synthetic frame repeat-pads into the
        # declared clip shape, which validate_value below then checks.
        result = pipeline.apply_observation(observation, source=bench_obs, state=PipelineState())
        policy_proprio = policy_consumes.proprioception
        if policy_proprio.ee_pose is not None:
            policy_proprio.ee_pose.validate_value(result.state["ee_pose"])
        if policy_proprio.joint_pos is not None:
            policy_proprio.joint_pos.validate_value(result.state["joint_pos"])
        for wanted in policy_consumes.cameras:
            wanted.validate_value(result.sensors[wanted.name])
    except Exception as exc:  # an adapter's data map failed on probe data
        _fail(checks, "observation", "observation pipeline failed on probe data", exc)
        return

    # The proprioception ee_pose carries the same meaning-preserving rotation
    # invariant as the action side. Its gripper field is finger-joint *qpos*
    # (position), not a commanded polarity value, so there is no polarity check
    # here — only position and rotation. Compare what the policy actually consumes.
    out_ee = policy_consumes.proprioception.ee_pose
    if in_ee is not None and out_ee is not None and in_ee_values is not None:
        out_ee_values = [float(v) for v in result.state["ee_pose"]]
        _compare_ee_pose(in_ee, in_ee_values, out_ee, out_ee_values, checks, not_checked)

    # Each consumed camera: shape/dtype IS exercised (validate_value above did not
    # raise), so record it passed. Content — orientation, channel order — is not
    # checkable from synthetic data, which cannot tell up from down or RGB from BGR
    # without a ground-truth scene; list it under not_checked.
    for wanted in policy_consumes.cameras:
        checks.append(
            VerifyCheck(
                f"observation.camera.{wanted.name}.shape",
                True,
                f"shape {wanted.shape} and dtype {wanted.dtype} survive the chain",
            )
        )
        not_checked.append(
            f"camera {wanted.name!r} orientation/channel content: "
            "synthetic data cannot reveal a flip or a channel swap"
        )


def _compare_ee_pose(
    in_spec: EEObservationSpec,
    in_values: list[float],
    out_spec: EEObservationSpec,
    out_values: list[float],
    checks: list[VerifyCheck],
    not_checked: list[str],
) -> None:
    """Compare the position and rotation meaning of an ee_pose round-trip.

    The position is always checked. The rotation is only checked when the frame is
    unchanged across the chain: a frame rebase (e.g. `FrameRebaseAdapter`) is a
    static reorientation with no coordinate-free ground truth, so comparing the two
    frames' rotations by geodesic angle would falsely report a large change. When
    the frame changed, the rotation is routed to `not_checked` instead of compared.
    """
    in_pos, in_rot, _ = ee_step_layout(in_spec.rotation, None)
    out_pos, out_rot, _ = ee_step_layout(out_spec.rotation, None)
    _compare_position(
        "observation.ee_pose.position", in_values[:in_pos], out_values[:out_pos], checks
    )
    if in_spec.frame != out_spec.frame:
        not_checked.append(
            f"observation.ee_pose.rotation: static rebase {in_spec.frame} -> "
            f"{out_spec.frame}, no coordinate-free ground truth"
        )
        return
    _compare_rotation(
        "observation.ee_pose.rotation",
        in_values[in_pos : in_pos + in_rot],
        in_spec.rotation,
        out_values[out_pos : out_pos + out_rot],
        out_spec.rotation,
        checks,
    )


def _compare_position(
    name: str, source: list[float], target: list[float], checks: list[VerifyCheck]
) -> None:
    if len(source) != len(target):
        checks.append(VerifyCheck(name, False, f"position length changed: {source} -> {target}"))
        return
    ok = bool(np.allclose(source, target, atol=_POSITION_TOL))
    checks.append(
        VerifyCheck(
            name,
            ok,
            f"position preserved ({source} -> {target})"
            if ok
            else f"position changed ({source} -> {target})",
        )
    )


def _compare_rotation(
    name: str,
    source: list[float],
    source_fmt: RotationFormat,
    target: list[float],
    target_fmt: RotationFormat,
    checks: list[VerifyCheck],
) -> None:
    # Representation-aware: decode both encodings to rotations and measure the
    # geodesic angle between them. Raw float equality must not be used — it would
    # flag quaternion double-cover, axis-angle 2pi wrap, gimbal lock and 6D
    # re-orthonormalisation as failures even when the rotation is identical.
    from manifold.lib.rotation import geodesic_angle  # lazy — avoids circular import at module load

    angle = geodesic_angle(source, source_fmt, target, target_fmt)
    ok = angle <= _ROTATION_TOL_RAD
    checks.append(
        VerifyCheck(
            name,
            ok,
            f"rotation preserved (geodesic {angle:.2e} rad)"
            if ok
            else f"rotation changed by {angle:.4f} rad ({source_fmt} -> {target_fmt})",
        )
    )


def _compare_gripper(
    name: str,
    source: float,
    source_fmt: GripperFormat,
    target: float,
    target_fmt: GripperFormat,
    checks: list[VerifyCheck],
) -> None:
    # Compare by openness, not raw value: an inverted polarity preserves the float
    # but maps an open gripper to closed, which is the exact silent failure.
    src_open = openness(source, source_fmt)
    tgt_open = openness(target, target_fmt)
    ok = abs(src_open - tgt_open) <= _GRIPPER_TOL
    checks.append(
        VerifyCheck(
            name,
            ok,
            f"gripper openness preserved ({src_open:.3f})"
            if ok
            else f"gripper openness changed {src_open:.3f} -> {tgt_open:.3f} (polarity?)",
        )
    )


def _step_openness(step: int) -> float:
    """The probe gripper openness for chunk step `step`: alternating open/closed.

    Alternating polarities across chunk steps means a multi-step probe exercises
    both a fully-open and a fully-closed gripper, so an adapter that preserves open
    but mangles closed (e.g. one that always emits "open") fails on the closed step.
    A single-step probe (chunk_size 1) only sees the open value here; `_run_action`
    runs a separate closed-gripper probe so that case still spans both ends.
    """
    return 1.0 if step % 2 == 0 else 0.0


def _probe_action_values(spec: ValueSpec) -> list[float]:
    """An asymmetric probe action laid out under `spec`.

    For an end-effector action, each chunk step carries the known probe rotation
    (encoded into the spec's format), a per-step-distinct position (the base probe
    position shifted by the step index, so a per-step bug that swaps or drops a
    step is visible), and a per-step-alternating gripper openness. For any other
    action space, a zero-filled value of the right length (length is all that side
    can check).
    """
    from manifold.lib.rotation import convert  # lazy — avoids circular import at module load

    if not isinstance(spec, EEActionSpace):
        return spec.example()
    pos_len, _, _ = ee_step_layout(spec.rotation, spec.gripper)
    rotation = convert(_PROBE_ROTVEC, RotationFormat.AXIS_ANGLE, spec.rotation)
    out: list[float] = []
    for step in range(spec.chunk_size):
        position = [coord + step for coord in _PROBE_POSITION[:pos_len]]
        out.extend(position)
        out.extend(rotation)
        if spec.gripper is not None:
            out.append(from_openness(_step_openness(step), spec.gripper))
    return out


def _probe_observation(contract: ObservationSpace) -> Observation:
    """An asymmetric probe observation laid out under `contract`.

    Proprioception ee_pose carries the known probe position and rotation (encoded
    into its declared format) plus zeroed gripper qpos; any other proprioception
    channel is length-correct zeros (only its length and rotation, where present,
    are checked). Each camera is a deterministic per-pixel gradient so shape is
    genuinely exercised by distinct values rather than a uniform fill, and each
    camera declaring calibration also carries a pose in `extrinsics`. No RNG.
    """
    proprio = contract.proprioception
    state: dict[str, np.ndarray] = {}
    for name in Proprioception.model_fields:
        spec = getattr(proprio, name)
        if spec is None:
            continue
        if name == "ee_pose" and isinstance(spec, EEObservationSpec):
            state[name] = _probe_ee_pose(spec)
        else:
            state[name] = np.asarray(spec.example(), dtype=np.float32)
    sensors = {
        camera.name: _gradient_frame(camera.shape, camera.dtype) for camera in contract.cameras
    }
    instruction = "" if contract.instruction else None
    return Observation(
        state=state,
        sensors=sensors,
        instruction=instruction,
        extrinsics=_probe_extrinsics(contract),
    )


def _probe_extrinsics(contract: ObservationSpace) -> dict[str, np.ndarray]:
    """One 4x4 camera-to-frame pose per calibrated camera, distinct per camera.

    A camera declaring `calibration` means the benchmark publishes that camera's pose
    in every observation (ADR 0008), so a probe that omits it does not lay out the
    contract it claims to: an adapter reading `Observation.extrinsics` fails on the
    probe alone. Each pose is a valid rigid transform — identity rotation, a
    per-camera translation — so the poses are asymmetric like the frames, and
    float64, because a pose is read as geometry rather than fed to a network.
    """
    poses: dict[str, np.ndarray] = {}
    for index, camera in enumerate(contract.cameras):
        if camera.calibration is None:
            continue
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = (index + 1) * np.array([0.1, 0.2, 0.3])
        poses[camera.name] = pose
    return poses


def _augment_probe_for_layout(
    observation: Observation, layout: NativeLayout
) -> tuple[Observation, set[str]]:
    """Add a probe value for each layout-named STATE channel the proprio spec omits.

    A layout may pull state channels the `Proprioception` spec does not model (a
    mobile-base pose, a relative-EE quaternion riding in `obs.state` under its own
    key), which `_probe_observation` cannot synthesize. Each such key is filled with a
    length-correct ramp sized from `_entry_width` so the structural pack can run; only
    the width matters, so the invented content is irrelevant.

    Returns the augmented observation and the set of state channels it synthesized.
    Keys sourced from these are reported as not-checked: an invented value cannot
    confirm anything the layout declares about an unmodeled channel.
    """
    state = dict(observation.state)
    synthesized: set[str] = set()
    for entry in layout.entries:
        if entry.source is not SourceKind.STATE or entry.source_name is None:
            continue
        if entry.source_name in state:
            continue
        width = _entry_width(entry)
        state[entry.source_name] = np.arange(width, dtype=np.float32)
        synthesized.add(entry.source_name)
    return (
        Observation(
            state=state,
            sensors=observation.sensors,
            instruction=observation.instruction,
            extrinsics=observation.extrinsics,
        ),
        synthesized,
    )


def _keys_from_synthesized(layout: NativeLayout, synthesized: set[str]) -> set[str]:
    """The layout keys whose source channel was synthesized (not in the proprio spec).

    A key is synthesized-only when its entry's source channel is one `verify` had to
    invent. Split groups emit per-component keys; an entry sourcing a synthesized
    channel taints all the keys it produces. These keys are excluded from the green
    "packing checked" tally and reported under not_checked.
    """
    tainted: set[str] = set()
    for entry in layout.entries:
        if entry.source is not SourceKind.STATE or entry.source_name not in synthesized:
            continue
        tainted.update(_entry_emitted_keys(entry))
    return tainted


def _entry_emitted_keys(entry: LayoutEntry) -> set[str]:
    """The native keys one layout entry emits (a Split fans into its components)."""
    keys: set[str] = set()
    has_split = False
    for op in entry.ops:
        if isinstance(op, Split):
            has_split = True
            keys.update(component.key for component in op.components)
    if not has_split:
        keys.add(entry.key)
    return keys


def _probe_ee_pose(spec: EEObservationSpec) -> np.ndarray:
    """The probe ee_pose value: known position and rotation, zeroed gripper qpos."""
    from manifold.lib.rotation import convert  # lazy — avoids circular import at module load

    pos_len, _, _ = ee_step_layout(spec.rotation, None)
    rotation = convert(_PROBE_ROTVEC, RotationFormat.AXIS_ANGLE, spec.rotation)
    gripper_qpos = [0.0] * (spec.gripper.dim if spec.gripper is not None else 0)
    return np.asarray([*_PROBE_POSITION[:pos_len], *rotation, *gripper_qpos], dtype=np.float32)


def _gradient_frame(shape: tuple[int, ...], dtype: str) -> np.ndarray:
    """A deterministic per-pixel gradient of the given shape and dtype (no RNG).

    Distinct values across the array exercise shape genuinely — a uniform fill
    cannot tell a transposed or wrongly-strided frame from a correct one — while
    staying reproducible run to run.
    """
    np_dtype = np.dtype(dtype)
    count = int(np.prod(shape)) if shape else 1
    ramp = np.arange(count, dtype=np.float64)
    if np.issubdtype(np_dtype, np.integer):
        # Wrap into a byte range so the values stay distinct yet never overflow the
        # narrowest integer dtype (uint8) the catalog uses; 251 is prime, so the
        # gradient does not align with any row/column stride of a typical frame.
        ramp = np.mod(ramp, 251.0)
    return ramp.reshape(shape).astype(np_dtype)


@dataclass(frozen=True)
class VerifyCheck:
    """One thing `verify` exercised and the outcome.

    `name` locates it in the channel aggregate (e.g. ``"observation.ee_pose.rotation"``,
    ``"action.gripper"``, ``"observation.camera.agentview.shape"``); `passed` is the
    outcome; `detail` is a human-readable note, and is what a serving gate surfaces
    when a check fails.
    """

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class VerifyReport:
    """A coverage manifest: what `verify` checked, what passed, what it could not check.

    This is richer than the static `check`'s `Report`, whose
    COMPATIBLE/INCOMPATIBLE invariants cannot carry a *checked-and-failed* state. A
    `VerifyReport` covers both directions: `checks` are the things actually
    exercised on probe data (each passed or failed), and `not_checked` lists, as
    short reasons, the things `verify` did *not* assert content for — so a clean
    result means "the bridge was exercised," never "the pairing works."

    `static_passed` records whether the upstream `check_compatibility` passed. When
    it did not, the dynamic probes are skipped and the static reasons appear as
    failed checks named ``"static"``; `ok` is False regardless of the (empty)
    dynamic checks.
    """

    checks: tuple[VerifyCheck, ...] = ()
    not_checked: tuple[str, ...] = ()
    static_passed: bool = True

    @property
    def passed(self) -> tuple[VerifyCheck, ...]:
        """The checks that passed."""
        return tuple(check for check in self.checks if check.passed)

    @property
    def failed(self) -> tuple[VerifyCheck, ...]:
        """The checks that failed."""
        return tuple(check for check in self.checks if not check.passed)

    @property
    def ok(self) -> bool:
        """True iff the static check passed and no exercised check failed.

        A pairing with nothing checkable but a clean static pass is `ok` — `verify`
        did not find a broken conversion. It still says nothing about real behaviour;
        that is what `not_checked` is for.
        """
        return self.static_passed and not self.failed

    @property
    def reasons(self) -> tuple[str, ...]:
        """The details of every failed check — what a serving gate emits on rejection."""
        return tuple(check.detail for check in self.failed)

    def summary(self) -> str:
        """A one-line tally, e.g. ``"7 checked, 0 failed, 3 not checked"``."""
        return (
            f"{len(self.checks)} checked, {len(self.failed)} failed, "
            f"{len(self.not_checked)} not checked"
        )


def verify(
    policy: PolicySignature,
    benchmark: Benchmark,
    pipeline: Pipeline | None = None,
) -> VerifyReport:
    """Run asymmetric probe data through the pipeline and report a coverage manifest.

    Runs `check_compatibility` first. On a static failure, returns a `VerifyReport`
    surfacing each static reason as a failed ``"static"`` check and runs no probes
    (a pipeline that does not type-check cannot be exercised). Otherwise it folds
    asymmetric probe data through each side and records a `VerifyCheck` per
    meaning-preserving comparison, plus the `not_checked` entries for everything out
    of an offline probe's reach.
    """
    if pipeline is None:
        pipeline = Pipeline()
    static = check_compatibility(policy, benchmark, pipeline)
    if not static.ok:
        return VerifyReport(
            checks=tuple(VerifyCheck("static", False, reason) for reason in static.reasons),
            static_passed=False,
        )

    checks: list[VerifyCheck] = []
    not_checked: list[str] = []
    _run_observation(
        benchmark.observation_space,
        policy.observation_space,
        Pipeline(observation=pipeline.observation),
        checks,
        not_checked,
    )
    _run_action(
        policy.action_space,
        benchmark.embodiment.action,
        Pipeline(action=pipeline.action),
        checks,
    )
    # When the pairing declares a native-packing phase, exercise it too — fold the
    # observation into the model's tensor dict, and a synthetic model output back
    # into an Action — so the coverage manifest reflects the phase the seam now
    # extends to (ADR-0001). Absent a packing phase (every existing pairing) this is
    # a no-op and verify is unchanged.
    if pipeline.pack is not None or pipeline.unpack is not None:
        _run_packing(policy, pipeline, checks, not_checked)
    not_checked.extend(_STANDING_NOT_CHECKED)
    return VerifyReport(checks=tuple(checks), not_checked=tuple(not_checked))


__all__ = ["VerifyCheck", "VerifyReport", "verify"]
