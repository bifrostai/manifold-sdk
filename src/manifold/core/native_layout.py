"""The native tensor layout a checkpoint's own wire format demands.

`check_compatibility` matches at the coarse channel seam (proprioception, cameras,
instruction), but a checkpoint consumes a flat dict of named tensors with the keys,
dtypes, batch/time axes, and per-key arithmetic its wire format fixes. Bridging the
channel form to that tensor dict is **native packing**, the second pipeline phase
(ADR-0001). A `NativeLayout` is the typed declaration the packing adapter applies: a
small reusable structural algebra carrying **no convention math** (rotation,
gripper, frame, and threshold transforms are phase-one convention adapters), only
the structural ops modeled below as `LayoutOp` variants — rename, `DtypeCast`,
`BatchAxis`, `Slice`, `Split`, `Assemble`.

The layout is a downstream, profile-local target — not a field of
`PolicySignature`, so matching stays at the channel seam. It is consumed by the
packing adapter and by `verify`. A declaration can still disagree with the
checkpoint's true metadata; that residual is irreducible, and `verify` running the
real model is the backstop.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from manifold.lib.compat import StrEnum


class SourceKind(StrEnum):
    """Where one entry's value comes from in an SDK `Observation` (the pack side).

    A channel is named by its role: a camera by its sensor name (`obs.sensors`), a
    proprioception value by its state key (`obs.state`), or the language
    instruction. This is the only place the layout reaches back into the channel
    aggregate; everything after is structural arithmetic on the array it pulls.
    """

    CAMERA = "camera"  # obs.sensors[name] — a camera frame or clip.
    STATE = "state"  # obs.state[name] — a proprioception vector ("ee_pose", "base_position").
    INSTRUCTION = "instruction"  # obs.instruction — the language string.


class ChannelKind(StrEnum):
    """Which channel of an SDK `Observation` an entry feeds (the env→obs direction).

    The mirror of `SourceKind`: where `SourceKind` says *where a value comes from*
    when packing an `Observation` into a native dict, `ChannelKind` says *where a
    value goes* when unpacking a native env dict back into an `Observation`. An
    entry tagged with a `TargetChannel` is read from the env dict (its `source_name`
    naming the env key) and routed into the named channel; several entries targeting
    the same `STATE` role gather (concatenate) into that one state vector.
    """

    CAMERA = "camera"  # obs.sensors[name] — a camera frame, renamed from an env key.
    STATE = "state"  # obs.state[name] — a proprioception role, gathered from N env keys.
    INSTRUCTION = "instruction"  # obs.instruction — the language string.


class TargetChannel(BaseModel):
    """The `Observation` channel one env→obs entry feeds, with a gather order.

    `kind` selects the channel aggregate (a camera, a proprioception state role, or
    the instruction); `name` is the sensor name (`CAMERA`) or state role (`STATE`),
    and is unused for `INSTRUCTION`. `order` sequences entries that target the same
    `STATE` role: the unpack-to-`Observation` adapter concatenates them ascending by
    `order` (ties broken by declaration order), so `eef_pos`(0) + `eef_quat`(1) +
    `gripper_qpos`(2) gather into one `ee_pose` vector in that fixed order — the same
    multi-source gather the action `unpack` does, but routed per channel.

    This is the only new concept the env→`Observation` direction needs; it is
    an optional field on `LayoutEntry` (`target=None` ⇒ a pack/unpack entry, which
    never reads it), so the existing packing layouts and adapters are unaffected.
    """

    model_config = ConfigDict(frozen=True)

    kind: ChannelKind
    name: str | None = None
    order: int = 0


class DtypeCast(BaseModel):
    """Cast a tensor to a target dtype, optionally making it C-contiguous.

    Models `np.ascontiguousarray(x, dtype=np.uint8)` and `x.astype(np.float32)`.
    `contiguous` requests a C-contiguous copy (what a checkpoint that indexes raw
    buffers needs); when False the cast may share memory. The dtype is a numpy dtype
    string (``"uint8"``, ``"float32"``), the same vocabulary `Camera.dtype` already
    uses.
    """

    model_config = ConfigDict(frozen=True)

    op: Literal["dtype_cast"] = "dtype_cast"
    dtype: str
    contiguous: bool = False


class BatchAxis(BaseModel):
    """Prepend a leading batch and/or time axis of size one.

    The checkpoint wire format expects a `(batch, …)` or `(batch, time, …)` tensor;
    the SDK channel value carries neither. `batch` prepends one leading axis, `time`
    a second after it — so `batch=True, time=True` turns a `(dim,)` vector into
    `(1, 1, dim)` (the RLDX per-axis `scalar`/state reshape) and a `(H, W, C)` frame
    or `(n_frames, H, W, C)` clip into `(1, …)` (a `[None]` index). Both default
    off, so an entry that does not need batching omits this op.
    """

    model_config = ConfigDict(frozen=True)

    op: Literal["batch_axis"] = "batch_axis"
    batch: bool = True
    time: bool = False


class Slice(BaseModel):
    """Keep one contiguous ``[start:stop)`` window of the source vector.

    Models `ee[0:3]`, `ee[6:8]`, `chunk[step][:7]` — extracting a sub-vector from a
    flat channel value. `stop` is None for "to the end". The window is taken on the
    last (feature) axis after any earlier op; it is the structural counterpart of a
    convention adapter's per-field math, without semantics of its own.
    """

    model_config = ConfigDict(frozen=True)

    op: Literal["slice"] = "slice"
    start: int = 0
    stop: int | None = None


class Component(BaseModel):
    """One named piece of a source vector, for `Split` and `Assemble`.

    `start`/`stop` is the ``[start:stop)`` window of the entry's single current source
    vector this component spans (a single-element window is a scalar axis, e.g.
    `state.x` at `[0:1]`); `stop` None means to the end.

    `key` is used only by `Split`, where it is the model tensor key the window is
    emitted under. On `Assemble` the components are windows of the same single source
    concatenated in order and `key` is ignored (the assembled vector takes the entry's
    own key) — there is no multi-source assemble; the applier windows the one current
    source.
    """

    model_config = ConfigDict(frozen=True)

    key: str
    start: int = 0
    stop: int | None = None


class Split(BaseModel):
    """Fan one source vector out into several named keys, each a window of it.

    Models the RLDX per-axis flat layout: one `ee_pose` vector becomes `state.x`,
    `state.y`, … `state.gripper`, each a `[start:stop)` window. The split keys
    REPLACE the entry's own key in the produced dict (the entry's `key` is the
    group, not an emitted tensor), and any `dtype_cast`/`batch_axis` on the same
    entry applies to every split component uniformly.
    """

    model_config = ConfigDict(frozen=True)

    op: Literal["split"] = "split"
    components: tuple[Component, ...]


class Assemble(BaseModel):
    """Build one vector by concatenating windows of the current source in a fixed order.

    Models Cosmos's gripper-FIRST `concat(gripper_qpos, eef_pos, eef_quat)`: the
    components are windows of the same single source vector (the entry's current
    working array), concatenated on the feature axis in the listed order — gripper-first
    is just listing the gripper window first. Each component's `key` is ignored here;
    only its `[start:stop)` window matters. The order is the contract; reordering it is
    a different (and silently wrong) layout, which is exactly why it is declared rather
    than implicit.
    """

    model_config = ConfigDict(frozen=True)

    op: Literal["assemble"] = "assemble"
    components: tuple[Component, ...]


# The structural ops, discriminated on `op`. No convention math lives here —
# rotation/gripper/frame transforms are phase-one convention adapters; these only
# rename, reshape, cast, slice, split, and assemble. Ordering within an entry is the
# order they appear in `Entry.ops`, applied left to right.
#
# The applier supports the compositions the profiles use: a `Slice`, `DtypeCast`, or
# `BatchAxis` may appear before or after a `Split`/`Assemble`, and any `DtypeCast`/
# `BatchAxis` following a `Split` applies to every split component uniformly. A
# `Split` or `Assemble` operates on the entry's single current working array, so a
# `Split` followed by another `Split`/`Assemble` (which would need a multi-array
# working set) is not supported — no profile needs it. Keep the fan-out (`Split`/
# `Assemble`) last among the array-reshaping ops in an entry.
LayoutOp = Annotated[
    DtypeCast | BatchAxis | Slice | Split | Assemble,
    Field(discriminator="op"),
]


class LayoutEntry(BaseModel):
    """One model tensor key and how to build its value from an SDK channel.

    `key` is the native tensor key the checkpoint consumes (``"video.image"``,
    ``"proprio"``, ``"state.x"`` for a split group). `source`/`source_name` identify the
    SDK channel the value derives from: a camera by sensor name, a proprioception
    value by state key, or the instruction (which does not need one). `ops` is the
    ordered structural pipeline applied to that channel value (empty ⇒ a pure
    rename). A `Split` op turns the entry into several keys (its `key` then names
    the group); every other op leaves it a single key.

    `target` is the optional env->`Observation` role this entry feeds (ADR-0001, decision 4): when
    set, the same entry runs the OTHER direction -- `source`/`source_name` then names
    the env-dict key the value is read from, the `ops` reshape it, and `target` routes
    the result into an `Observation` channel (a camera, a gathered state role, or the
    instruction). It defaults to `None`, the pack/unpack form, which the packing
    adapters use and never consult `target` for -- so adding it is fully additive.
    """

    model_config = ConfigDict(frozen=True)

    key: str
    source: SourceKind
    source_name: str | None = None
    ops: tuple[LayoutOp, ...] = ()
    target: TargetChannel | None = None

    @model_validator(mode="after")
    def _validate_ops(self) -> LayoutEntry:
        """Reject op compositions the applier cannot execute, at construction time.

        The applier (`adapters/packing/_ops.py`) windows a SINGLE working array after a
        fan-out, and the env->`Observation` adapter routes each entry to exactly one
        channel. Encoding those limits here turns what were opaque runtime `ValueError`s
        (deep in the applier, or in the unpack loop) into a clear construction-time error,
        so a malformed layout fails where it is written rather than mid-rollout.
        """
        # An instruction is read verbatim, in EITHER direction: a pack entry pulling
        # `obs.instruction` (source INSTRUCTION) or an env->obs entry routing into it
        # (target.kind INSTRUCTION, which `unpack_obs` likewise forbids ops on).
        routes_instruction = self.source is SourceKind.INSTRUCTION or (
            self.target is not None and self.target.kind is ChannelKind.INSTRUCTION
        )
        if routes_instruction and self.ops:
            raise ValueError(
                f"INSTRUCTION entry {self.key!r} does not take ops (the language string is read "
                f"verbatim); got {[type(op).__name__ for op in self.ops]}."
            )
        seen_fanout = False
        for op in self.ops:
            if seen_fanout and isinstance(op, (Split, Assemble)):
                raise ValueError(
                    f"entry {self.key!r}: a {type(op).__name__} cannot follow a Split — the "
                    f"applier windows one working array after a fan-out. Keep the fan-out last."
                )
            if isinstance(op, Split):
                seen_fanout = True
        if self.target is not None and any(isinstance(op, Split) for op in self.ops):
            raise ValueError(
                f"env->Observation entry {self.key!r} (target set) cannot use a Split: it routes "
                f"to exactly one channel, not many. Use separate entries gathered by target."
            )
        return self

    @classmethod
    def from_camera(
        cls,
        env_key: str,
        *,
        channel: str | None = None,
        dtype: str = "uint8",
        contiguous: bool = True,
    ) -> LayoutEntry:
        """An env->`Observation` camera entry: read `env_key`, recast, route to a sensor.

        Reads the raw frame the runner published under `env_key`, casts it to `dtype`
        (C-contiguous by default — what a model indexing raw buffers needs), and routes
        it into `obs.sensors[channel]` (the sensor name defaults to `env_key`). The
        long-hand equivalent is a `LayoutEntry` carrying a `DtypeCast` op and a
        `TargetChannel(kind=CAMERA)`.
        """
        name = channel if channel is not None else env_key
        return cls(
            key=f"camera.{name}",
            source=SourceKind.CAMERA,
            source_name=env_key,
            ops=(DtypeCast(dtype=dtype, contiguous=contiguous),),
            target=TargetChannel(kind=ChannelKind.CAMERA, name=name),
        )

    @classmethod
    def from_state(
        cls,
        role: str,
        env_key: str,
        *,
        dim: int | None = None,
        order: int = 0,
        dtype: str = "float32",
    ) -> LayoutEntry:
        """An env->`Observation` state entry: read `env_key`, cast/slice, gather into `role`.

        Reads the raw state vector under `env_key`, casts it to `dtype`, optionally keeps
        the leading `[0:dim]` window, and routes it into the `role` state channel at the
        given gather `order`. Several `from_state(role, …)` entries sharing one `role`
        concatenate (ascending by `order`) into that one state vector — e.g. `eef_pos`(0)
        + `eef_quat`(1) + `gripper_qpos`(2) gather into a 9-D `ee_pose`.
        """
        ops: tuple[LayoutOp, ...] = (DtypeCast(dtype=dtype),)
        if dim is not None:
            ops = (*ops, Slice(start=0, stop=dim))
        return cls(
            key=f"{role}.{env_key}",
            source=SourceKind.STATE,
            source_name=env_key,
            ops=ops,
            target=TargetChannel(kind=ChannelKind.STATE, name=role, order=order),
        )

    @classmethod
    def from_instruction(cls, env_key: str = "instruction") -> LayoutEntry:
        """An env->`Observation` instruction entry: route the env's language string.

        Reads the language string the runner published under `env_key` (default
        ``"instruction"``) into `obs.instruction`. Structural ops do not apply — the
        string is read verbatim.
        """
        return cls(
            key="instruction",
            source=SourceKind.INSTRUCTION,
            source_name=env_key,
            target=TargetChannel(kind=ChannelKind.INSTRUCTION),
        )


class NativeLayout(BaseModel):
    """A checkpoint's native input (or output) dict as an ordered set of entries.

    Each `LayoutEntry` declares one model tensor key (or, via `Split`, a group of
    them), the SDK channel it derives from, and the structural ops that reshape the
    channel value into the wire form. The entry order is preserved so a layout reads
    top to bottom like the `to_obs`/`read_action` callable it replaces.

    This is the spec a `PackToNativeLayout` adapter's `to_spec` advances to (the
    phase boundary), and the target `verify` exercises end to end. It does not apply
    convention math (ADR-0001): the only failure a malformed layout
    can introduce is a structural one — a wrong key, dtype, axis, or window — which
    `verify` running the layout catches, and a value-level disagreement with the
    checkpoint's true metadata, which only the real model under `verify` can reach.
    """

    model_config = ConfigDict(frozen=True)

    entries: tuple[LayoutEntry, ...] = ()

    def keys(self) -> tuple[str, ...]:
        """Every tensor key the layout emits, splits expanded, in declaration order.

        A `Split` entry contributes its components' keys (the group key is not
        itself emitted); every other entry contributes its own key. This is the set
        of keys a packed dict has, so `verify` can assert coverage against it.
        """
        out: list[str] = []
        for entry in self.entries:
            split = next((op for op in entry.ops if isinstance(op, Split)), None)
            if split is not None:
                out.extend(component.key for component in split.components)
            else:
                out.append(entry.key)
        return tuple(out)


__all__ = [
    "Assemble",
    "BatchAxis",
    "ChannelKind",
    "Component",
    "DtypeCast",
    "LayoutEntry",
    "LayoutOp",
    "NativeLayout",
    "Slice",
    "SourceKind",
    "Split",
    "TargetChannel",
]
