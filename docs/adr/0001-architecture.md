# 0001 — Architecture

Status: Accepted

## Context

`manifold-sdk` is the open-source contract of the Manifold platform: what a
user authors their policy against, and what benchmarks are described in, so
the platform can pair the two and run an evaluation. An earlier design used a
single flat `Contract` — a bag of `role → spec` entries matched entry by
entry. It worked, but it conflated three things that change independently (the
robot, the task's sensors, the byte wire) and had no notion of "this is a
Franka," so embodiment-level knowledge and the transforms between action spaces
had nowhere to live. This SDK separates robot, sensors, and wire format into
typed primitives.

## The shape at a glance

A policy and a benchmark meet at a **seam**. The benchmark publishes what it
sees and expects; the policy declares the form it consumes and emits. A
**pipeline** of typed adapters bridges any mismatch — first by reconciling
**conventions** (still SDK `Observation`/`Action`), then by **packing** into the
model's native tensor dict and back.

```
   BENCHMARK                                                              BENCHMARK
 (obs + sensors)                                                       (canonical action)
        │ Observation                                          Action      ▲
        ▼                                                                   │
  ┌────────────────────┐      ┌──────────┐                       ┌────────────────────┐
  │ convention adapters │────▶ │  SEAM    │                       │ convention adapters │
  │ (phase 1, obs)      │      │ channel  │                       │ (phase 1, action)   │
  │ polarity·rebase·    │      │  form    │                       │ polarity·threshold· │
  │ re-encode·resize·   │      │(matching)│                       │ binarize            │
  │ frame-stack         │      └────┬─────┘                       └─────────▲──────────┘
  └────────────────────┘           │ PackToNativeLayout                     │ UnpackFromNativeLayout
                                    ▼ (rename/cast/batch/slice/split/assemble)│ (slice one step from the chunk)
                           ┌──────────────────┐  native dict  ┌───────────────┐
                           │ native tensors   │──────────────▶│ MODEL (forward │
                           │ (NativeLayout)   │               │ pass — out of  │
                           └──────────────────┘               │ reach, class 4)│
                                                              └───────────────┘
  check_compatibility: static, at the channel SEAM   │   verify: dynamic, end-to-end incl. packing
```

| Term | One-line meaning |
|---|---|
| **embodiment** | A robot. Owns one **canonical action space** plus its proprioception. |
| **canonical action space** | The single representation an embodiment is defined in; every other action space converts *into* it. |
| **benchmark** | An embodiment in an environment under a task, *exposing* an observation space and an action space. |
| **channel** | One typed, convention-bearing component of an observation/action: proprioception, a camera, an instruction. |
| **observation space / action space** | The two Gym-style interfaces each side of a pairing exposes. |
| **`PolicySignature`** | The typed `{observation_space, action_space}` contract — the matching seam. Core, torch-free. |
| **`PolicyProfile` → `PolicyEndpoint`** | The loadable identity (weights + native layout) and the loaded, on-device model it produces. |
| **`Session`** | One connection's inference state (chunk queue, optional session identity). |
| **adapter** | A typed, composable transform that bridges one spec to another across the seam. |
| **pipeline** | The ordered pair of adapter chains (one per side) that bridges a pairing. |
| **pairing** | A given policy against a given benchmark — the relationship a pipeline bridges. |
| **`check_compatibility` / `verify`** | Static and dynamic gates over a pairing+pipeline. |
| **`NativeLayout`** | A native dict's wire format — a checkpoint's input/output tensors on the policy end, a simulator's env dict on the benchmark end — declared per profile/runner, applied by a packing adapter. |
| **`TargetChannel` / `UnpackToObservation`** | The env→`Observation` direction of a layout: a per-entry tag (`ChannelKind` + gather `order`) routing each env key into an `Observation` channel, and the adapter that applies it. |

## Decision

### 1. Embodiment-centric, with a canonical action space

The primitive is the **embodiment** — a robot owning one **canonical action
space** plus its proprioception. A **benchmark** is an embodiment placed in an
**environment** under a **task**; it *exposes* an observation space and an action
space (the interface a policy pairs against), and the
embodiment/environment/task decomposition is the structure that produces them.
Policies declare the action space they were trained on and the observation they
consume.

An observation is not flat — it is an aggregate of typed **channels**
(proprioception, one or more cameras, an instruction), each carrying its own
conventions (decision 6). A channel has a real *source* (a wrist camera belongs to
the embodiment, a scene camera to the environment), but bridging depends
only on a channel's conventions, not its source, so provenance stays a deferred,
optional tag rather than a modeled axis.

The canonical action space carries its robotics conventions inline — rotation
format, gripper polarity, delta-vs-absolute — so two parties can never agree on
a value's shape while disagreeing on its meaning, the failure mode that leaves a
working arm scoring 0%. Semantic axes beyond this encoding layer (action /
observation **normalization**, **temporal** rate / horizon) are not
modeled in this first cut; they can be added later as optional spec fields plus
opt-in adapters without a restructure.

The policy layer is **benchmark-free**: benchmark knowledge lives only in the
`Pipeline` and the runner. A checkpoint's fine-tune provenance appears only in
profile/weight names (e.g. `RLDX_FRANKA_LIBERO` encodes the embodiment and the
fine-tune dataset, not a dependency on the benchmark object).

### 2. The seam: signature vs loaded identity

The core models the **matching contract** and nothing that needs to load a
model. `PolicySignature` is the seam: pure Pydantic, torch-free, read by both
sides. The runnable model and its private wire format live one layer out, in
`recipes/` and the policy author's own package.

| Aspect | `PolicySignature` | Loaded identity (`PolicyProfile` → `PolicyEndpoint`, + `NativeLayout`) |
|---|---|---|
| Role | the seam / matching contract | the runnable model + its private wire format |
| Audience | model-agnostic (both sides read it) | model-private (the policy author owns it) |
| Layer | `core/` (pure Pydantic, torch-free) | `recipes/` and the policy author's package (loads weights, holds the device model) |
| Contents | `observation_space`, `action_space` | weights, embodiment_tag, the loaded model; `NativeLayout` (per checkpoint) |
| Used by | `check_compatibility`, `verify` (matching) | `verify` (packing phase), `evaluate` / `serve` |
| Readable without loading the model? | yes | no |
| `NativeLayout` on it? | no — off the matching seam | yes — carried by the packing adapters |

A `PolicyProfile` is the loadable identity — weights, embodiment tag, signature,
and its declared native input/output layouts. It imports torch-free so the
signature can be read and a pipeline checked without loading the model. Loading
it yields a `PolicyEndpoint`: the model resident on the device, exposing
`.signature` and minting `Session`s.

A `Session` is one connection's inference state — the open-loop action-chunk
queue and, for a stateful policy, its session identity:

- `advance(native) -> (raw_chunk, step)` — the primary path when packing is
  used: take the model's native input dict, return the raw output chunk and the
  step index to read from it.
- `infer(observation) -> Action` — the no-packing degenerate path.
- `reset()` — at each episode boundary.
- `close()` — on disconnect.

A session outlives a single episode, so it is not a "rollout"
(which in RL means one episode). `PolicyProfile`, `PolicyEndpoint`, and `Session`
are Protocols; the policy author supplies the concrete classes.

### 3. Compatibility through typed adapters — an iso, not a lens

The seam is bidirectional. On the **observation** side an adapter converts the
benchmark's published form to the form the policy consumes (a pre-processor);
on the **action** side an adapter converts the policy's emitted action to the
embodiment's canonical action space (a post-processor). A policy is compatible
when, on each side, the forms already match or a chain of adapters bridges them.

An adapter is **typed** (`from_spec` / `to_spec`), so a chain is provable from
specs *before* a rollout, not merely runnable. It is a **plain value**, not a
self-registering object: a caller (or a resolver) assembles the adapters it
selects; there is no global registry. One adapter covers a format *family* — it
declares the two spec types it bridges plus an applicability check over their
parameters, composing pure math from `lib/` — so one rotation-format adapter
covers every format pair rather than their cross-product. The adapter protocol —
its typed endpoints and the lookup over them — is the one part of the public
surface that cannot evolve additively, so it is versioned from day one.

**The seam is a chain of bidirectional adapters — an iso, not a lens.** Each
adapter is two opposite-direction maps that are *independent* (the action map
does not take the observation that produced it), which makes the optic an
**isomorphism**, not a lens — and that is the precise reason the binding law is
**round-trip invertibility**, which `verify` enforces, not the lens laws. A
lossless adapter must round-trip (an approximate iso under `verify`'s tolerance);
a *lossy* adapter is non-invertible, which is exactly why it is opt-in.
Composition is ordinary function composition, hence associative, hence
**placement-free for the computation** — `server + adapter ≡ adapter + benchmark`,
so a connector can run co-located with the policy, with the benchmark, or in a
standalone proxy. `check_compatibility` *is* the type-checker for this algebra:
it folds each adapter over the specs and accepts iff the chain reduces to a
matching seam. One caveat the optics literature flags and we adopt: round-tripping
is only quasi-compositional in the presence of an effect, so a **stateful** adapter (the
frame-history buffer) sits outside the clean iso laws — the formal reason it
stays author-inserted, cannot be auto-resolved, and `verify` cannot fully cover
it.

### 4. The pipeline reaches the model: conventions, then packing

The seam does not stop at the policy's channel form — it extends all the way to
the model's native input and back from its raw output. One rule decides where
any step lives:

> output still a valid SDK `Observation`/`Action` ⇒ convention adapter;
> output the model's native dict ⇒ packing adapter (crosses the seam).

| Phase | Adapter kind | Codomain | Examples | Auto-resolvable? |
|---|---|---|---|---|
| 1 convention | convention adapter (spec-endomorphic) | a valid SDK `Observation`/`Action` | polarity, rebase, rotation re-encode, resize, frame-stack, gripper threshold, binarize | lossless: yes; lossy: opt-in |
| 2 packing | packing adapter (crosses the seam) | the model's native tensor dict (`NativeLayout`) | `PackToNativeLayout` / `UnpackFromNativeLayout` | no — author-inserted |

The packing residue is a property of the **model's wire convention, not the
benchmark** — two checkpoints on the *same* benchmark can demand entirely
different native tensor formats — so it is declared per profile, never on the
benchmark. It is a small **reusable algebra**, not bespoke per checkpoint:

- **rename** — map an SDK channel to a model key (an entry with no ops).
- **cast** — `DtypeCast` to a target dtype, optionally contiguous.
- **batch/time** — `BatchAxis` prepends a batch and/or time axis.
- **slice** — `Slice` keeps a `[start:stop)` window of a source vector.
- **split** — `Split` fans one vector into several named windowed keys.
- **assemble** — `Assemble` concatenates named windows in a fixed order.

A profile declares a `NativeLayout` (each model key, the SDK channel it derives
from, shape, dtype, batch/time prefix, component order); a single generic
`PackToNativeLayout` (with symmetric `UnpackFromNativeLayout`) *applies* it. The
layout lives in exactly one place, so the generic adapter cannot drift from it.
It is a **downstream, profile-local target** consumed only by the packing adapter
and `verify`, and is **not** placed on `PolicySignature` — matching
stays at the coarse channel seam, and `verify` owns the packing phase end to
end. A declaration can still disagree with the checkpoint's true metadata; that
residual is irreducible, and `verify` running the real model is the backstop.

Because the codomain rule pulls *all* convention math up into adapters, two
adapter gaps are part of the contract: gripper thresholding generalizes beyond
`EEActionSpace` to the whole-body `UnifiedActionSpace`, and a `DiscreteBinarize`
adapter covers any discrete action dim (e.g. a control-mode head). Without these,
a whole-body checkpoint's discretization would be stranded in the packing layer —
the symptom that the convention layer is missing a slot, not that packing is
irreducible.

**The seam is symmetric — one `NativeLayout`, both ends.** The same structural
bridge runs at the benchmark end. A simulator hands the runner a raw native env
dict (nested keys, or a flat `robot0_*` bag), which must become a channel-typed
`Observation` — the mirror of packing an `Observation` into a model's tensor
dict. This does not need a new concept: it is the existing layout's *gather*, applied
the other direction. `UnpackToObservation` reads each entry's env key, runs the
same `rename`/`cast`/`slice`/… algebra, and routes the result into an
`Observation` channel through a per-entry `TargetChannel` (a `ChannelKind` plus a
gather `order`) — so several env keys (`eef_pos`, `eef_quat`, `gripper_qpos`)
gather into one `ee_pose` state vector exactly as `unpack` gathers a model's
split tensors into one `Action`.

| Crossing | Direction | Structural move | Adapter |
|---|---|---|---|
| policy `pack` | `Observation` → model dict | scatter (one channel → many keys) | `PackToNativeLayout` |
| policy `unpack` | model output → `Action` | gather (many keys → one vector) | `UnpackFromNativeLayout` |
| benchmark obs | env dict → `Observation` | gather, routed per channel | `UnpackToObservation` |
| benchmark action | `Action` → env action | identity (the env consumes the canonical action) | identity |

```
              BENCHMARK end                 SEAM (channel form)             POLICY end
  env dict ──unpack──▶ ┌─────────────┐                            ┌─────────────┐ ──pack──▶ model
 (raw native)  layout  │ conventions │──▶ Observation / Action ◀──│ conventions │  layout   tensors
               gather  └─────────────┘  (cameras·proprio·text)    └─────────────┘  scatter
  env action ◀─identity────────────────────────────────────────────────── unpack ◀── model out
       ▲                                                                                ▲
  sim-API extraction ── irreducible endpoint ──                            the model forward
```

So the convention/packing split holds uniformly on both ends: every *convention*
difference is an adapter, every *structural* crossing is a declared layout, and
the only imperative code sits at the two irreducible endpoints — the simulator's
API and the model's forward pass. A benchmark's env layout is profile-local
presentation, declared in the runner (it refers to sim-specific env keys), mirroring
how a policy's `NativeLayout` lives in its profile — off `PolicySignature` and out
of core. A benchmark publishes a *superset* of channels and each policy consumes a
subset; supporting one benchmark across multiple embodiments is designed for
(presentation is owned per benchmark, not by the robot) but deferred until a
benchmark needs it.

### 5. A pipeline, validated then verified, executed two ways

A **pipeline** is an ordered pair of adapter chains (observation side, action
side) that bridges a **pairing**. The core exposes the building blocks and
does not decide strategy:

- `check_compatibility(policy, benchmark, pipeline=None)` judges, *statically
  from specs*, whether the pipeline carries each side's form to the matching
  seam, the policy's required sensors are published, and an instruction is
  available if needed. It does not search, and it does not hold adapters of its own.
- `verify(policy, benchmark, pipeline=None)` confirms *dynamically*, by running
  synthetic example data through the pipeline — now including the packing phase —
  and checking the result against the benchmark's specs. It catches an adapter
  whose data map is wrong, which a static check cannot.
- `pipeline.apply_observation` / `pipeline.apply_action` fold the same chains
  over *live* data: the production counterpart of what `verify` exercises on
  synthetic data. Pure functions over plain values, public for the escape-hatch
  reason below.

A pairing is *executed* one of two ways with no new mechanism. `evaluate(endpoint,
benchmark, …)` runs it **in-process, no server** (the researcher's local loop,
and the zero-hop degenerate of the connector algebra). `serve(endpoint)` hosts it
over the bridge for **remote and sharded** evaluation. The server is one
transport for the core, not a requirement; both modes run the same gate
(`check_compatibility` + `verify`) policy-side before the first step.

**Resolution** — *finding* a pipeline by searching a pool of adapters — is
opinionated (which adapter, how to order, prefer lossless) and lives in the
swappable `recipes/` layer or a consumer's own higher layer. A resolver searches
only the pool it is given; there is no global registry. It walks the channel-spec
graph and so cannot *discover* stateful, `tap`, or packing adapters — they do not
advance that graph — so those are author-inserted, never resolved.

**A pipeline is a value, not (yet) an artifact.** It is composed in memory from
adapter instances, not a serialized config. A serialized *pipeline artifact*
(fetched from a catalog by id, emitted by an editor or LLM, shipped over the
wire) is a deliberate non-goal of this cut: rehydrating one needs a name→adapter
lookup, which is exactly the versioned adapter lookup above, and the
contract-closing loop pairs a policy with its *home* benchmark — a clean match
whose pipeline is empty. Serialization can be added later; nothing here commits
to it.

**The catalog is a convenience layer over open primitives.** Callers can bypass
it and assemble pipelines directly from the public primitives, or author new
adapters against the versioned protocol. Bridging degrades gracefully: use a
ready-made pipeline if one fits; let a `recipes/` resolver compose one if it
can; otherwise hand-assemble a `Pipeline` from catalog adapters and `check` /
`verify` / `apply` it; failing that, author a new adapter against the versioned
protocol. The last two paths are why `apply` is public: a hand-built pipeline a
caller cannot run is not an escape hatch. New pipelines are added as code, by
PR, not as runtime data.

**Adaptation runs on the policy side, by the policy author — though the SDK keeps
it location-free.** Manifold publishes the benchmarks; they are the fixed,
authoritative targets, and the policy author's job is to make their policy
*conform*. So the flow inverts the usual intuition: the benchmark advertises its
contract; the policy side runs `check_compatibility` + `verify`, builds the
bridging pipeline, and `apply`s it before the first observation. Because these
are pure functions over plain values they run equally on a benchmark runner or a
standalone proxy; the intended product workflow runs adaptation on the policy
side, and introduces no "proxy" kind: a proxy is a deployment of these
primitives, not a new thing in the model.

**One check, three gates.** The same primitives serve a **dev-time** check (a
policy author learns which benchmarks fit), an **upload-time** check (the backend
matchmaker enumerates compatible benchmarks), and a **runtime handshake** (a
quick smoke test when the benchmark connects) — with no new mechanism.

### 6. Conventions declared *and* verified

Declaring a convention and statically checking it is not enough: a contract can
be well-typed yet **wrong**, passing `check_compatibility` and then failing
silently at ~0%. Such a silent zero falls into one of four classes:

| Class | What goes wrong | Example | Caught by |
|---|---|---|---|
| 1 unmodeled axis | convention exists but has no spec field | a camera needing a 180° flip with no way to declare it | nothing until modeled |
| 2 declaration vs data | our pipeline ships data violating a correct declaration | wxyz quaternion under declared scalar-last QUATERNION | `verify` (asymmetric probe) |
| 3 declaration vs reality | both sides agree on a declaration that's the opposite of true | inverted gripper sign | observability / active probe |
| 4 outside the seam | genuinely model-internal behaviour | normalization, the forward pass, a checkpoint-load patch | not reachable — a non-goal |

Class 4 is named because adapters and `verify` do not reach
*inside* the model. Native packing, by contrast, is now a pipeline adapter, so
`verify` *does* cover it — the not-checked region shrinks to what truly runs
inside the network.

Against these there are five rungs, each stating what it proves:

| Rung | Mechanism | Proves | When | Built? |
|---|---|---|---|---|
| 1 | `check_compatibility` | declarations mutually consistent | static | yes |
| 2 | `verify` | the declared bridge correct, within coverage | offline | yes |
| 3 | introspection | a declaration matches a checkpoint's metadata | offline | yes, for lerobot checkpoints (`recipes/lerobot.py`) |
| 4 | observability (tap) | a human can SEE the real data + behaviour | empirical | yes |
| 5 | active behavioural probe | the running pair behaves as declared | live | needs scripted env |

The rungs run from static and certain to live and best-effort. A green check at
one rung proves only what that rung states.

**Camera channels carry typed convention specs.** Proprioception carries its conventions
inline, but a camera reduced to a name-only string does not — and that asymmetry
*is* the unmodeled-axis class. So every channel a policy consumes is a typed,
convention-bearing spec; `check` / `verify` / `resolve` and adapters operate over
channels uniformly. Cameras gain small typed convention enums:

| Camera convention | Mismatch is silent? | Lossless / lossy bridge? | Bridged by |
|---|---|---|---|
| orientation | yes — silent and plausible | lossless, else `INCOMPATIBLE` | lossless flip/rotate adapter |
| channel-order | yes — silent and plausible | lossless, else `INCOMPATIBLE` | lossless reorder adapter |
| resolution | loud or absorbed | lossy (flags as lossy, never hard-incompatible) | resize adapter |

Only the target *shape* is declared; crop, aspect, and interpolation are
resize-adapter parameters, not axes — the same no-speculative-modeling principle
that keeps the action space monolithic. Declaring resolution also lets a resolver
insert the resize.

**`verify` uses asymmetric probe data and reports coverage, not a pass/fail verdict.** Symmetric zeros cannot reveal a polarity
flip or an element reorder, so `verify` pushes *asymmetric* probe data — a known
non-identity rotation, an unambiguous "open" gripper, a non-origin pose — and
asserts the semantic invariant survives the round-trip. Comparison must be
representation-aware or it flags differences that are not there: quaternion
double-cover, axis-angle 2π wraparound, Euler gimbal lock, and 6D
re-orthonormalization all break raw float equality, so rotations are compared by
*geodesic angle within tolerance*. And `verify` returns a **coverage manifest**,
not a verdict — *checked and passed*, *checked and failed*, *not checked / out of
reach* — so a clean result means "the bridge was exercised," never "the pairing
works."

### 7. State, observability, and the escape hatch

**Adapter-first authoring; the adapter is the one extension point.** All
adjustment of wire data happens by composing adapters into the pipeline, never by
hand-editing a server's inference: shared catalog adapters for common
conversions, a *custom* adapter for bespoke glue, and *graduation* of a recurring
custom adapter into the catalog. The escape hatch is "write an adapter," so even
bespoke handling stays legible, composable, and visible to `verify`. An adapter
acts on the data flowing through the pipeline — up to and including the model's
native tensors; anything purely *inside* the policy network is provisioning, the
fourth failure class, which no adapter reaches.

**Outside the seam, the runner is a dumb native driver.** The benchmark runner
sits *outside* the SDK seam and never adapts wire data on the policy's behalf: it
ships the env's *true native form*. Convention math — quat→axis-angle, frame
rebase, gripper polarity — lives in the policy-side pairing pipeline as ordinary
phase-one adapters (`ProprioRotationAdapter`, with a non-wrapping mode for the
[0, π] distribution shift scipy's `as_rotvec` would otherwise introduce;
`DynamicFrameRebaseAdapter`; the observation-side gripper adapters), never in
the runner. What stays in the runner is only the env→`Observation` *structural
gather*, and that is the declared env `NativeLayout` applied by
`UnpackToObservation` (decision 4), not hand-rolled. The one irreducible residue
is the **sim-API extraction** — reading the simulator's raw output into the flat
dict the layout gathers: nested-key traversal, dropping a vector env's `n_envs`
batch axis, a present-key fallback. Like the model forward, no declaration can
reach it.

**Object-model envs keep their extraction in float64 — the lone exception.**
SIMPLER's WidowX pose is not on the obs dict; the runner fabricates it off the
unwrapped env, composing `inv(base_pose) @ ee_pose` and reordering `mat2quat`'s
scalar-first quaternion. Hoisting that rebase into a policy-side
`DynamicFrameRebaseAdapter` is *not* byte-equivalent: the rebase is computed in
float64 at extraction, whereas the seam would quantize the world-frame pose to
float32 first, and float32 matrix arithmetic is not associative — the requantized
product drifts enough to push the proprio off distribution. So for object-model
envs the frame rebase stays in float64 extraction, the sim-API endpoint the
symmetry already permits. SIMPLER still assembles its `Observation` through the
same `UnpackToObservation` layout as the dict-envs — only its *extraction* is
heavier, fabricating the pose in float64 rather than reading it off the obs dict.
Every runner therefore shares one assembly path; the object-model env differs
only in the size of its irreducible endpoint. A byte-equivalence test against a
float64 reference is what surfaced this float32 ordering.

**Pipelines may carry episodic state — beside the frozen value, never on it.**
Some adaptation is stateful and per-episode: a frame-history buffer (single frame
→ stacked clip) is a genuine data-shape transform, hence a stateful adapter,
whereas the open-loop action-chunk *horizon* is serve-loop control flow and stays
out of the pipeline. State does not live on the pipeline value, which is frozen
config; a separate state object is threaded into the apply operations — the one
runtime concept that bumps the adapter protocol version. It is keyed per
`(connection, lane)` and reset per-lane, not globally — forced by vectorized
environments (many lanes stepped in lockstep, finishing asynchronously) and by
parallel benchmark workers. It therefore lives with the *pairing*, never the
shared policy server: buffering frames on a shared server would interleave
concurrent workers' histories and corrupt them silently — the silent-failure
class at the infrastructure level — which makes "state with the pairing"
mandatory. Adapters merely *signal* the state key they touch — a label, not a
permission model; collisions are intentional sharing the author owns.

**Observability adapters are ordinary pipeline adapters: a `tap` inserts inline
and records field-level summaries without changing the data.** When the contract
leaves a gap, the author must see the real data at the seam — so the SDK
provides a `tap` adapter (after the shell idiom): a lossless identity on spec
*and* data whose only effect is to observe (record per-field shape, dtype,
range; dump frames; log the stream). Because it is an adapter it
composes: drop one before and after a transform to see its effect, or into a
misbehaving pairing to see reality at that seam. A `describe` inspector and stream
record/replay (a `Recorder`) round it out, in `recipes/inspect.py`. This gives the
*passive* half of behavioural probing — a `tap` on each side logs the
command-and-response stream, so an inverted gripper is read directly off the
finger positions rather than guessed from a score. The *active* half (a
controlled known-signal episode with an automated oracle) needs the env scripted.
Static diagnostics stay minimal.

**Introspection is a head start, not a verifier.** Rung 3 is a `recipes/`-level,
format-specific helper that reads a checkpoint's own metadata and *suggests*
signature fields for the author to confirm; `recipes/lerobot.py` does this for
lerobot checkpoints. It only ever sees what the format records (most checkpoints
encode neither frame convention nor gripper polarity), so it is
populate-and-confirm, explicitly not a guarantee, and it is format glue, not
core. `recipes/inspect.py` is observability (`describe` / `Recorder`), a
different rung.

### 8. Serving is concurrent, sharded, and placement-free

One loaded `PolicyEndpoint` serves many runner shards: a thread per connection,
each with its own `Session` and `PipelineState`, and an endpoint-owned lock around
the shared model forward (the speedup is from overlapping the shards' simulators,
not a parallel forward).

| | Stateless policy | Stateful policy |
|---|---|---|
| Example | Cosmos (`reset` is a no-op) | RLDX (per-episode recurrent memory) |
| Sharing the loaded model | trivial | not free — must isolate per-session memory |
| Per-`Session` requirement | none | thread a `session_id` into the policy's own session API; signal reset at each episode boundary |
| Teardown guardrail | n/a | `Session.close()` retires the per-session state; ids are never reused |

The stateful contract is the same one NVIDIA Triton's sequence batcher documents
(a correlation id per sequence, with start/end flags). `close()` closes a
silent-failure path at the infrastructure level: a dropped shard must not leak a
stale memory slot. A policy whose episode state is managed *outside* the SDK's
session lifecycle (inside an opaque inference library the SDK cannot key)
requires a per-session model replica to fan out, and the endpoint design must
account for that when adopting such a policy.

Placement is a transport choice — for the computation, not the lifecycle. The
in-process fold is the default (zero hops); a `Transport` seam marks where a
proxy, shared-memory, or iroh transport plugs in without touching the
choreography or the pipeline. The qualifier matters: the *transform* is
placement-free, but a stateful component's *lifecycle* (its `PipelineState` lane,
a policy session slot) is pinned to wherever it runs — a live rollout's state
cannot be migrated by re-associating the algebra.

**Async is deferred, and the seam it needs is protected.** Everything
shipped is synchronous lockstep — one observation in, one action out (the chunk
queue already decouples 1 inference : K actions). Real deployments are async:
observations arrive on the sensor clock and actions are consumed on the control clock at
different rates, and real-time chunking (RTC) blends a freshly-inferred chunk
against the still-executing one. So the observation and action sides stay
**separable flows** (their own state, queues, clocks); the inference boundary
stays **liftable to a batched form** without changing per-session keying; and —
since the protocol has no external consumers yet — the bridge frames already
**reserve** the fields async will need: a per-frame `lane` id (vectorized /
multi-lane within one connection) and an `action_prefix` + monotonic `timestep`
on the observation/action frames (RTC inpainting). Reserving them now means no
protocol break later: single-lane lockstep needs none; vectorized lanes need the
lane id; RTC needs the prefix + timestep.

### 9. Layering

| Layer | Contents | Imports `recipes/`? |
|---|---|---|
| `core/` | kinds: `PolicySignature`, `Pipeline`, `NativeLayout`, `check_compatibility`, `verify` (pure, torch-free) | never |
| `lib/` | pure helper math (rotation conversions, etc.) | never |
| `wire/` | the msgpack codec and the public bridge protocol (data-plane serialization) | never |
| catalog directories | one concrete embodiment / sensor / benchmark / policy / adapter per file | no |
| `recipes/` | opinionated, swappable strategies: resolvers, the serving harness, `inspect`, `Session`/`PolicyEndpoint`, `evaluate` / `serve` | n/a |

The core never imports `recipes/`. The serving harness — the loop that runs the
bridge choreography — is a reference `recipes/` artifact, backend-agnostic (it
takes an inference callable), not core; the incremental path is to ship the recipe
and promote it to a blessed base class only if the shape proves stable.

### 10. Type modeling — one mechanism per role

Several class-modeling mechanisms are in use; each maps to a role, so the choice
is not per-author taste. The wire boundary is hand-coded (msgpack over dicts), so
a pydantic model here means an *authored spec*, not *serialization*.

| The type is… | Use |
|---|---|
| Authored spec / catalog data — validated, value-compared, part of the pairing (embodiments, benchmarks, `PolicySignature`, the native-layout ops) | frozen `BaseModel` |
| A discriminated union of specs | frozen `BaseModel` variants + `Field(discriminator=…)`, tag a `Literal` |
| Internal value / result / record — no validation, hot-path or a small returned record (`Observation`, `Action`, `Pipeline`, the reports, `Pairing`) | `@dataclass(frozen=True)` (`eq=False` when it holds a numpy array); cross-field invariants in `__post_init__` |
| Stateful accumulator — mutates over an episode or run (`PipelineState`, `Recorder`) | mutable `@dataclass` |
| Wire frame / payload | hand-coded dict codec in `wire/` — untrusted input validated structurally, no pydantic on the wire |
| Behavior contract the SDK owns and subclasses in-tree (the adapter hierarchy) | `ABC` |
| Structural seam an external or alternate implementation plugs into (transport, endpoint, profile) | `Protocol` — `@runtime_checkable` only when isinstance-checked |
| Closed choice set referenced across modules and carried on the wire | `StrEnum` |
| Closed set used only as a union discriminator or a single local alias | `Literal` |

Result and record objects follow the frozen-dataclass row uniformly — the invariant
lives in `__post_init__`, not a pydantic validator, and a `NamedTuple` is not used
for a domain record (it leaks tuple indexing and iteration).

## Alternatives considered

- **Keep the flat `role → spec` Contract.** Rejected: no home for
  embodiment-level reuse or action-space transforms, and it conflates robot,
  sensors, and wire. Separating those is the point of the design.
- **Make "unified action space" the canonical.** Rejected — and the term is kept
  distinct: a *unified action space* is a fixed-width padded buffer a
  multi-embodiment base model emits (pi0.5 = 32, RDT-1B = 128). It is a model-side
  superset, not a robot's true action space; mapping it down to the canonical is
  one adapter, not the canonical itself.
- **Bake a matchmaker / auto-resolver into the core.** Rejected: pairing strategy
  is opinionated and context-dependent (a CLI resolves interactively; a backend
  applies a fixed policy). Forcing one strategy into the public contract makes the SDK harder to
  embed. Ship strategies as reference `recipes/` instead.
- **Decompose the action space into per-field sub-units (a compositional spec).**
  Representation adapters are naturally per-field and orthogonal, but rejected for
  now: optional spec fields evolve additively, so the
  monolithic spec does not lock in anything the compositional model would avoid, and the
  genuinely breaking surface is the adapter protocol — which is versioned
  regardless.
- **Declare the model's native input dict as a schema on `PolicySignature` (a
  checked contract).** Rejected: it puts wire-level tensor layout and dtype onto
  the matching contract, reopening the seam the design keeps clean and making a
  versioned public surface out of checkpoint-private metadata. The native layout
  is instead a downstream, profile-local `NativeLayout` target applied by a
  packing adapter and exercised by `verify` — declared, but off the matching seam.
- **Serialize the pipeline into a data artifact now (a marketplace / visual / LLM
  authoring channel).** Rejected for this cut. It makes the pipeline a noun
  rehydrated through a name→adapter lookup — reopening the one versioned,
  non-additive surface — to serve authoring channels the contract-closing loop (a
  clean-match home pairing) does not need. Adapters and pipelines are code;
  serialization stays possible later without pre-committing the lookup.

## Consequences

- Compatibility is verifiable before any episode runs — mismatches surface at
  pairing time, not as malformed values mid-rollout.
- The observation side is part of the contract, not just the action side. A
  policy declares the proprioception form it consumes, so `check_compatibility`
  catches an observation-form mismatch and an adapter bridges it.
- The pipeline spans the whole seam, benchmark to model and back. Convention
  bridging and native packing are both adapters. The packing residue reduces to a
  reusable structural algebra applied by one generic adapter, so `verify` covers
  it and the not-checked region shrinks to genuinely model-internal behaviour. The
  loaded identity therefore declares its native layout on the profile (consumed by
  the packing adapter and `verify`) rather than carrying bespoke obs/action
  packing callables.
- The seam is symmetric: the same `NativeLayout` algebra bridges both ends — a
  model's tensor dict on the policy end (`pack`/`unpack`) and a simulator's env
  dict on the benchmark end (`UnpackToObservation`). The runner ships native form
  and declares its env layout; only the sim-API extraction (and, for object-model
  envs, a float64 frame rebase) stays imperative.
- Third-party packages can add adapters without changes to the core.
- Consumers must choose (or adopt a recipe for) a pairing strategy — the SDK will
  not silently pick one. That is the intended cost of keeping the core embeddable.
- A pairing runs in-process via `evaluate` or over the bridge via `serve`, on the
  same gate and the same pipeline; the server is a transport, not a requirement.
- Serving is concurrent and sharded off one loaded endpoint; stateful policies
  thread a per-session id with a `close`-on-teardown guardrail, and a policy whose
  state the SDK cannot key needs a per-session replica to fan out.
- The adapter protocol is versioned from the start. A future shift to a per-field
  resolver is the one change that would break third-party adapters, so consumers
  can pin the protocol version.
- A resolver (in `recipes/`) searches a pool of adapter instances the caller has
  already aimed at the benchmark's parameters; it composes multi-hop chains and
  terminates by de-duplicating specs by value. Stateful, `tap`, and packing
  adapters are author-inserted, not resolved.
- Retargeting needs an embodiment's kinematic model; embodiments may later grow a
  kinematic spec to enable it. Out of scope until a benchmark needs it.
- **Deferred, with seams reserved.** Normalization is the open declarable axis:
  primarily policy-internal (a wrong-stats wrapper fails silently while the
  contract reads COMPATIBLE), so it is an author-owned job caught at most by
  introspection; the genuine seam case (a benchmark publishing pre-normalized
  data) is reserved as an optional spec field plus opt-in adapter, and when built
  must carry a stats *identity*, not merely a scheme enum, since
  matching-scheme/differing-stats is precisely the silent failure. Async serving
  is deferred with its wire fields reserved. Neither is a restructure when
  added.
