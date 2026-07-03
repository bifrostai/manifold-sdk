# Manifold SDK

Pair a robot **policy** with a **benchmark**: the types both are described in,
the compatibility check between them, and the adapters that bridge them.

`manifold-sdk` is the public contract of the Manifold platform. You author your
policy against these types; the platform runs it against benchmarks described in
the same terms.

> Status: early. Licensed MPL-2.0. The reasoning behind the shape below is in
> [`docs/adr/0001-architecture.md`](docs/adr/0001-architecture.md).

## Concepts

- **Embodiment** — the robot itself: its **canonical action space** (the one
  action representation it is defined in) plus its proprioception (what it
  reports about its own state). Reusable across benchmarks; e.g. a Franka arm.
- **Sensor** — an exteroceptive input the policy consumes (a camera). Owned by
  the *benchmark*, not the robot — the same arm under a different task exposes
  different sensors.
- **Benchmark** — an embodiment placed in a task, plus the sensors and task
  signals (instruction, success) it publishes. Defines what your policy
  receives and the canonical action space it must drive.
- **Policy signature** — what you declare about your policy: the action space it
  was trained to emit, and the proprioception and sensors it consumes.
- **Adapter** — a typed transform on one side of the seam. An *action adapter*
  converts the policy's action to the benchmark's form; an *observation adapter*
  converts the benchmark's observation to the policy's. Each is a lossless
  representation conversion (rotation format, gripper polarity, frame, …). One
  adapter spans a format family — it declares the spec types it bridges and
  composes the pure math in `lib/` — rather than one per pair.
- **Pipeline** — the ordered observation and action adapter chains that bridge a
  given policy-and-benchmark pairing.

A policy is compatible with a benchmark when each side's form already matches —
or a supplied pipeline of adapters bridges it — and it consumes only sensors the
benchmark publishes. `check_compatibility()` proves this *before* any rollout.

> The contract models the **encoding** of an action — shape, rotation format,
> gripper polarity, frame, delta-vs-absolute. It does not model **normalization**
> (the per-dataset statistics a policy was trained against) or **temporal** rate
> and horizon. Both cause silent failures; adding them later as optional spec
> fields and opt-in adapters would not break an existing pairing.
> `check_compatibility()` proves your declared semantics agree — not that a
> policy is in-distribution, which only a rollout can show.

## Layout

```
manifold/
├── core/         the pairing types and the checks over them: embodiment,
│                 sensor, benchmark, policy, adapter, pipeline, conventions,
│                 check_compatibility, verify
├── lib/          pure math adapters compose from (gripper, rotation) — no
│                 domain types
├── embodiments/  catalog: one robot-and-control-mode per file
├── sensors/      catalog: camera constructors keyed on name and mount
├── benchmarks/   catalog: one task suite per file, importing its embodiment
├── adapters/     one transform per file, grouped by the side it acts on
└── recipes/      swappable pairing strategies: resolver, serving harness,
                  inspect — built on the core, not part of it
```

The **core is mechanism, not strategy**: it never chooses an adapter or decides
how to pair. That lives in `recipes/` (reference implementations you can use or
replace) or in your own higher layer.

## Pairing a policy with a benchmark

Declare your policy with the core types and check it against a catalog benchmark.
The catalog (`embodiments/`, `sensors/`, `benchmarks/`, `adapters/`) and the
serving helpers (`recipes.serve`, `recipes.evaluate`) are part of the package.

```python
from manifold.core import (
    Camera, EEActionSpace, GripperFormat, PolicySignature, RotationFormat, check_compatibility,
)
from manifold.benchmarks.libero import LIBERO

policy = PolicySignature(
    action_space=EEActionSpace(
        rotation=RotationFormat.AXIS_ANGLE,
        gripper=GripperFormat.SIGNED_OPEN_LOW,
        delta=True,
    ),
    cameras=[Camera(name="agentview", shape=(256, 256, 3))],
)

report = check_compatibility(policy, LIBERO)   # compatible
```

If your policy's action space differs from the benchmark's, supply a pipeline of
transforms that bridges it — here, a policy whose gripper opens on the high
value while LIBERO's opens on the low value:

```python
from manifold.core import Pipeline, verify
from manifold.adapters.action import GripperPolarityAdapter

high_gripper = PolicySignature(
    action_space=EEActionSpace(
        rotation=RotationFormat.AXIS_ANGLE,
        gripper=GripperFormat.SIGNED,
        delta=True,
    ),
    cameras=[Camera(name="agentview", shape=(256, 256, 3))],
)
pipeline = Pipeline(action=[GripperPolarityAdapter(target=GripperFormat.SIGNED_OPEN_LOW)])

report = check_compatibility(high_gripper, LIBERO, pipeline=pipeline)   # compatible_via_pipeline
report = verify(high_gripper, LIBERO, pipeline)                         # every check passes
```

`check_compatibility` proves the declared specs agree; `verify` also runs
synthetic data through the pipeline, so a transform whose math is wrong fails even
when its shapes match.

## Extending the catalog

- A new robot, or a robot in a new control mode → a file in `embodiments/`
  binding one `Embodiment` constant.
- A new task suite → a file in `benchmarks/` that imports its embodiment and
  builds its cameras from the `sensors/` constructors.
- A new action-space conversion → a typed `ActionAdapter` in `adapters/action/…`
  (or an `ObservationAdapter` for the observation side). Adapters are plain
  values: a caller collects the ones it selects into a `Pipeline` and passes it to
  `check_compatibility` (and to a resolver in `recipes/`). Nothing registers itself.
