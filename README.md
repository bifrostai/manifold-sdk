# Manifold SDK

Manifold is the platform for accelerating robotics research.

It runs your sim evaluations at scale, producing detailed reports on performance and
failure modes. Simply bring your policy and Manifold handles the rest for you.

Get started with our [documentation](https://docs.manifold.bifrost.ai/quickstart).

## The SDK

Manifold SDK provides a language to declare policy inputs and outputs, check compatibility
across policies and embodiments, and bridge mismatches in formats.

It's a MPL-2.0 licensed Python package for running policies against benchmarks and evaluations.
It's built for agents and ships with a set of well-tested
[skills](https://github.com/bifrostai/manifold-skills), which gives agents accurate context
and best practices out of the box.

It achieves this by having:
- Comprehensive types for action spaces, sensors, and embodiments
- Compatibility checks that provide fast verification
- Typed adapters that bridge mismatches (rotation format, gripper polarity, retargeting)

This helps you or your agent connect policies quickly and safely, reducing the risk
of frustrating bugs that silently influence experiment results.

## How it works

Manifold enforces a clean separation of model code from evaluation scaffolding.

Using the types provided, policies now explicitly declare a `Signature` (what the model's
intended inputs and outputs are), preventing casts, conversions, and transforms from
being littered alongside your model inference code.

Errors are also caught early by:
- `check_compatibility`, which proves the declared policy and benchmark specs match,
- `verify` which tests the pipeline with dummy data to catch conversion errors immediately.

```python
from manifold.core import (
    Camera, EEActionSpace, GripperFormat, PolicySignature, RotationFormat, check_compatibility,
)
from manifold.benchmarks.libero import LIBERO

policy = PolicySignature(
    action_space=EEActionSpace(
        rotation=RotationFormat.AXIS_ANGLE,
        gripper=GripperFormat.SIGNED,
        delta=True,
    ),
    cameras=[Camera(name="agentview", shape=(256, 256, 3))],
)

report = check_compatibility(policy, LIBERO)
```

## Mismatches

If you are trying to evaluate your policy against a benchmark with a different action space,
you can bridge the two with a pipeline, which allows the policy to continue being evaluated
while ensuring both sides still truthfully advertise their capabilities for clarity.

In this example, the policy natively outputs gripper values in [-1, +1] where -1 is **open**.
If the benchmark requires [-1, +1] where -1 is **closed**, you would only discover the mismatch
when reviewing the results. Instead, with the Manifold SDK, we avoid this problem by making the
conversion explicit and easy to identify:

```python
from manifold.core import Pipeline, verify
from manifold.adapters.action import GripperPolarityAdapter

low_gripper = PolicySignature(
    action_space=EEActionSpace(
        rotation=RotationFormat.AXIS_ANGLE,
        gripper=GripperFormat.SIGNED_OPEN_LOW,
        delta=True,
    ),
    cameras=[Camera(name="agentview", shape=(256, 256, 3))],
)
pipeline = Pipeline(action=[GripperPolarityAdapter(target=GripperFormat.SIGNED)])

report = check_compatibility(low_gripper, LIBERO, pipeline=pipeline)
report = verify(low_gripper, LIBERO, pipeline)
```

## Extensibility

To add a new robot or control mode, create a file in `embodiments/` that
defines an `Embodiment` with its action space and proprioception.

To support a new action-space conversion (e.g. a rotation format your policy
uses), write an `ActionAdapter` in `adapters/action/`. For observation-side
conversions, write an `ObservationAdapter` instead. Then add your adapter to a
`Pipeline` and pass it to `check_compatibility` the same way the gripper
example above does.
