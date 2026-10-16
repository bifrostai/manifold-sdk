# 0010 - A whole-body action is a composite of body-part groups

Status: Pending

## Context

A mobile manipulator acts with more than its arms. Besides each arm's position,
rotation and gripper, its action can carry base motion, torso joints and a
controller's mode value. It also reports state that no arm field holds, such as the
base's pose and the torso's joints.

A `UnifiedActionSpace` with the arm as its payload leaves those other values
unlabelled. ADR 0001 keeps the unified action space for a model-side padded
buffer, not a robot's action.

The base values cannot be an optional field of `EEActionSpace`. ADR 0001 grows a
spec by optional fields, and a peer without a field drops it. That is tolerable for
a field that declares a value of its own, under its own key, but not for one that
adds slots inside an existing value: the peer then misreads the value's layout,
and every end-effector adapter, which rebuilds an action arm by arm, breaks on it.

## Decision

A whole-body action is a `CompositeActionSpace`: groups back to back, in the order
that the robot reads them in, then an optional mode slot.

- An arm group is an unchanged `EEActionSpace` or `JointActionSpace`.
- `BodyActionSpace` covers the rest: a torso or a mobile base by `part`, saying
  whether its values are an absolute target, a delta or a velocity. A base also
  names the frame that it moves in.
- `mode_slot` declares where the mode value cuts: above 0 (`binary_0`) or
  at 0.5 and above (`binary_0.5`). The slot comes after the last group.

Each group is a whole spec with its conventions inline, split by body part, so
this differs from the per-field compositional spec that ADR 0001 rejected. An
existing stateless adapter that keeps a group's kind applies to a single group
through `GroupAdapter`, which takes `lossless` from the adapter that it wraps. The
resolver bridges two whole-body actions group by group, searching each differing
group on its own.

Proprioception gains `base_pose` (a position then a rotation, in a declared frame)
and `torso_pos` (joints, no gripper). Each is a new value under its own state key,
so each is additive.

## Consequences

An older SDK refuses a composite action and silently drops `base_pose` and
`torso_pos`, so it can accept a pairing that this SDK refuses.

A committed spec gains `"base_pose": null` and `"torso_pos": null` when its
benchmark moves to this SDK.

`verify` checks a composite group by group, including closed grippers and both
sides of the mode cut. When a chain rebases the ee_pose into the base frame by the
declared base pose, it compares the ee_pose against the probe seen from that pose.

An embodiment that declares a composite is no longer described by ADR 0001's
Decision 4, which calls the unified action whole-body. An adapter that produces a
unified action, such as one that widens an arm-only action or snaps a mode value,
does not reach a composite until it has a composite counterpart.
