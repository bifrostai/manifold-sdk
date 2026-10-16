# 0010 - A whole-body action is a composite of body-part groups

Status: Pending

## Context

A mobile manipulator acts with more than its arms. RoboCasa's PandaOmron reads 12
values: the arm's position, rotation and gripper, three base velocities, a torso
lift, and the mode value of robosuite's HybridMobileBase controller. It also
reports state no arm field holds, such as the base's pose and the torso's joints.

`PANDA_OMRON_WHOLE_BODY` declares its action as `UnifiedActionSpace(width=12)` with
the arm as the payload, so the last five values are unlabelled. ADR 0001 keeps the
unified action space for a model-side padded buffer, not a robot's action.

The base values cannot be an optional field of `EEActionSpace`. ADR 0001 grows a
spec by optional fields, and a peer without a field drops it. That is tolerable for
a field that declares a value of its own, under its own key, but not for one that
adds slots inside an existing value: the peer then misreads the value's layout, as
the wire's version 5 notes for `arm_count`, and every end-effector adapter, which
rebuilds an action arm by arm, breaks on it.

## Decision

A whole-body action is a `CompositeActionSpace`: groups back to back, in the order
the robot reads them, then an optional mode slot.

- An arm group is an unchanged `EEActionSpace` or `JointActionSpace`.
- `TorsoActionSpace` and `MobileBaseActionSpace` cover the rest, each saying
  whether its values are an absolute target, a delta or a velocity.
- `hybrid_mode_slot` declares where the mode value cuts: `binary_0` (robosuite) or
  `binary_0.5` (RoboCasa365's gym wrapper).

This is not the per-field compositional spec ADR 0001 rejected: each group is a
whole spec with its conventions inline, and the split is by body part.

Proprioception gains `base_pose` (a position then a rotation, in a declared frame),
`torso_pos` (joints, no gripper) and `extras` (any other value, by state key, width
and meaning in words). Each is a new value under its own state key, so each is
additive.

## Consequences

An older SDK refuses a composite action and silently drops `base_pose`,
`torso_pos` and `extras`, so it can accept a pairing this SDK refuses.

A committed spec gains `"base_pose": null`, `"torso_pos": null` and `"extras": []`
when its benchmark moves to this SDK.

`verify` checks a composite group by group, including closed grippers and both
sides of the mode cut. When a chain rebases the ee_pose into the base frame by the
declared base pose, it compares the ee_pose against the probe seen from that pose.

No embodiment declares a composite yet. PandaOmron moves when its pairings do,
and its unified-only adapters (`BasePinWiden`, `UnifiedGripperThresholdAdapter`,
`DiscreteBinarize`) need composite counterparts first. Until then ADR 0001's
Decision 4, which calls the unified action whole-body, still describes PandaOmron.
