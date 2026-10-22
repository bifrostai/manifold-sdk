# 0009 - Manifold events are typed and stored on each step

Status: Accepted

## Context

The grader needs to know what happened during an episode, such as grasps,
collisions, and each fact about the scene that started or stopped holding.
Benchmarks reported this as lines of text that the grader matched with patterns.
A line worded differently matched nothing and raised no error, so every new
benchmark had to copy an earlier benchmark's wording.

Some failures, such as the arm hitting the table, depend on values that only
the simulator has. The grader cannot work them out reliably from the replay.

## Decision

Each replay step carries a list of Manifold events. Each type of event has its own
class, and stores its type as `event_type`, such as `inside`.

An event has no step of its own. The step that stores it is when it happened.

A benchmark records what its simulator observed. A fact, such as `grasped` or
`inside`, is recorded when it starts or stops holding. An occurrence, such as
`gripper_hit`, is recorded when it happens. All benchmarks share one list of
names, so a fact has the same name in every benchmark.

The grader decides what the events mean, such as whether a task's conditions
were met. The grader also decides whether a failure happened, unless the
benchmark records that failure itself (see Failures).

`Note` is the only free text, and the grader never acts on it.

## What a benchmark records

A benchmark records every fact its task uses that holds at step 0, and after
that only the changes. It uses the same object names as the task, and calls the
gripper `gripper`. `grasped` means the gripper is closed on the object and holds
it. `gripper_touching` means the gripper is in contact with the object.

## Failures

A benchmark can record a failure event with one of the types listed below.
Nothing outside the table counts as a failure.

It does this only for a type that its simulator detects more accurately than the
grader can from the replay. It shows this on labelled episodes before it starts
recording the type.

The log header lists the types that the benchmark records. For those types, no
event means no failure. The grader works out every other type itself.

| Category | Types |
|---|---|
| Execution, low-level | `no_contact`, `grasp_position_offset`, `grasp_orientation_offset`, `no_grasp`, `place_position_offset`, `place_orientation_offset`, `actuation_incomplete` |
| Execution, unexpected dynamics | `slip_drop`, `collision_object`, `collision_environment`, `object_moved_unaided` |
| Execution, environment | `environment_occupied`, `environment_blocking` |
| Planning, wrong action, or perception | `wrong_destination`, `wrong_object`, `view_occluded` |
| Planning, missing action | `prerequisite_skipped`, `step_missed` |
| Planning, wrong order | `wrong_order` |
| Planning, ambiguous reference | `ambiguous_instruction` |

## Consequences

Every step stores an `events` list, empty when nothing happened, and
`REPLAY_LOG_VERSION` is 5. An SDK older than this change refuses a version 5
log rather than reading it without its events, so nothing mistakes it for a
log whose benchmark recorded none. A version 3 or 4 log still reads, with no
events.

Each benchmark decides how its simulator evaluates a fact. The list fixes the
names, not the thresholds.

A tool that converts a replay log into another format must copy the events, or
the grader cannot see them in the converted file.

An SDK that does not know the `failure` event refuses a log that contains
one. It ignores the list of types in the header.
