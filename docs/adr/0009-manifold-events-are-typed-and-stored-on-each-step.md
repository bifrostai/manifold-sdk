# 0009 - Manifold events are typed and stored on each step

Status: Accepted

## Context

A reader of a replay needs to know what happened during an episode, such as
grasps, collisions, and each fact about the scene that started or stopped
holding. Benchmarks reported this as lines of text that a reader matched with
patterns. A line worded differently matched nothing and raised no error, so
every new benchmark had to copy an earlier benchmark's wording.

## Decision

Each replay step carries a list of Manifold events. Each type of event has its own
class, and stores its type as `event_type`, such as `inside`.

An event has no step of its own. The step that stores it is when it happened.

A benchmark records what its simulator observed and nothing it derived. A fact,
such as `grasped` or `inside`, is recorded when it starts or stops holding. An
occurrence, such as `gripper_hit`, is recorded when it happens. All benchmarks
share one list of names, so a fact has the same name in every benchmark.

The reader decides what the events mean, such as whether a task's conditions
were met, or whether a drop or a hit counts as a failure.

`Note` is the only free text, and a reader never acts on it.

## What a benchmark records

A benchmark records every fact its task uses that holds at step 0, and after
that only the changes. It uses the same object names as the task, and calls the
gripper `gripper`. `grasped` means the gripper is closed on the object and holds
it. `gripper_touching` means the gripper is in contact with the object.

## Consequences

Every step stores an `events` list, empty when nothing happened, and
`REPLAY_LOG_VERSION` is 5. A reader older than this change refuses a version 5
log rather than reading it without its events, so no reader mistakes it for a
log whose benchmark recorded none. A version 3 or 4 log still reads, with no
events.

Each benchmark decides how its simulator evaluates a fact. The list fixes the
names, not the thresholds.

A tool that converts a replay log into another format must copy the events, or
readers of the converted file do not see them.
