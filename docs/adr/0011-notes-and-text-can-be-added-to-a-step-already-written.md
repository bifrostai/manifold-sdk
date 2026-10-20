# 0011 - Notes and text can be added to a step already written

Status: Pending

## Context

A replay log is written forwards, one step at a time, so a step's events and text are
fixed when the step is written. Some of what a benchmark reports is known only later.
RoboCasa has a labeller that splits an episode into subtasks. The labeller needs the
whole episode before it can say on which step each subtask started.

Without a way to add to an earlier step, a benchmark would have to put all of it on the
last step or rewrite the log. A rewrite copies the whole file. It also depends on the
framing that ADR 0005 keeps inside the SDK.

## Decision

`ReplayLogWriter.annotate(step, text=..., notes=...)` appends an annotations frame for a
step that has been written. `read_replay_log` adds the frame's text lines to the end of
that step's channel values, joined with `TEXT_LINE_SEPARATOR`. It adds the frame's notes
after the step's own events.

Notes are the only events that an annotations frame carries. Every other event is
recorded on the step where the simulator observed it (ADR 0009).

## Consequences

`REPLAY_LOG_VERSION` stays 5. A reader from before this change skips a frame type that
it does not know. It reads the log without the added notes and text lines. A grader reads neither notes nor text, so it grades the log
the same way. ADR 0009 bumped the version so that no reader would mistake a log for one
without events. Here the difference is only in what a reader shows, and bumping the
version would make every older reader refuse the whole log.

A tool that reads replay logs with a reader of its own, such as a runner's converter,
must merge annotations frames the same way, or its output lacks them.

`TEXT_LINE_SEPARATOR` is the separator that a runner splits a text channel at. An
annotation refuses a line that would not read back as one line after the join: a line
that holds the separator, starts or ends with part of it, is empty, or has spaces at
either end.
