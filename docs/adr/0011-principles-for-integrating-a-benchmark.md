# 0011 - Principles for integrating a benchmark

Status: Accepted

## Context

Each benchmark that we integrated came with its own copy of an episode loop and a
recorder. A developer who fixed a bug in one copy left the bug in the other
copies. The copies also disagreed with one another. For example, they defined a
grasp in three ways. To integrate a new benchmark, a developer had to
learn one of these copies and copy it again.

We want two things at once. A developer should integrate any benchmark fast. The
results should not lose anything that the earlier benchmarks recorded.

## Decision

### What the author writes

A benchmark differs from other benchmarks in its values and in how it calls its
upstream code. The author writes those two things, as a config file and a hooks
file. Shared code does the rest.

The hooks file only calls upstream. Images and actions stay in upstream's form,
because the config's spec describes that form, and the SDK's pipeline converts it
for each policy.

### Work

A benchmark gets its work from the task queue. The server builds the queue from the
benchmark's rubric, which lists the tasks and the number of episodes for each. A
container claims a task, runs its episodes, and reports the results.

### Values

The author states every value that the benchmark uses. Shared code does not fill
in a value that the author left out. A reader then sees all the values in a
single place.

### Lifecycle steps

Shared code runs the same steps for all benchmarks: start, load, reset, step,
unload and stop. The author writes each step separately, and no piece of
benchmark code runs several steps at once. A reader can then see where each call
to upstream happens, and shared code can rely on the order of the steps.

A hook does one step and returns. A value that a later hook needs goes on the env
that `on_load` returns. RoboLab's `on_load`, for example, returns a `RoboLabEnv`
with Isaac Lab's env and the task's facts.

Some values stay the same for a whole episode, such as the instruction. `on_reset`
returns such a value once, and the Client puts it on each observation of the
episode. In the same way, `env_key` returns what identifies a task's env, and the
Service keeps the env for the next task when the key is the same.

### Levels of code

We put code at one of three levels:

- **Shared:** code that all benchmarks need.
- **Simulator:** code that all benchmarks on one simulator need.
- **Benchmark:** code that only one benchmark needs.

A framework on top of a simulator adds values but no new level.

We prefer a rule in shared code to a setting in each benchmark. With a setting,
each author must look up a value that shared code could work out.

### Reports and their meaning

A benchmark reports what its upstream code observed, such as whether an object is
on a plate. It does not decide what that means for the episode. Failure analysis
and the rubric decide what the reports mean. Failure analysis can then change
without a change to any benchmark.

Shared code records the reports in the same way for all benchmarks. It does not
depend on which report it records.

### Seeds

Each hooks file sets its own seed, because each upstream takes the seed through its
own call. MuJoCo and SAPIEN take a seed at each reset, Isaac Lab takes one seed for
a whole batch of envs, and Arena takes one when it builds an env.

### Benchmarks that disagree

Where earlier benchmarks disagreed, we write a single rule in shared code. The
records keep the information that the earlier benchmarks kept, even where the
rule changes a value. We list each such change for reviewers.

### Names

A concept has a single name, and the name says which part of the SDK defines the
concept. A reader who meets a name can then find where it comes from. A new name
needs a concept that no existing name covers.

### Simulator dependencies

The SDK installs and imports without any simulator. Code for a simulator imports
that simulator only when it runs. Machines that run no benchmark, such as policy
containers and developer laptops, can then install the SDK.

### Replays

Shared code writes the replay of an episode. A simulator class reads values from
the simulator, such as the pose of a body or the torque of a joint. The
simulator class writes nothing and stores no history of the episode.

The Client runs the episode loop, so it has the action, the observation, the
events and the result of each step. The Client gives these values to its replay
recorder. The replay recorder asks the simulator class for the values that depend
on the simulator. It then writes the step to the replay file.

MuJoCo and Isaac Sim read a pose or a torque in different ways, so that code is
at the simulator level. The content of a replay is the same for all benchmarks,
so that code is at the shared level. If each simulator class wrote its own
replay, each one would have its own copy of the code for the events and the
file.

The Service runs the lifecycle steps, and the simulator's process starts and
stops with them. The Service therefore creates the simulator, and calls
its `start`, `attach`, `post_step` and `stop` methods.

### Simulator classes

The code for a simulator is in a module of `manifold.benchmark.simulators`, such
as `mujoco.py`. The modules name their parts in the same way. After a reader
learns one module, the reader can read the others. A module has these parts:

- `Benchmark<Simulator>SimulatorConfig`: the simulator's values from the
  benchmark's config.
- `<Simulator>Simulator`: the subclass of `Simulator`.
- Private constants, in capitals with a leading underscore.

Some simulators are slow to look up values that stay the same for an episode,
such as joint indices. Such a module also has `_<Simulator>EpisodeCache`.
`replay_bodies` reads these values at the start of an episode. The other replay
methods then use the cache at each step.

The simulator class has only the public methods and the property that `Simulator`
declares. Shared code calls only these, so it would not call a new public method.
The lifecycle methods are `start`, `stop`, `attach` and `post_step`. The names
of the replay methods start with `replay_`. Only the Client's replay recorder
calls the replay methods. The property `supported_versions` gives the version
range of each Python package that the simulator reads, and the Service logs a
warning for an installed version outside it.

One public method can be added. Some config values have to be set on an env before
the env is built, and the simulator gets the env only after it is built. The
simulator class then sets them in a static method, which the hooks call while they
build the env, such as `IsaacSimSimulator.apply_settings`.

The other methods are private, and their names start with an underscore:

- If a private method reads `self`, it is an instance method.
- If a private method uses only its arguments, it is a static method. A reader
  can then see from its definition that it has no state.
- If a private method returns a value, its name is the name of that value, such
  as `_body_name`.
- If a private method returns nothing, its name is a verb for what it does, such
  as `_log_skipped_geoms`.

Private methods are methods of the class. When a reader sees a name such as
`_body_name`, the reader finds its definition in the same class. If two
simulators share code, that code is a public function in `simulators/__init__.py`,
such as `bounding_box_part`.

### Hooks files

The next author of a hooks file starts from an existing hooks file, so each hooks
file follows these rules:

- The hooks come first, in the order that the Service calls them. The code that
  they use comes after them.
- A comment says why the code does something, in one to three lines. History,
  counts and steps to undo a change go in the commit message.
- The file has no commented-out code. It reads no switch that no config or image
  sets.
- The code checks only for cases that the pinned upstream version produces.
- Names come from the SDK or from the upstream code.
- A hook writes its messages through the `logger` that the Service passes it. The
  file uses neither `print`, `warnings.warn` nor a logger of its own.

### Crashes

A benchmark does not recover from a crash. The process ends, and the system that
handed out the work retries it. Recovery code in each benchmark would repeat that
retry.

## Consequences

Integrating a benchmark costs only what is different about it. A benchmark on a
simulator that the SDK supports gets the full record of its episodes. A benchmark
on another simulator gets a smaller record until someone adds that simulator.

Where earlier benchmarks disagreed with one another, the new records differ from
theirs. Reviewers check each difference against the list.

A new or renamed concept changes the SDK's public names. Stored records keep their
own names, so old records still read.
