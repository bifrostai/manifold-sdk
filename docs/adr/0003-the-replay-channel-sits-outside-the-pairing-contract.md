# 0003 - The replay channel sits outside the pairing contract

Status: Accepted

## Context

A replay of an episode needs more than the two sides exchange. Rendering the arm
and the objects needs the scene's geometry once and a pose per body per step;
naming a grasp needs per-object contact. No policy reads any of it.

`check_compatibility` folds a pipeline over the composite `ObservationSpace`,
which `Benchmark.observation_space` derives from the embodiment's
proprioception, the published sensors and the instruction flag. A channel added
there is a channel every pairing has to reconcile: a benchmark either publishes
it or declares it does not, and a policy signature declares it ignores it. The
check would then arbitrate channels no policy consumes.

Nor does the data belong on the wire at all. A benchmark worker already writes
its rollup into a directory the runner mounts, one per run and per shard, and a
runner starts the workers for its own side, so the two always share a
filesystem.

## Decision

A channel belongs to the contract if a policy could consume it, and to the
replay otherwise.

Depth is in the contract. An RGB-D policy is a real consumer, `Camera.dtype`
already expresses a float camera, and only the codec cannot carry one yet.

Geometry, per-step body poses and contacts go in a replay log: a file per
episode, written beside the rollup, holding the state a benchmark extracted from
its simulator. The runner reads the log and builds the replay from it.

## Consequences

The bridge stays byte-verbatim and frame-blind. Nothing parses the stream to
separate the replay from the run, so no framing defect can corrupt a policy's
input, and the frame types and protocol version are untouched by this decision.

Meshes never cross the wire. The largest payload in a replay is written locally
to a mount instead.

Building the replay stays in the runner, so `rerun` is a runner dependency and
no benchmark image carries it. What a benchmark writes is extracted simulator
state; what a runner writes is the recording.

A policy obtains depth only by declaring it. Nothing else can reach it, because
the replay log is never part of the stream the policy reads.

`rollup_in` requires exactly one `*.json` in the results directory, so a replay
log cannot be JSON. That constraint is the existing rollup contract, not a new
one this decision adds.

A benchmark that does not write a replay log still runs. Its episodes yield the
camera and plot views and no rendered scene, so the panels a replay offers vary by
benchmark rather than being guaranteed by the contract.
