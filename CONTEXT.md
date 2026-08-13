# manifold-sdk domain model

Terms the contract relies on. Add a term when it resolves in design discussion;
keep the definition to what distinguishes it from its neighbours.

## The replay

**replay** - the Rerun recording of one episode, which a runner builds from that
episode's replay log once it ends. The SDK defines the log; the recording is
the consumer's.
_Avoid_: recording, rollout video, trace

**replay log** - the per-episode file of simulator state a benchmark extracts
for the replay, holding scene geometry, per-step body poses and contacts.
Written beside the rollup in the results directory, and read there by the runner,
so it is never part of the stream the two sides exchange, and no policy signature
declares it. See ADR 0003.
_Avoid_: telemetry, info, sidecar, trace

**success signal** - the per-step flag saying the benchmark's task is satisfied,
declared by `Benchmark.success_signal`. Distinct from an episode's recorded
`success`, which is the rollup's verdict once the episode ends.
_Avoid_: reward, done, terminated
