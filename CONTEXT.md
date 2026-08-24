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
declares it. Reuses the wire's framing: length-prefixed msgpack frames, appended
as the episode runs. See ADR 0003 and ADR 0005.
_Avoid_: telemetry, info, sidecar, trace

**scene body** - one named rigid body in a replay log, carrying a visual mesh and
a pose per step. The unit a replay's rendered scene is assembled from, covering
the arm's links and the gripper's fingers as well as the objects and fixtures,
because nothing reading a log distinguishes them. See ADR 0004.
_Avoid_: link, geom, entity

**success signal** - the per-step flag saying the benchmark's task is satisfied.
Replay data rather than a contract channel, since no policy consumes it, so a
replay log declares it and `Benchmark` does not. Distinct from an episode's
recorded `success`, which is the rollup's verdict once the episode ends.
_Avoid_: reward, done, terminated

## Serving

**remote driver** - a driver whose endpoint does not hold a model and answers each
forward by calling an inference server it dials over the network. The signature,
the native layouts and the session are unchanged, so a policy served this way is
not a distinct kind of version: only `forward` differs.
_Avoid_: proxy policy, hosted policy, remote policy
