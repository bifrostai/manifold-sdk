# 0012 - A benchmark version may hold several embodiments

Status: Pending

## Context

A `Benchmark` holds one embodiment, and the `Client` sends that one spec on every
policy connection for the whole run. Each action is checked against that one action.

Some benchmarks vary the robot from task to task. ManiLadder's 112 tasks use single
arms, dual arms, grippers, dexterous hands and a mobile base, and its levels and
categories each mix several of them. Under one embodiment per version, such a
benchmark either splits into many versions that no upstream table matches, or leaves
out the tasks that its spec cannot describe.

ADR 0001 designs for one benchmark across several embodiments and defers it until a
benchmark needs it. ManiLadder needs it.

## Decision: the spec lists each embodiment

`Benchmark` gains an optional `embodiments` field. Each entry holds the fields that
describe one robot: its embodiment, its cameras, its action range and its end
effectors. No two entries
share an embodiment name, because a policy server finds a pairing by benchmark name
and embodiment name.

When `embodiments` is set, the top-level `embodiment`, `sensors` and `action_range`
equal the first entry, and a validator checks it. An older reader then still sees a
valid single-embodiment spec. A spec without `embodiments` carries
`"embodiments": null`, which older readers ignore.

`Benchmark.for_embodiment(name)` returns the single-embodiment `Benchmark` for one
entry. Run values such as `step_seconds` and `image_format` stay one value for the
version.

A benchmark config may give each embodiment its own finger pads and simulator robot
fields. An Isaac Sim config refuses several embodiments, because Isaac Sim loads one
robot for the whole container.

## Decision: one connection for each embodiment

Before the first episode, the `Client` opens one policy connection for each
embodiment and each env. Each connection sends today's HELLO with the spec from
`for_embodiment`, so the policy checks each embodiment on its own. If the policy
refuses any embodiment, the run fails at start-up and runs no episode. Every
connection opens, even on a shard whose tasks use one embodiment, so that a refusal
shows on every shard.

## Decision: the hooks give each episode's embodiment

The hooks return the episode's embodiment name in `ResetResult.embodiment`. None
means the spec's only embodiment. The `Client` fails the run when the hooks return
None for a spec with several embodiments, or a name that the spec does not list,
because the hooks then disagree with the spec.

The `Client` runs the episode on that embodiment's connection, checks each action
against that embodiment's action, and adds the name to the episode's `attributes`. The recorder writes each episode's replay header from that
embodiment.

The rubric generator puts each task's embodiment in the task's attributes, so that
the benchmark page shows which tasks use which robot before a run. The hooks stay the
only source at run time, and the rubric attribute is for display.

## Decision: each pairing may state its own signature

One server already serves several pairings of one profile, and finds each by
benchmark name and embodiment name. Two things are new. A `Pairing` may carry its
own `PolicySignature`, which the pairing check uses in place of the profile's,
because a dual-arm robot needs a second wrist camera that a single-arm signature
lacks. A pairing module may export `PAIRINGS`, a list, beside today's `PROFILE`,
`BENCHMARK` and `PIPELINE`.

The pairings of one server still share one profile.

## Decision: each embodiment renders only the cameras its pairing reads

The SIGNATURE reply keeps `signature` and gains `signatures`, a new top-level key
with each pairing's signature under its embodiment name. The bridge protocol rule
calls a new top-level key additive, so the protocol version does not change.

Before it builds the scene, the benchmark narrows each embodiment's cameras to the
cameras that its pairing reads, plus the replay's cameras, as it does today for one
embodiment. An embodiment keeps every camera when the reply has no `signatures`, when
no pairing covers it, or when its pairing reads a camera that it does not publish.

A policy that pairs with one embodiment keeps its module and launcher unchanged.

## Alternatives

- **The embodiment on RESET over one connection.** An older policy ignores an
  unknown RESET key and keeps its first pipeline, so this needs a new bridge protocol
  version and a refusal of older policies.
- **A new handshake before each episode.** A refused embodiment shows up mid-run,
  after other episodes have finished, and each episode pays a handshake.
- **A padded action with no stated meaning.** Only the one checkpoint trained on that
  padding could pair, and the spec would no longer state rotation, gripper sign or
  arm count.

## Consequences

A policy must accept every embodiment of a version to run it. A policy trained on
part of a benchmark needs a version that holds only that part.

An older policy refuses an embodiment name that its SDK does not list, so the run
fails at start-up. Each new robot still needs an `EmbodimentName` and an SDK release.

The policy server holds one connection and one session for each env and embodiment.

The manifold backend reads only the top-level embodiment, for failure analysis and
the operator catalog. Before a version with several embodiments is registered, both
must read each episode's embodiment attribute. Otherwise failure analysis applies the
first embodiment's hand joints to every episode.

## Consequences for scores

The SDK does not check normalisation statistics. A policy with wrong statistics for
one embodiment pairs and scores 0% on it with no error.

A version's score averages over its episodes, so an embodiment with more tasks
weighs more. Results show a score per embodiment through the episode attributes.
