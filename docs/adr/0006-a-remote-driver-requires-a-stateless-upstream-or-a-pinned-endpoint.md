# 0006 - A remote driver requires a stateless upstream, or a pinned endpoint

Status: Accepted

## Context

A driver may answer each forward by calling a policy server over the network,
rather than a model loaded in a process on its own machine.

A model in a local process answers every call itself. A remote server may answer
two calls of one session from two different instances: a load balancer in front
of it, or an autoscaler behind it, routes each call independently, and the
calling driver cannot tell which instance answered.

That matters because some policies are stateful. A policy that answers each
observation from that observation alone is indifferent to which instance answers
it. A model holding its own per-session scratch is not: step 21 may reach an
instance that was never sent steps 1 to 20. It answers from a blank state and
returns a plausible action. Nothing raises, the rollout completes, and the score
is indistinguishable from a weak policy's.

Chunking itself is safe. `OpenLoopChunkQueue` holds the chunk buffer in the
driver, on this side of the network, so an upstream call carries only the native
dict.

## Decision

Either a remote driver's upstream answers from the native dict alone, so that
routing cannot matter, or its endpoint is pinned to one instance, so that
routing is fixed for the whole run.

## Consequences

The two shapes differ in throughput by roughly the shard count. A stateless
upstream serves N shards from N instances, each under its own lock, so N
forwards run at once - more concurrency than a locally served policy, whose
shards share the one endpoint lock. A pinned instance queues all N shards behind
one forward and adds a network round trip per call, which makes it slower than
serving the same policy locally.

Neither `check_compatibility` nor `verify` can detect a violation.
`check_compatibility` compares a signature against a benchmark. `verify` runs
synthetic probe data through the pipeline without loading a model or minting a
session, so it reports real policy behaviour as `not_checked`.

The requirement is therefore prose only, and a disputed score cannot be settled
from either gate. The cheap way to detect a violation is for every response to
carry the identity of the instance that answered it, and for the driver to raise
when that identity changes within a session.
