# Contributing

`manifold-sdk` pairs a robot policy with a benchmark. Contributions are welcome.

## Setup

Install [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just), then:

```
uv sync
```

## Before you push

Run the quality gate — it must be fully green:

```
just check
```

That runs, in order: a formatting check, lint (ruff), type-check (ty), and the
test suite. To format in place, run `just fmt`.

## Conventions

- The design and its reasoning live in the ADRs under `docs/adr/`. Read them
  before changing the core contract.
- Validate at boundaries; rely on types and guards internally.
- Absolute imports only; Google-style docstrings; plain prose (state the fact,
  no figurative language).
- Tests cover project logic, not framework behaviour.
