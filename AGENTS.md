# AGENTS.md — manifold-sdk code conventions

House conventions for this repository. They describe how the code is *shaped*;
the rule for *which type mechanism to reach for* (BaseModel / dataclass / ABC /
Protocol / StrEnum / Literal) lives in `docs/adr/0001-architecture.md` §10. The
quality gate is in `CONTRIBUTING.md`.

## Module layout

Order the top-level members of a module, top to bottom:

1. module docstring
2. `from __future__ import annotations`
3. imports (and any `if TYPE_CHECKING:` block)
4. module-level constants
5. private helpers (`_name` functions and classes)
6. public classes and functions
7. `__all__`

A private helper is defined **above** the public member that uses it (define
before use). Where a module-load-time reference forces a different order (a
constant that instantiates a class defined below it), keep the order the load
requires.

## Public vs private

A module-level name is **public** (no leading underscore, listed in `__all__`)
iff it is referenced outside its defining module; otherwise it is `_`-prefixed. A
module that is deliberately internal is named `_name.py` (as `adapters/packing/_ops.py`
is). Every non-private submodule is re-exported through its package `__init__`, so
nothing public is reachable only by a deep import.

## `__all__`

Every module with public members declares `__all__` at the end of the file, one
entry per line. Its ordering is enforced by ruff `RUF022` (ALL_CAPS constants,
then classes, then functions) — run `just fmt`; do not hand-sort.

## Imports

Absolute only — relative imports are banned (ruff `TID`). Import at module top
unless (a) breaking a real import cycle or (b) loading an optional dependency;
annotate either exception with a one-line `#` reason (as `core/verify.py` and
`wire/bridge.py` do). Type-only imports go under `if TYPE_CHECKING:`. Every module
begins with `from __future__ import annotations`.

## Comments

- Document a module-level constant with a `#` comment on the line(s) **above** it —
  not an attribute docstring below it.
- Inline comments are full sentences: capitalize the first word (backtick a
  leading code token, e.g. `` # `eq=False` avoids a numpy __eq__. ``) and end with
  a period.
- No ASCII-divider section banners. Separate sections with a blank line and, if
  needed, a plain `#` comment.

## Docstrings

Google convention (`pydocstyle`), kept terse.

- Every module opens with a one-line noun-phrase summary ending in a period; add a
  blank line and a body only when the module needs it.
- First line by role: **imperative** for behaviour (adapters and acting methods —
  "Resize…", "Append…"); a **noun phrase** for data/enums, `@property` getters,
  and `produce()`-style return descriptions ("The same spec with…").
- Describe parameters and return values in prose, **not** Google `Args:`/`Returns:`
  blocks. Use `Raises:` only for a typed public exception a caller is expected to
  catch.

## Naming

- Constants are `UPPER_SNAKE_CASE`.
- Factory constructors are named `from_*` (`from_camera`, `from_state`, `from_array`) —
  not `of` / `build` / `create` / `make`.
- Booleans on the public API are bare adjectives/nouns (no `is_` / `has_` prefix).

## Enums

Always `StrEnum` (via `manifold.lib.compat.StrEnum` for the 3.10 floor). Public,
and re-exported through the package `__init__` alongside their siblings.

## Exhaustiveness

Closed-set dispatch (every `StrEnum` member, every discriminated-union variant)
ends in `else: assert_never(x)` (`manifold.lib.compat.assert_never`) so `ty`
proves exhaustiveness at check time — never a runtime `raise ValueError("unknown
…")` tail. Use `if`/`elif` chains — `isinstance` for union variants, `is` for
enum members — not `match`: `ty`'s negative narrowing of `match` class patterns
is unimplemented (`@Todo`), which `assert_never` silently accepts, making a
`match`-based check inert (it won't fire even with a variant missing). Verify a
new exhaustiveness check by deleting a branch and confirming `ty check` errors
before trusting it.

This only applies to a value the type system actually closes over (an enum or a
`Field(discriminator=...)` union). A parameter typed as plain `str`/`Any` that
happens to come from a closed set at the call site (e.g. a wire format read off
an untrusted frame, or a dict-lookup miss on caller-supplied input) is a runtime
boundary check, not exhaustiveness — keep its `raise`.

## Domain vocabulary

Use one word per concept:

- **signature** — the policy side of a pairing (`PolicySignature`). Not "contract".
- **spec** — a pydantic shape/channel descriptor (`EEActionSpace`, `ObservationSpace`,
  the `*Spec` family, `from_spec`/`to_spec`). Not "contract".
- **adapter** — a transform object in the seam (the `Adapter` hierarchy). "transform"
  is fine as a *verb*; the object is an adapter.
- **contract** / **protocol** — reserve for the wire agreement (the bridge and codec),
  not the policy or its specs.
- **sensor** vs **camera** — `sensor` is the superset; `camera` is today's only
  modality (`Sensor = Camera`). Keep them distinct.
- **policy** vs **model** — `policy` is the role; `model` is the checkpoint. Keep them
  distinct.

## Quality gate

`just check` (format check, lint, type-check, pytest) must be green before pushing.
Never push red.
