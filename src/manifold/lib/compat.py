"""Small standard-library compatibility shims.

`enum.StrEnum` and `typing.assert_never` only exist on Python 3.11+, but the
SimplerEnv/GR00T evaluation venvs are pinned to Python 3.10 (SAPIEN and the
Isaac-GR00T stack require it). Re-export the stdlib versions where available
and fall back to equivalents on 3.10, so the SDK imports and behaves
identically on both.
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
    from typing import assert_never
else:  # pragma: no cover - exercised only on Python 3.10 runtimes
    from enum import Enum
    from typing import NoReturn

    class StrEnum(str, Enum):
        """Backport of py3.11 enum.StrEnum: members are also plain strings."""

        def __str__(self) -> str:
            return str(self.value)

    def assert_never(arg: NoReturn, /) -> NoReturn:
        """Backport of py3.11 typing.assert_never: unreachable-code assertion."""
        raise AssertionError(f"Expected code to be unreachable, but got: {arg!r}")


__all__ = ["StrEnum", "assert_never"]
