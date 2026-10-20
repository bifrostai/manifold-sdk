"""Read and check a benchmark's config file.

A benchmark version is a JSON config file plus a hooks file, `hooks.py`, next to it.
The config holds the facts: where upstream comes from, which Service runs it, how
many envs run at once, the run seed and step limit, the values that only the hooks
read, and the spec. The spec is the same JSON as the `spec.json` that registration
reads, and `--check` writes it unchanged. The hooks file holds the upstream calls
that differ between benchmarks.

A benchmark with several images, such as one for each suite, keeps the shared values
in a base config. Each image's config names the base with `extends` and lists only
the values that it changes:

    {"extends": "config-base.json", "run": {"max_steps": 520}}

`load_config` applies those values to the base with JSON Merge Patch (RFC 7386): a
nested key replaces only that key, a list replaces the whole list, and `null`
removes the key.

`BenchmarkConfig` checks a config. `BenchmarkConfig.model_json_schema()` gives the
same rules as a JSON Schema, which an editor can use to check a config as it is
written.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from manifold.benchmark.simulators.isaacsim import BenchmarkIsaacSimSimulatorConfig
from manifold.benchmark.simulators.mujoco import BenchmarkMujocoSimulatorConfig
from manifold.benchmark.simulators.sapien import BenchmarkSapienSimulatorConfig
from manifold.core.benchmark import Benchmark
from manifold.wire import ImageFormat

if TYPE_CHECKING:
    from pathlib import Path


class _BenchmarkConfigBase(BaseModel):
    """A section of the config file, such as `upstream` or `run`.

    A config spells out every value, so a section takes no key that it does not
    list, and it cannot change once it is read.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class BenchmarkVersionConfig(_BenchmarkConfigBase):
    """The benchmark version that the config describes, and its rubric file."""

    version: str
    rubric: str


class BenchmarkUpstreamConfig(_BenchmarkConfigBase):
    """Where upstream's code comes from, and where it is in the image."""

    repo: str
    commit: str
    patches: tuple[str, ...]
    path: str
    data: str


class BenchmarkSingleServiceConfig(_BenchmarkConfigBase):
    """A Service with a simulator for each env."""

    type: Literal["single"]


class BenchmarkBatchedServiceConfig(_BenchmarkConfigBase):
    """A Service with a simulator that holds all envs.

    `reset_together` is set for a simulator that can only restart all envs at once.
    """

    type: Literal["batched"]
    reset_together: bool


class BenchmarkEnvConfig(_BenchmarkConfigBase):
    """How many envs a container runs at once."""

    num_envs: int = Field(ge=1)


class BenchmarkRunConfig(_BenchmarkConfigBase):
    """The run seed, the step limit, the step period, and how images reach the policy.

    `max_steps` is None when upstream ends episodes itself, or when the hooks return
    each task's own limit in `ResetResult.max_steps`. `step_seconds` is the
    simulated time between two steps, such as 0.05 for a 20 Hz controller;
    the replay divides a change between steps by it to plot a velocity.
    `image_format` is how each camera image is encoded on its way to the policy:
    `raw`, `jpeg` or `png`.
    """

    seed: int
    max_steps: int | None = Field(ge=1)
    step_seconds: float = Field(gt=0)
    image_format: ImageFormat


class BenchmarkReplayConfig(_BenchmarkConfigBase):
    """The cameras that each replay frame holds, by the spec's names.

    The live view streams `cameras[0]`.
    """

    cameras: tuple[str, ...] = Field(min_length=1)


class BenchmarkConfig(_BenchmarkConfigBase):
    """A benchmark's config file: its values, its Service, and its spec.

    `simulator` is the simulator that the replay reads its scene and physics
    from, or is None for a benchmark whose replay holds no scene. `spec` is the
    SDK's `Benchmark`: its name, embodiment, sensors and instruction. A policy's
    signature is checked against it.

    `extras` holds the values that only the hooks read, such as a warm-up step count
    from upstream's eval script, under upstream's names. A value that several
    benchmarks share gets its own key instead.
    """

    benchmark: BenchmarkVersionConfig
    upstream: BenchmarkUpstreamConfig
    service: Annotated[
        BenchmarkSingleServiceConfig | BenchmarkBatchedServiceConfig,
        Field(discriminator="type"),
    ]
    extras: dict[str, Any]
    env: BenchmarkEnvConfig
    run: BenchmarkRunConfig
    simulator: (
        Annotated[
            BenchmarkMujocoSimulatorConfig
            | BenchmarkIsaacSimSimulatorConfig
            | BenchmarkSapienSimulatorConfig,
            Field(discriminator="name"),
        ]
        | None
    )
    replay: BenchmarkReplayConfig
    spec: Benchmark

    @model_validator(mode="before")
    @classmethod
    def _check_spec_keys(cls, data: Any) -> Any:
        """Require every key of the spec, because `Benchmark` fills in some of them."""
        spec = data.get("spec") if isinstance(data, dict) else None
        if isinstance(spec, dict):
            keys = set(Benchmark.model_fields)
            missing, unknown = keys - set(spec), set(spec) - keys
            if missing or unknown:
                raise ValueError(
                    f"spec must have exactly the keys {sorted(keys)}; it lacks "
                    f"{sorted(missing)} and has the unknown keys {sorted(unknown)}"
                )
        return data

    @model_validator(mode="after")
    def _check_replay_cameras(self) -> BenchmarkConfig:
        names = {sensor.name for sensor in self.spec.sensors}
        unknown = [camera for camera in self.replay.cameras if camera not in names]
        if unknown:
            raise ValueError(
                f"replay.cameras lists cameras that are not in spec.sensors: {unknown}"
            )
        return self


def _merge_patch(target: Any, patch: Any) -> Any:
    """`target` with `patch` applied by JSON Merge Patch (RFC 7386)."""
    if not isinstance(patch, dict):
        return patch
    merged = dict(target) if isinstance(target, dict) else {}
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = _merge_patch(merged.get(key), value)
    return merged


def _read_config(path: Path, extended_by: tuple[Path, ...]) -> dict[str, Any]:
    """The config at `path` as JSON, with each base config that it extends applied.

    `extended_by` lists the configs that extend this one, to find a loop.
    """
    resolved = path.resolve()
    if resolved in extended_by:
        loop = " -> ".join(str(config) for config in (*extended_by, resolved))
        raise ValueError(f"the configs extend each other in a loop: {loop}")
    config = json.loads(resolved.read_text())
    base = config.pop("extends", None)
    if base is None:
        return config
    return _merge_patch(_read_config(resolved.parent / base, (*extended_by, resolved)), config)


def load_config(path: Path) -> BenchmarkConfig:
    """Read a config file, apply it to the base config that it extends, and check it.

    `extends` is a path relative to the config's directory.

    Raises:
        pydantic.ValidationError: If the config breaks a rule of `BenchmarkConfig`.
        ValueError: If the configs extend each other in a loop.
    """
    return BenchmarkConfig.model_validate(_read_config(path, ()))


__all__ = [
    "BenchmarkBatchedServiceConfig",
    "BenchmarkConfig",
    "BenchmarkEnvConfig",
    "BenchmarkReplayConfig",
    "BenchmarkRunConfig",
    "BenchmarkSingleServiceConfig",
    "BenchmarkUpstreamConfig",
    "BenchmarkVersionConfig",
    "load_config",
]
