"""The config file of a benchmark, and the functions to read and check it.

A benchmark version consists of a JSON config file and a hooks file, `hooks.py`, in
the same directory. The config states where upstream comes from, which Service runs
it, how many envs run at once, the run seed and the step limit, the values that only
the hooks read, and the spec. `--check` writes the spec and the run values to the
`spec.json` that registration reads. The hooks file contains the upstream calls.

A benchmark with several images, such as an image for each suite, can put the shared
values in a base config. The config of each image lists the base config in `extends`
and lists only the values that it changes:

    {"extends": "config-base.json", "run": {"max_steps": 520}}

`load_config` applies those values to the base with JSON Merge Patch (RFC 7386): a
nested key replaces only that key, a list replaces the whole list, and `null`
removes the key.

`BenchmarkConfig` checks a config. `BenchmarkConfig.model_json_schema()` returns the
same rules as a JSON Schema. An editor can use the schema to check a config while a
developer writes it.
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

    A section rejects keys that it does not declare, because the config states every
    value. A section cannot change after `load_config` reads it.
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
    """A Service with a simulator that contains all envs.

    `reset_together` is true for a simulator that can reset its envs only all at once.
    """

    type: Literal["batched"]
    reset_together: bool


class BenchmarkEnvConfig(_BenchmarkConfigBase):
    """How many envs a container runs at once."""

    num_envs: int = Field(ge=1)


class BenchmarkRunConfig(_BenchmarkConfigBase):
    """The run seed, the step limit, the step period, and how images reach the policy.

    `max_steps` is None when upstream ends episodes itself, or when the hooks return the
    limit of each task in `ResetResult.max_steps`. `step_seconds` is the simulated time
    between two steps, such as 0.05 for a 20 Hz controller. The replay recorder divides
    a change between two steps by `step_seconds` to compute a velocity. `image_format`
    is the encoding of each camera image on its way to the policy: `raw`, `jpeg` or
    `png`.
    """

    seed: int
    max_steps: int | None = Field(ge=1)
    step_seconds: float = Field(gt=0)
    image_format: ImageFormat


class BenchmarkReplayConfig(_BenchmarkConfigBase):
    """The cameras that each replay frame contains, by their names in the spec.

    The live view streams `cameras[0]`.
    """

    cameras: tuple[str, ...] = Field(min_length=1)


class BenchmarkConfig(_BenchmarkConfigBase):
    """A benchmark's config file: its values, its Service, and its spec.

    `simulator` configures the simulator that the replay reads the scene and physics
    from. It is None for a benchmark whose replay contains no scene. `spec` is the
    SDK's `Benchmark`: its name, embodiment, sensors and instruction. Pairing checks a
    policy's signature against `spec`.

    `extras` contains the values that only the hooks read, under upstream's names. An
    example is a count of warm-up steps from upstream's eval script. A value that
    several benchmarks use gets its own key in the config.
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
        """Require all keys of the spec, because `Benchmark` has defaults for some keys."""
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
    """The JSON of the config at `path`, merged onto each base config that it extends.

    `extended_by` lists the configs that extend this config. The function uses the list
    to detect a loop.
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
