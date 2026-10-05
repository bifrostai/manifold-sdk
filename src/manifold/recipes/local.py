"""Start a local policy server."""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable

from manifold.adapters.observation.camera_resolution import ResizeCameras
from manifold.adapters.observation.camera_rotate_180 import Rotate180Cameras
from manifold.adapters.observation.camera_vertical_flip import FlipVerticalCameras
from manifold.adapters.observation.proprio_rotation import ProprioRotationAdapter
from manifold.core.adapter import ActionAdapter, ObservationAdapter
from manifold.core.benchmark import Benchmark
from manifold.core.pipeline import Pipeline
from manifold.core.policy import PolicySignature
from manifold.core.values import Action, Observation
from manifold.core.verify import probe_observation
from manifold.recipes.function import (
    FunctionEndpoint,
    StatefulEndpoint,
    _StatefulPolicy,
    check_chunk,
)
from manifold.recipes.resolve import resolve
from manifold.recipes.serving import serve as serve_endpoint

# The instruction the startup probe sends when the signature needs one. Some
# models (pi05, for one) reject an empty instruction.
_PROBE_INSTRUCTION = "probe"


def _server_address(server: str) -> tuple[str, int]:
    value = server.removeprefix("tcp:")
    host, separator, port = value.rpartition(":")
    if not host or not separator or not port:
        raise ValueError("MANIFOLD_SERVER_URL must use tcp:HOST:PORT")
    return host, int(port)


def _default_pool(
    signature: PolicySignature,
    benchmark: Benchmark,
) -> list[ActionAdapter | ObservationAdapter]:
    """Build the built-in adapters that could bridge `benchmark` to `signature`.

    Only cameras that both sides declare get adapters. A camera whose
    orientation differs gets its own flip and rotation, so the resolver can fix
    each camera separately. A camera whose shape differs gets a resize. The
    proprioception rotation is added when the two `ee_pose` formats differ.
    """
    pool: list[ActionAdapter | ObservationAdapter] = []
    published = {camera.name: camera for camera in benchmark.sensors}
    resize_targets: dict[str, tuple[int, ...]] = {}
    for camera in signature.cameras:
        source = published.get(camera.name)
        if source is None:
            continue
        if source.orientation != camera.orientation:
            pool.append(FlipVerticalCameras(cameras=(camera.name,)))
            pool.append(Rotate180Cameras(cameras=(camera.name,)))
        if source.shape != camera.shape:
            resize_targets[camera.name] = camera.shape
    if resize_targets:
        pool.append(ResizeCameras(targets=resize_targets))
    policy_ee_pose = signature.proprioception.ee_pose
    benchmark_ee_pose = benchmark.embodiment.proprioception.ee_pose
    if (
        policy_ee_pose is not None
        and benchmark_ee_pose is not None
        and policy_ee_pose.rotation != benchmark_ee_pose.rotation
    ):
        pool.append(ProprioRotationAdapter(target=policy_ee_pose.rotation))
    return pool


def _probe_predict(predict: Callable[[Observation], Action], signature: PolicySignature) -> None:
    """Call `predict` once on a synthetic observation and check the result.

    The values are discarded. A result that does not match the signature's
    `chunk_size` and action space raises here, before `serve` binds the port,
    so the script exits in the user's terminal instead of failing a run.
    """
    observation = probe_observation(signature.observation_space, instruction=_PROBE_INSTRUCTION)
    check_chunk(predict(observation), signature)


def _stateful_predict(policy: type[_StatefulPolicy]) -> Callable[[Observation], Action]:
    """Create an instance of `policy`, call its `reset`, and return its `predict`.

    Raises `TypeError` if the class lacks a `reset` or a `predict` method.
    """
    missing = [name for name in ("reset", "predict") if not callable(getattr(policy, name, None))]
    if missing:
        raise TypeError(
            f"{policy.__name__} must define reset() and predict(); it lacks "
            f"{' and '.join(f'{name}()' for name in missing)}"
        )
    instance = policy()
    instance.reset()
    return instance.predict


def serve(
    policy: Callable[[Observation], Action] | type[_StatefulPolicy],
    signature: PolicySignature,
    *,
    pipeline: Pipeline | Callable[[Benchmark], Pipeline] | None = None,
    server: str | None = None,
) -> None:
    """Serve `policy` on the local address from `MANIFOLD_SERVER_URL`.

    `policy` is either a stateless `predict` function, or a stateful class
    that implements `reset()` and `predict()`. Each shard gets its own
    instance of the class, and `reset()` is called at the start of every
    episode. Load all weights at the top level of the script, outside the
    function or the class.

    ```python
    model = load_model("checkpoint/")

    class Policy:
        \"\"\"Stateful class where reset() is called at the start of every episode.

        Do not load weights in this class.
        \"\"\"

        def reset(self):
            self.hidden = None

        def predict(self, obs):
            actions, self.hidden = model(obs, self.hidden)
            return actions

    manifold.serve(Policy, signature)
    ```

    `serve` calls `predict` from a separate thread for each shard, so calls
    from different shards can run at the same time. The weights at the top of
    the script are shared by all shards. For a model on a GPU, hold a single
    lock around all GPU work in `predict`, including preprocessing and noise,
    so that one call at a time uses the GPU:

    ```python
    lock = threading.Lock()

    def predict(obs):
        with lock:
            return model(obs)
    ```

    A model call returns the actions. Under the default
    `chunk_size=1`, it returns a single action as a flat vector or a single row.
    Otherwise it returns a 2-D array with `signature.chunk_size` rows, and
    `signature.action_space` describes each row. Each shard runs
    `signature.execution_steps` of those actions before the next call. Before
    `serve` opens the port, it calls `predict` once on a synthetic observation,
    with a separate instance for a class. It raises `ValueError` if the result
    does not match the signature, and `TypeError` if a class lacks `reset()`
    or `predict()`.

    Pass `pipeline` to choose the adapters yourself, as a `Pipeline` or as a
    function that builds one for each benchmark. Without it, the resolver
    builds a pipeline for each benchmark from the adapters that
    `_default_pool` picks. The server checks the pipeline either way, before
    it accepts a benchmark.
    """
    address = server if server is not None else os.environ.get("MANIFOLD_SERVER_URL")
    if address is None:
        raise ValueError("MANIFOLD_SERVER_URL is required")
    host, port = _server_address(address)
    if inspect.isclass(policy):
        _probe_predict(_stateful_predict(policy), signature)
        endpoint: FunctionEndpoint | StatefulEndpoint = StatefulEndpoint(policy, signature)
    else:
        _probe_predict(policy, signature)
        endpoint = FunctionEndpoint(policy, signature)
    serve_endpoint(
        endpoint,
        pipeline=pipeline if pipeline is not None else _resolving_pipeline(signature),
        host=host,
        port=port,
    )


def _resolving_pipeline(signature: PolicySignature) -> Callable[[Benchmark], Pipeline]:
    """Return a function that resolves a pipeline for each benchmark."""

    def pipeline(benchmark: Benchmark) -> Pipeline:
        resolved = resolve(signature, benchmark, _default_pool(signature, benchmark))
        if resolved is None:
            raise ValueError(
                f"the built-in adapters cannot bridge {benchmark.name} to this "
                "policy signature. Pass `pipeline` to choose the adapters yourself."
            )
        return resolved

    return pipeline


__all__ = ["serve"]
