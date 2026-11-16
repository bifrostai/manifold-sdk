"""The start command of a benchmark image.

    python -m manifold.benchmark /opt/manifold/config.json

The command reads the config and imports `hooks.py` from the config's directory. It
asks the policy which cameras it reads, before the simulator starts, and narrows the
spec to those cameras and the replay's. It then builds the Service and runs the
Client. The runner passes the addresses of the task server, the policy and the live
view in environment variables.

An image build runs the command with `--check`. The check reads the config and the
hooks file, writes the spec to `spec.json` beside the config, and exits:

    python -m manifold.benchmark --check /opt/manifold/config.json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from manifold.benchmark._client import Client, request_policy_cameras
from manifold.benchmark._live_view import live_view_publisher
from manifold.benchmark._service import make_service
from manifold.benchmark._worker import WorkerSession
from manifold.benchmark.config import BenchmarkConfig, load_config
from manifold.benchmark.hooks import load_hooks
from manifold.core.names import CameraName

if TYPE_CHECKING:
    from collections.abc import Iterable

# The runner sets these environment variables when it starts the container.
_SERVER_URL_ENV = "MANIFOLD_SERVER_URL"
_TASK_SERVER_ADDRESS_ENV = "MANIFOLD_TASK_SERVER_ADDRESS"
_WORKER_ID_ENV = "MANIFOLD_WORKER_ID"
# The runner sets this variable only when a user watches the live view.
_LIVE_VIEW_SERVER_ADDRESS_ENV = "MANIFOLD_LIVE_VIEW_SERVER_ADDRESS"


def _log_by_level() -> None:
    """Write log records below ERROR to stdout, and ERROR and above to stderr.

    Each handler picks its records with a filter rather than a level, because a
    simulator can set the level of every root handler, as Isaac Lab's `AppLauncher` does.
    """
    below_error = logging.StreamHandler(sys.stdout)
    below_error.addFilter(lambda record: record.levelno < logging.ERROR)
    from_error = logging.StreamHandler(sys.stderr)
    from_error.addFilter(lambda record: record.levelno >= logging.ERROR)
    logging.basicConfig(
        format="%(levelname)s %(name)s: %(message)s", handlers=[below_error, from_error]
    )


def _render_only_policy_cameras(config: BenchmarkConfig, policy_address: str) -> BenchmarkConfig:
    """Return `config` with a spec that publishes only the cameras that the run needs.

    Those are the cameras that the policy reads and the cameras of the replay. The
    spec stays whole when the policy does not answer GET_SIGNATURE. It also stays
    whole when the policy reads a camera that the spec does not publish, because the
    policy's adapter may read that camera from a spec camera with another name.
    """
    # `python -m` names this module `__main__`, which is outside the `manifold`
    # loggers that show INFO.
    logger = logging.getLogger("manifold.benchmark")
    published = [str(sensor.name) for sensor in config.spec.sensors]
    logger.info("the spec publishes the cameras %s", _names(published))
    cameras = request_policy_cameras(policy_address)
    if cameras is None:
        logger.info("the policy sent no signature, so every camera is rendered")
        return config
    logger.info("the policy reads the cameras %s", _names(sorted(cameras)))
    unpublished = sorted(set(cameras) - set(published))
    if unpublished:
        logger.info(
            "the spec does not publish %s, so every camera is rendered", _names(unpublished)
        )
        return config
    kept = {*cameras, *config.replay.cameras}
    spec = config.spec.with_sensors({CameraName(camera) for camera in kept})
    logger.info(
        "rendering %s, and dropping %s",
        _names(str(sensor.name) for sensor in spec.sensors),
        _names(camera for camera in published if camera not in kept),
    )
    return config.model_copy(update={"spec": spec})


def _names(cameras: Iterable[str]) -> str:
    """Return camera names as a list for the log, or "none"."""
    return ", ".join(cameras) or "none"


def main() -> None:
    """Run the benchmark that a config file describes."""
    parser = argparse.ArgumentParser(prog="python -m manifold.benchmark", description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="check the config and the hooks file, write spec.json beside the config, and exit",
    )
    parser.add_argument("config", type=Path, help="the benchmark's config file")
    args = parser.parse_args()

    config = load_config(args.config)
    benchmark = config.spec
    hooks = load_hooks(config, args.config.parent)
    if args.check:
        spec = args.config.parent / "spec.json"
        spec.write_text(benchmark.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(f"checked {args.config} and its hooks file, and wrote {spec}", flush=True)
        return
    # Other packages log each request at INFO, so only the `manifold` loggers show INFO.
    # The hooks write to `manifold.benchmark.hooks`.
    _log_by_level()
    logging.getLogger("manifold").setLevel(logging.INFO)
    policy_address = os.environ[_SERVER_URL_ENV]
    # The Service and the hooks get the narrowed spec.
    config = _render_only_policy_cameras(config, policy_address)
    service = make_service(config, hooks)
    # Some simulators, such as Isaac Sim, end the process when they stop. So the
    # Service stops after the live view and the task server session close.
    try:
        service.start()
        with (
            live_view_publisher(
                config.spec,
                os.environ.get(_LIVE_VIEW_SERVER_ADDRESS_ENV),
                config.replay.cameras[0],
            ) as live_view,
            WorkerSession(
                os.environ[_TASK_SERVER_ADDRESS_ENV],
                worker_id=os.environ[_WORKER_ID_ENV],
            ) as session,
        ):
            Client(config, service, session, policy_address, live_view).run()
    finally:
        service.stop()


if __name__ == "__main__":
    main()
