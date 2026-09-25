"""ROBOTWIN: the 50-task dual-arm SAPIEN manipulation suite (RoboTwin 2.0).

The 50 tasks in `envs/` are bimanual tabletop problems — pick-place, handover,
stacking, articulated opening — built on an annotated object library of 731 instances
across 147 categories. Each task is an executable scene program rather than a
declarative spec: it spawns its objects, scripts an expert, and defines its own
success predicate in Python.

The suite is scored under two conditions that share one task set, selected by config
rather than by task: `demo_clean` (no scene randomization) and `demo_randomized`
(random background textures, ~10 clutter objects, +-3cm table height, random lighting,
and instructions drawn from an unseen template pool). The leaderboard calls these
clean2clean and clean2random and ranks on their mean.

This constant takes the embodiment both configs bind — `embodiment: [aloha-agilex]` —
in absolute joint targets, which is the action space `take_action(action_type="qpos")`
consumes.

Sensors follow the config's own camera section: a fixed scene camera plus one camera on
each wrist, all D435 at 320x240 (`_camera_config.yml`). RoboTwin calls the scene one
`head_camera`; it is declared here as `agentview`, the catalog's name for a benchmark's
primary third-person view. The shared name does not let another suite's policy pair
without an adapter: a camera is found by its name and must then agree on orientation,
channel order, shape, dtype, modality and calibration, and no other suite's `agentview`
is 240x320 -- SIMPLER's is 480x640, Arena's 360x640, LIBERO's square -- so the shape
alone forces a resize. Nor does the name carry the viewpoint. These cameras declare no
calibration, so nothing here records where they point.

RoboTwin can also publish a third-person `observer` view and depth, but both configs
leave them off, so neither is declared here. Every task carries a language instruction,
sampled per episode from the task's own template file.
"""

from __future__ import annotations

from manifold.core.benchmark import Benchmark
from manifold.embodiments.aloha_agilex_joint_absolute import ALOHA_AGILEX_JOINT_ABSOLUTE
from manifold.sensors.cameras import agentview, wrist_left, wrist_right

_SHAPE = (240, 320, 3)

ROBOTWIN = Benchmark(
    name="robotwin",
    embodiment=ALOHA_AGILEX_JOINT_ABSOLUTE,
    sensors=[agentview(_SHAPE), wrist_left(_SHAPE), wrist_right(_SHAPE)],
    instruction=True,
)
