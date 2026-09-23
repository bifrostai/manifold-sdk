"""ROBOTWIN: the 50-task dual-arm SAPIEN manipulation suite (RoboTwin 2.0).

The 50 tasks in `envs/` are bimanual tabletop problems, such as pick-place, handover,
stacking, and opening articulated objects. They are built on an annotated object
library of 731 instances across 147 categories. Each task is a scene program written in
Python. The program spawns the objects, scripts an expert, and defines the success
condition for that task.

The suite is scored under two conditions that share one task set, selected by config
rather than by task: `demo_clean` (no scene randomization) and `demo_randomized`
(random background textures, ~10 clutter objects, +-3cm table height, random lighting,
and instructions drawn from an unseen template pool). The leaderboard calls these
clean2clean and clean2random and ranks on their mean.

Both configs set `embodiment: [aloha-agilex]`. This constant declares that embodiment
in absolute joint targets, which is the action space that
`take_action(action_type="qpos")` consumes.

The cameras follow the camera section of the config (`_camera_config.yml`): three
scene views, and one camera on each wrist. All five are D435 at 320x240.

The three scene views are not interchangeable, so each one is declared under its own
name.

`head_camera` is the close working view. It frames the arms' reach at 37 degrees, and
is declared here as `head`. `observer_camera` is a 93-degree establishing shot of the
whole scene. It is declared as `agentview`, which is the catalog's name for the primary
third-person view of a benchmark. Both configs must set `third_view: true` before
`observer_camera` is rendered. `front_camera` sits at table height and looks across the
surface. It is declared as `front`.

A policy from another suite cannot pair on a shared name alone. A camera is found by
its name, and it must then agree on orientation, channel order, shape, dtype, modality
and calibration. No other suite declares an `agentview` of 240x320: SIMPLER uses
480x640, Arena uses 360x640, and LIBERO uses a square shape. The shape alone therefore
forces a resize. These cameras declare no calibration, so nothing here records the
direction they point in.

RoboTwin can also publish depth, but both configs leave it off, so it is not declared
here. Every task carries a language instruction,
sampled per episode from the task's own template file.
"""

from __future__ import annotations

from manifold.core.benchmark import Benchmark
from manifold.embodiments.aloha_agilex_joint_absolute import ALOHA_AGILEX_JOINT_ABSOLUTE
from manifold.sensors.cameras import agentview, front, head, wrist_left, wrist_right

_SHAPE = (240, 320, 3)

ROBOTWIN = Benchmark(
    name="robotwin",
    embodiment=ALOHA_AGILEX_JOINT_ABSOLUTE,
    sensors=[
        head(_SHAPE),
        agentview(_SHAPE),
        front(_SHAPE),
        wrist_left(_SHAPE),
        wrist_right(_SHAPE),
    ],
    instruction=True,
)
