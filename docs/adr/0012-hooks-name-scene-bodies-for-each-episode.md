# 0012 - Hooks name scene bodies for each episode

Status: Proposed

## Context

In RoboCasa365's failure analysis, a reader cannot tell which object or
fixture a failure is about. The Failures tab shows generic task names, such as
"Did not close fxtr" or "Did not place obj in cab". It shows the kitchen's
fixtures by their body names, such as `blender_island_group_1`.

Some tasks pick their objects and fixtures for each episode from a set of
candidates. For example, PickPlaceCounterToCabinet picks a cabinet from the
kitchen's cabinets and an object such as beer or jam. The task's rubric and
the facts that its hooks report always use the same names, such as
`inside(obj, cab)`.

The grader grades the facts that the hooks report, so grading works. To
explain a failure, the grader also matches the rubric's names to the scene's
geometry. The MuJoCo simulator names the scene's bodies by a rule, so it names
the picked cabinet `cab_2_main_group` in one episode and `cab_1_left_group` in
another. The grader cannot match these names to `cab`.

Only the hooks know which candidate the task picked. Only the SDK writes the
scene.

## Decision

The hooks tell the SDK which body or site the task picked for each name in an
episode. `ResetResult.task_names` maps a name to a body (`TaskBody`) or to a
site (`TaskSite`), such as `{"cab": TaskBody("cab_2_main_group_main")}`. The
MuJoCo simulator gives that body the name when it writes the scene.

A picked fixture's parts are the bodies under it in the model's body tree,
such as its doors, shelves and knobs. A part with a name of its own takes the
fixture's name with the rest of that name, so `cab_2_main_group_door` becomes
`cab_door`, and the scene records `cab` as its parent. A picked site, such as a
burner, becomes a scene body that moves with the stove.

`ResetResult.display_names` maps a name to the words that a person reads for
the pick, such as `beer` for `obj`. The scene stores the words as `display`.

## Consequences

- A benchmark that does not pick candidates leaves both fields empty and
  writes the same scene as before.
- Isaac Sim and SAPIEN raise an error for a non-empty `task_names`.
- A fridge shelf has no body and no site, so a hook cannot name it yet.
- The runner must pass `display` on before the grader can read it.
