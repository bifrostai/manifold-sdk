# 0012 - Hooks give a picked body the task's name

Status: Pending

## Context

Failure analysis matches the names that a task's facts use, such as `cab` in
`inside(obj, cab)`, to the bodies of the replay's scene. It needs the match to tell a
touch of the cabinet that a goal opens from a bump into another fixture.

Some tasks pick their fixtures and objects for each episode from a set of
candidates. RoboCasa365's OpenCabinet picks one of a kitchen's cabinets and keeps it
in its `fxtr` attribute. The MuJoCo simulator labels a body with its model's name, so
the picked cabinet is `singlecabinet_3_right_group_1` in one episode and
`cab_1_left_group` in another, and nothing in the scene says which body `fxtr` is.

A fixture's parts do not have the fixture's name either. RoboCasa builds a cabinet's
door and handle as models of their own, which the simulator labels apart from the
cabinet. The simulator labels each body with a joint outside the robot as an object,
so a hinged or sliding part, such as a kettle's switch, becomes an object without a
name. Failure analysis cannot match a touch of the handle to the cabinet, and a press
of the switch is not recorded.

Only the hooks read which candidate the task picked, and only the SDK writes the
scene.

## Decision

For each name that a task's facts use, the hooks return the body that the task picked
in the episode. `ResetResult.task_bodies` maps that name to the simulator's name for
the body, such as `{"cab": "cab_2_main_group_main"}`.

The MuJoCo simulator gives that body the task's name, and each body under it in the
model's body tree too, so that a cabinet's doors, handles and hinged parts all have
the name `cab`. Each part of a named fixture is a fixture, a hinged one too. The
bodies of a named object keep their roles. When the model has no body that
`task_bodies` maps a name to, the scene keeps its labels, with a warning, so that a
wrong name loses only the replay's labels and not the episode.

Isaac Sim and SAPIEN ignore `task_bodies`. No benchmark on them needs it.

## Consequences

- A benchmark that does not pick candidates leaves `task_bodies` empty, and the SDK
  writes its scene unchanged.
- The replay records a touch of any part of a named fixture under the fixture's name,
  so it does not say which part the robot touched.
- A site, such as a burner on a stove, has no body, so `task_bodies` cannot map a
  name to it. A benchmark whose facts use a site's name gives the grader the fixture
  that holds the site some other way.

## Alternatives

- Name each part apart, such as `cab_door_handle`, and record the fixture as its
  parent. That keeps the part, but adds a key to the scene that the runner and the
  grader must carry, for a distinction that no grading rule uses.
- Record a named site as a body of its own. RoboCasa's burner is a disc a
  centimetre thick, half sunk into the stove's top, so the grader's rest test finds
  a pan on the stove rather than on the burner.
