# dual_xarm_double_insert_sim

A **minimal, self-contained** MuJoCo simulation of the bimanual dual-xArm7 **double-insert**
task, plus tools to **replay a recorded action trajectory** (`.npz`) back through the sim and
recover the per-step IK mocap targets.

This is a focused extraction from the RAC project
([`Leo428/RAC`](https://github.com/Leo428/RAC) → `dual_xarms/dual_xarms_sim`). Only the code
needed to build the sim env and replay trajectories is included — the real-robot / VR-teleop glue
(oculus server, streaming HDF5, packing/sockets task variants, VLM annotation, data collection) has
been removed. The upstream `Leo428/RAC` `OculusIntervention` wrapper is kept for optional teleop
(it is not used by any replay path).

## What's here

```
dual_xarms_sim/                     importable package (name kept from the source project)
├── fix_dual_xarms_sim.py           DoubleInsertDualXarmsGymEnv — the MuJoCo dual-xArm env + mink IK
├── mujoco_gym_env.py               MujocoGymEnv base class
├── relative_frame.py               RelativeFrame gym wrapper (EE/body-frame <-> world-frame actions)
├── replay_insert_data.py           >>> replay a recorded .npz trajectory in-sim, dump mocap sidecar
├── recover_mocap.py                reconstruct IK mocap targets from actions (no sim) + --verify
├── replay_abs_action.py            audit the absolute/chunk-delta action round-trip
├── oculus_intervention.py          OculusIntervention teleop wrapper (upstream Leo428/RAC; optional)
├── utils/transformation.py         SE(3) / adjoint helpers used by RelativeFrame
├── utils/network.py                oculus HTTP reader (used only by OculusIntervention)
└── ufactory_xarm7/                 scene assets
    ├── insert_scene.xml            the double-insert scene (deeper 0.09 sockets — local geometry)
    ├── dual_xarm7_small_hand.xml   robot + gripper model (included by insert_scene.xml)
    └── assets/                     18 STL meshes + 2 wood textures referenced by the scene
tests/                              pytest: mocap-recurrence bit-exactness guard
scripts/make_slim_npz.py            strip camera images -> small committable episode
data/
├── …_20260518_121749_slim.npz      sample episode, state-only (~2 MB, committed; default for replay + tests)
└── …_20260518_121749.npz           full episode incl. 4 JPEG cameras/step (~200 MB, git-ignored)
```

## Setup (uv)

```bash
cd dual_xarm_double_insert_sim
uv sync            # creates .venv and installs latest-compatible deps (MuJoCo, mink, gymnasium, ...)
```

## Two ways to "replay the actions"

| | `replay_insert_data.py` | `recover_mocap.py` |
|---|---|---|
| what it does | re-simulates the recorded **relative** actions through the physics env | reconstructs the IK mocap targets by **pure recurrence** over the recorded world actions (`og_action`) |
| uses physics? | yes (contact-rich) | no (`--verify` optionally cross-checks against the env) |
| fidelity | **qualitative** — reproduces the rollout visually, but contact-rich outcomes (insertion success) are chaotic and **not** bit-reproducible across numpy/scipy/mujoco versions | **bit-exact** and version-robust (verified 0.0 m / 3e-8 rad on mujoco 3.10) |
| use it for | watching/visualising a rollout, sanity checks | recovering ground-truth mocap targets for dataset conversion |

## Replay a trajectory (re-simulate in the env)

Reproduces the scene by seeding `np.random` with the episode's stored `seed`, then steps the
recorded (relative-frame) actions through a `RelativeFrame`-wrapped env. After each `env.step` it
records the left/right IK mocap targets (`data.mocap_pos/quat`, quats scalar-first wxyz) and writes
them to a sidecar `<input>_mocap.npz`.

> **Fidelity note:** this path prints `max |replay_reward - saved_reward|` and will report
> `WARNING: replay diverged` on the full episode. That is **expected** — insertion is contact-rich,
> so tiny floating-point differences (numpy/scipy/mujoco) compound into a different contact outcome.
> Pinning mujoco/mink to the original versions does **not** make it bit-reproducible. For an exact,
> version-robust check use `recover_mocap.py --verify` below.

```bash
# zero-arg: replays the bundled sample episode headless (rgb_array)
uv run python -m dual_xarms_sim.replay_insert_data

# quick smoke test — first 200 steps only
uv run python -m dual_xarms_sim.replay_insert_data --max-steps 200

# a specific file, with the on-screen viewer
uv run python -m dual_xarms_sim.replay_insert_data /path/to/episode.npz --render human
```

The script prints `max |replay_reward - saved_reward|` as a reproducibility check — it should be
~0 for a faithful replay.

### Headless rendering

`env` renders camera images each step, which needs an OpenGL backend. On a headless machine set:

```bash
MUJOCO_GL=egl uv run python -m dual_xarms_sim.replay_insert_data --max-steps 200
```

(use `osmesa` if EGL is unavailable).

## Recover mocap without the sim

`recover_mocap.py` reconstructs the same mocap targets purely from the recorded global actions
(a deterministic recurrence — no physics), and `--verify` cross-checks that recurrence against a
live env replay (expected bit-exact):

```bash
uv run python -m dual_xarms_sim.recover_mocap data/sim_dual_xarms_double_insert_20260518_121749.npz --verify
```

## Tests

```bash
uv run pytest
```

`tests/test_recover_mocap.py` guards the bit-exact property: it steps a global-action sequence
through the live env and asserts the pure `reconstruct_mocap` recurrence matches to < 1e-5 (a
self-contained synthetic case, plus the full bundled episode when present). The env needs an
OpenGL backend — set `MUJOCO_GL=egl` (or `osmesa`) if the default fails; the test skips itself if
no backend is available.

> If you have ROS on your `PYTHONPATH` (e.g. `/opt/ros/...`), its `launch_testing` pytest plugins
> break collection. Prefix with `PYTHONPATH=` to drop them: `PYTHONPATH= MUJOCO_GL=egl uv run pytest`.

## Notes

- The importable package is `dual_xarms_sim` (kept from the source to avoid rewriting imports).
- The committed sample is the **slim** episode (state + actions + seed, no camera images) — enough
  for both replay tools. The full 200 MB episode (4 JPEG cameras/step) is git-ignored; regenerate a
  slim copy from any full episode with `python scripts/make_slim_npz.py <full.npz> <out_slim.npz>`.
- The scene uses the **local** insert geometry (socket walls sized `0.09`, small-hand gripper),
  which is what the bundled episode was collected against — do not swap in the upstream
  `Leo428/RAC` scene or the replay will diverge.
- Validated on the **latest** stack (mujoco 3.10, mink 1.2, numpy 2.5, scipy 1.18): the
  `recover_mocap --verify` recurrence check is bit-exact (0.0 m / 3e-8 rad). Dependency versions are
  floored at known-good values (`mujoco>=3.3.6`, `mink>=0.0.13`) and otherwise resolve to the latest
  via uv. The original validated versions were `mujoco==3.3.6` / `mink==0.0.13` if you ever need to
  reproduce that exact stack (`uv sync` respects the committed `uv.lock`).
