# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A **minimal, self-contained** MuJoCo sim of the bimanual dual-xArm7 **double-insert** task, plus
tools to **replay a recorded action trajectory** (`.npz`) back through the sim and recover the
per-step IK mocap targets. It is a focused extraction from the RAC project
([`Leo428/RAC`](https://github.com/Leo428/RAC) → `dual_xarms/dual_xarms_sim`); the real-robot /
VR-teleop / data-collection glue has been stripped. The importable package is `dual_xarms_sim`
(name kept from the source to avoid rewriting imports).

## Commands

```bash
uv sync                                            # create .venv, install deps (respects uv.lock)

# Replay a trajectory by re-simulating it (physics; qualitative fidelity)
uv run python -m dual_xarms_sim.replay_insert_data                 # bundled sample episode, headless
uv run python -m dual_xarms_sim.replay_insert_data --max-steps 200 # quick smoke test
# on-screen viewer: on macOS use `mjpython` (NOT python) — see gotchas below
uv run mjpython -m dual_xarms_sim.replay_insert_data /path/ep.npz --render human

# Recover IK mocap targets without physics (bit-exact) and cross-check against the env
uv run python -m dual_xarms_sim.recover_mocap <npz> --verify

# Tests (mocap-recurrence bit-exactness guard)
uv run pytest                                      # all
uv run pytest tests/test_recover_mocap.py::test_recurrence_matches_env_synthetic  # single test
```

### Environment gotchas that break commands

- **Headless GL**: the env always builds a MuJoCo `Renderer`, needing an OpenGL backend. On a
  headless box prefix with `MUJOCO_GL=egl` (or `osmesa`). Without a backend the tests **skip**
  themselves rather than fail.
- **macOS on-screen viewer (`--render human`)**: MuJoCo's passive viewer must run on the main
  thread, so on macOS launch it with **`mjpython`**, not `python`
  (`uv run mjpython -m dual_xarms_sim.replay_insert_data ... --render human`). Plain
  `uv run python ... --render human` errors. `mjpython` ships inside the `mujoco` package, so it is
  already in `.venv/bin/` after `uv sync` — no extra install. Headless (`rgb_array`, the default)
  runs fine under plain `python`.
- **ROS on PYTHONPATH**: if `/opt/ros/...` is on `PYTHONPATH`, its `launch_testing` pytest plugins
  break collection. Prefix with `PYTHONPATH=` to drop them: `PYTHONPATH= MUJOCO_GL=egl uv run pytest`.

## Architecture

### Env layering
`MujocoGymEnv` (`mujoco_gym_env.py`, thin MjModel/MjData + render wrapper)
→ `DoubleInsertDualXarmsGymEnv` (`fix_dual_xarms_sim.py`, the actual task env + mink IK)
→ `RelativeFrame` (`relative_frame.py`, gym `Wrapper`, EE-frame ↔ world-frame action/obs transforms).

Recorded trajectories were collected through the **wrapped** stack, so replay must re-wrap with
`RelativeFrame`. `recover_mocap.py`, by contrast, operates on the **base-env** world-frame action
and needs no wrapper.

### Control model: actions drive IK mocap targets, not joints directly
The 14-D action is `[l_pos Δ(3), l_euler Δ(3), l_grip(1), r_pos Δ(3), r_euler Δ(3), r_grip(1)]`
(dims 0–6 left, 7–13 right; dims 6 & 13 are grippers). `env.step`:
1. clamps each pos/rot delta to per-step velocity limits (`limit_offset_norm`),
2. integrates the delta onto `data.mocap_pos/mocap_quat` (clipped to `*_CARTESIAN_BOUNDS`),
3. `set_ik_targets` runs **mink** IK (`FrameTask` + `PostureTask`, quadprog) toward the mocap pose
   and **steps physics** (`mj_step`) inside that call — so `env.step` advances the sim via IK, and
   the mocap body is the IK target, not a rendered marker.

Gripper ctrl is stored internally as `0..255` but exposed/consumed as `0..1` (`ctrl/255`).

Reward (`_compute_reward`, contact-pair based): `1.0` one peg inserted, `2.0` both pegs inserted +
all objects lifted, `3.0` everything seated on the block (→ `done`). `0.0` otherwise.

### Scene reset is seeded by the *global* `np.random`
`reset()` samples peg/socket XY + yaw with **`np.random`** (not the env's `RandomState`). Reproducing
a scene therefore requires `np.random.seed(seed)` **and** `env.reset(seed=seed)` — see the replay
scripts. The scene (`ufactory_xarm7/insert_scene.xml`) uses the **local** insert geometry (socket
walls `0.09`, small-hand gripper) matching the bundled episode; **do not** swap in the upstream RAC
scene or replay diverges.

## The two replay paths — and why they behave differently

| | `replay_insert_data.py` | `recover_mocap.py` |
|---|---|---|
| method | re-simulates the recorded **relative** actions through physics | pure deterministic **recurrence** over the recorded world action (`og_action`), no physics |
| fidelity | **qualitative** — visually reproduces the rollout; contact-rich insertion is chaotic and **not** bit-reproducible across numpy/scipy/mujoco versions | **bit-exact**, version-robust |

`replay_insert_data.py` printing `WARNING: replay diverged` on the full episode is **expected** —
tiny FP differences compound through contact. Pinning mujoco/mink does not fix it. For an exact check
use `recover_mocap.py --verify`.

**The mocap recurrence is the repo's gold-standard correctness property.** `reconstruct_mocap`
reproduces the env's per-step IK target purely from the world action; `tests/test_recover_mocap.py`
asserts it matches a live env to `< 1e-5` (synthetic case always; full bundled episode when present).
Keep this bit-exact as dependencies move forward — it is the invariant the tests defend.

`replay_abs_action.py` audits a separate absolute/chunk-delta action round-trip used by the dataset
conversion pipeline (documents an off-by-one anchoring bug: `naive` vs `fixed` vs mocap-anchored).

## Conventions & gotchas

- **Quaternion order is inconsistent by layer, on purpose:** MuJoCo (`mocap_quat`, raw sensors) is
  **scalar-first** `[w,x,y,z]`; `_compute_observation` converts obs quats to **scalar-last**
  `[x,y,z,w]` via `np.roll(-1)`, which is what `utils/transformation.py` (default scipy) expects.
  `recover_mocap` stores scalar-first. Always confirm which convention a given array is in.
- **`og_action`**: `RelativeFrame` writes the world-frame (adjoint-transformed) action into
  `obs["state"]["{left,right}/og_action"]`. That is the input `recover_mocap` replays — not the raw
  EE-frame action.
- **Data**: the committed sample is the **slim** episode (state + actions + seed, no camera images,
  ~2 MB) and is the default for replay + tests. The full ~200 MB episode (4 JPEGs/step) is
  git-ignored; regenerate a slim copy with `python scripts/make_slim_npz.py <full.npz> <out_slim.npz>`.
- **Optional teleop**: `oculus_intervention.py` + `utils/network.py` are the upstream RAC teleop
  wrapper, kept for reference; no replay path uses them.
- **Dependency floors**: versions are floored at known-good (`mujoco>=3.3.6`, `mink>=0.0.13`) and
  otherwise resolve to latest via uv. Validated bit-exact on mujoco 3.10 / mink 1.2 / numpy 2.5 /
  scipy 1.18. Original validated pin was `mujoco==3.3.6` / `mink==0.0.13`.
