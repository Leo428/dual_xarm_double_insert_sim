"""Bit-exactness guard for the mocap recurrence.

``recover_mocap.reconstruct_mocap`` reproduces the env's per-step IK mocap target as a pure
deterministic recurrence over the world-frame (global) action. That equivalence is the actual
"gold-standard" correctness property of this repo (see README), and it must stay bit-exact as
dependencies (mujoco / mink / numpy / scipy) move forward.

Two tests:
  * ``test_recurrence_matches_env_synthetic`` — self-contained, no data file needed; feeds a short
    synthetic global-action sequence to the live env and to the recurrence and compares.
  * ``test_verify_on_bundled_episode`` — the real 3247-step sample episode; skipped if absent
    (it is git-ignored, ~200 MB).

Both need an OpenGL backend for the env's Renderer; conftest sets ``MUJOCO_GL=egl``. If no backend
is available the env cannot be built and the test is skipped rather than failing.
"""

from pathlib import Path

import numpy as np
import pytest

from dual_xarms_sim.recover_mocap import global_actions_from_npz
from dual_xarms_sim.recover_mocap import quat_angle
from dual_xarms_sim.recover_mocap import reconstruct_mocap

TOL_POS = 1e-5  # metres
TOL_ANG = 1e-5  # radians

_SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sim_dual_xarms_double_insert_20260518_121749_slim.npz"


def _make_env():
    """Build the base env, skipping the test if no GL backend can initialise the Renderer."""
    from dual_xarms_sim.fix_dual_xarms_sim import DoubleInsertDualXarmsGymEnv

    try:
        return DoubleInsertDualXarmsGymEnv(control_freq=60, time_limit=2 * 60, render_mode="rgb_array")
    except Exception as exc:  # e.g. no EGL/OSMesa on a headless box
        pytest.skip(f"sim env could not initialise a renderer (no GL backend?): {exc}")


def _max_errs(env, global_action, mocap):
    """Step ``global_action`` through ``env`` and return (max_pos_err, max_quat_err) vs ``mocap``."""
    pos_err = ang_err = 0.0
    for k in range(len(global_action)):
        env.step(global_action[k].astype(np.float32))
        for side, idx in (("left", 0), ("right", 1)):
            pe = np.linalg.norm(env._data.mocap_pos[idx] - mocap[f"{side}/mocap_pos"][k])
            ae = quat_angle(env._data.mocap_quat[idx], mocap[f"{side}/mocap_quat"][k])
            pos_err = max(pos_err, pe)
            ang_err = max(ang_err, ae)
    return pos_err, ang_err


def test_recurrence_matches_env_synthetic():
    """The pure recurrence must equal the live env's mocap targets on a synthetic action seq."""
    rng = np.random.RandomState(0)
    # per arm [dpos(3), deul(3), grip(1)] x 2 = 14; magnitudes span the env's velocity clamp.
    global_action = rng.uniform(-0.02, 0.02, size=(100, 14)).astype(np.float64)
    mocap = reconstruct_mocap(global_action)

    env = _make_env()
    try:
        np.random.seed(0)
        env.reset(seed=0)
        pos_err, ang_err = _max_errs(env, global_action, mocap)
    finally:
        env.close()

    assert pos_err < TOL_POS, f"max pos err {pos_err:.3e} m exceeds {TOL_POS}"
    assert ang_err < TOL_ANG, f"max quat err {ang_err:.3e} rad exceeds {TOL_ANG}"


@pytest.mark.skipif(not _SAMPLE.exists(), reason="bundled sample episode not present (git-ignored)")
def test_verify_on_bundled_episode():
    """Full 3247-step real episode: recurrence must match the env bit-exactly."""
    npz = np.load(_SAMPLE, allow_pickle=True)
    global_action = global_actions_from_npz(npz)
    mocap = reconstruct_mocap(global_action)

    env = _make_env()
    try:
        seed = int(npz["seed"])
        np.random.seed(seed)
        env.reset(seed=seed)
        pos_err, ang_err = _max_errs(env, global_action, mocap)
    finally:
        env.close()

    assert pos_err < TOL_POS, f"max pos err {pos_err:.3e} m exceeds {TOL_POS}"
    assert ang_err < TOL_ANG, f"max quat err {ang_err:.3e} rad exceeds {TOL_ANG}"
