"""Reconstruct the per-step mocap (IK target) EE poses for collected npz episodes.

collect_insert_data.py does NOT record the mocap target. It is, however, a pure
deterministic recurrence over the recorded global action
(obs["state"]["{l,r}/og_action"]), so it can be recovered post-hoc with no physics:

    reset:   mocap = LEFT_HOME / RIGHT_HOME
    step k:  d_pos  = limit_offset_norm(og_action[:3],  MAX_LINEAR_VELOCITY)
             d_eul  = limit_offset_norm(og_action[3:6], MAX_ANGULAR_VELOCITY)
             mocap_pos  = clip(mocap_pos + d_pos, CARTESIAN_BOUNDS)
             mocap_quat = R.from_euler("xyz", d_eul) * mocap_quat      (no clip)

mocap_target[k] is the target AFTER step k, i.e. it aligns with obses[k] / tcp_pose[k].
Quaternions are scalar-first [w, x, y, z] (mujoco convention).

Usage:
    python -m dual_xarms_sim.recover_mocap <npz> [--verify]

`--verify` replays the recorded global action through the base env (bit-exact, see
replay_abs_action.py) and checks the recurrence against env._data.mocap_pos/quat.
"""

import argparse

import numpy as np
from scipy.spatial.transform import Rotation as R

from dual_xarms_sim.fix_dual_xarms_sim import (
    DoubleInsertDualXarmsGymEnv,
    LEFT_HOME, RIGHT_HOME,
    LEFT_CARTESIAN_BOUNDS, RIGHT_CARTESIAN_BOUNDS,
    _MAX_LINEAR_VELOCITY, _MAX_ANGULAR_VELOCITY,
)

CONTROL_FREQ = 60  # collect_insert_data.py uses control_freq=60
MAX_LIN = _MAX_LINEAR_VELOCITY / CONTROL_FREQ
MAX_ANG = _MAX_ANGULAR_VELOCITY / CONTROL_FREQ


def limit_offset_norm(offset, max_offset):
    """Verbatim from fix_dual_xarms_sim.DoubleInsertDualXarmsGymEnv.limit_offset_norm."""
    norm = np.linalg.norm(offset)
    if norm > max_offset:
        offset = offset / norm * max_offset
    return offset


def global_actions_from_npz(npz):
    """global_action[k] = concat(og_action of obses[k]) -- exactly data_format.py."""
    obses = npz["obses"]
    return np.stack([
        np.concatenate([o["state"]["left/og_action"], o["state"]["right/og_action"]])
        for o in obses
    ]).astype(np.float64)


def reconstruct_mocap(global_action):
    """Replay the env's mocap recurrence. Returns dict of (N,3)/(N,4) arrays."""
    n = len(global_action)
    out = {k: np.zeros((n, d)) for k, d in
           [("left/mocap_pos", 3), ("left/mocap_quat", 4),
            ("right/mocap_pos", 3), ("right/mocap_quat", 4)]}
    l_pos, l_quat = LEFT_HOME[:3].copy(), LEFT_HOME[3:].copy()
    r_pos, r_quat = RIGHT_HOME[:3].copy(), RIGHT_HOME[3:].copy()
    for k in range(n):
        g = global_action[k]
        l_pos = np.clip(l_pos + limit_offset_norm(g[0:3], MAX_LIN),
                        LEFT_CARTESIAN_BOUNDS[0], LEFT_CARTESIAN_BOUNDS[1])
        l_quat = (R.from_euler("xyz", limit_offset_norm(g[3:6], MAX_ANG))
                  * R.from_quat(l_quat, scalar_first=True)).as_quat(scalar_first=True)
        r_pos = np.clip(r_pos + limit_offset_norm(g[7:10], MAX_LIN),
                        RIGHT_CARTESIAN_BOUNDS[0], RIGHT_CARTESIAN_BOUNDS[1])
        r_quat = (R.from_euler("xyz", limit_offset_norm(g[10:13], MAX_ANG))
                  * R.from_quat(r_quat, scalar_first=True)).as_quat(scalar_first=True)
        out["left/mocap_pos"][k], out["left/mocap_quat"][k] = l_pos, l_quat
        out["right/mocap_pos"][k], out["right/mocap_quat"][k] = r_pos, r_quat
    return out


def quat_angle(qa, qb):
    """Geodesic angle (rad) between scalar-first quats; sign-insensitive."""
    ra = R.from_quat(qa, scalar_first=True)
    rb = R.from_quat(qb, scalar_first=True)
    return (ra * rb.inv()).magnitude()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_path")
    parser.add_argument("--verify", action="store_true",
                        help="cross-check the recurrence against the env's mocap")
    args = parser.parse_args()

    npz = np.load(args.npz_path, allow_pickle=True)
    g = global_actions_from_npz(npz)
    n = len(g)
    mocap = reconstruct_mocap(g)
    print(f"Reconstructed mocap targets for {n} steps (no sim).")
    print(f"  left/mocap_pos[0]   = {np.round(mocap['left/mocap_pos'][0], 5)}")
    print(f"  left/mocap_pos[-1]  = {np.round(mocap['left/mocap_pos'][-1], 5)}")

    if not args.verify:
        return

    seed = int(npz["seed"])
    env = DoubleInsertDualXarmsGymEnv(control_freq=60, time_limit=2 * 60,
                                      render_mode="rgb_array")
    np.random.seed(seed)
    env.reset(seed=seed)
    pos_err = quat_err = 0.0
    for k in range(n):
        env.step(g[k].astype(np.float32))
        for side in ["left", "right"]:
            idx = 0 if side == "left" else 1
            pe = np.linalg.norm(env._data.mocap_pos[idx] - mocap[f"{side}/mocap_pos"][k])
            qe = quat_angle(env._data.mocap_quat[idx], mocap[f"{side}/mocap_quat"][k])
            pos_err = max(pos_err, pe)
            quat_err = max(quat_err, qe)
    env.close()
    print(f"\n[verify] recurrence vs env mocap over {n} steps:")
    print(f"  max pos err = {pos_err:.3e} m   max quat err = {quat_err:.3e} rad")
    print("  -> RECOVERABLE: recurrence matches the env mocap"
          if max(pos_err, quat_err) < 1e-5 else "  -> MISMATCH")


if __name__ == "__main__":
    main()
