"""Audit + replay the absolute-action round-trip for collected npz data.

Pipeline being audited
----------------------
collect_insert_data.py -> npz: per-step EE-frame delta `actions`, and the
    GLOBAL-frame delta stored in obs["state"]["{l,r}/og_action"].
data_format.py         -> HDF5: actions/global_action[k] = concat(og_action of obses[k]);
    obses/state/.../tcp_pose[k] = tcp of obses[k].
sim_bimanual_assembly_config.py (_compose_arm_action / _compose_gripper_action)
    -> RLDS `action` (absolute) = compose(tcp_pose[k], global_action[k]).
openpi DeltaActions     -> chunk-relative delta (re-anchors to the current state).
_abs_action_to_env_delta-> converts a policy absolute action back to a base-env
    global step-wise delta for eval.

What this script checks
-----------------------
Test B  : _abs_action_to_env_delta exactly inverts the builder compose functions
          when given the SAME state.
Replay GT    : feed the recorded global delta straight to the base env
               (ground-truth / determinism baseline).
Replay NAIVE : build abs_action exactly as the pipeline does -- compose with the
               same-index (post-action) tcp -- then feed back through
               _abs_action_to_env_delta.
Replay FIXED : build abs_action composed with the PRE-action tcp (obses[k-1])
               then feed back.

Usage:
    python -m dual_xarms_sim.replay_abs_action <npz> [--render human|rgb_array]
    add --mode {gt,naive,fixed,chaos,all}  (naive = WITH the bug, fixed = bug fixed)
"""

import argparse
import copy

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter
from tqdm import tqdm

from dual_xarms_sim.fix_dual_xarms_sim import DoubleInsertDualXarmsGymEnv
from dual_xarms_sim.recover_mocap import reconstruct_mocap


# --- builder conversion (verbatim from sim_bimanual_assembly_config.py) -------
def _compose_arm_action(tcp_pose, global_arm_action):
    state_R = R.from_quat(tcp_pose[3:7])
    delta_R = R.from_euler("xyz", global_arm_action[3:6])
    composed_rpy = (delta_R * state_R).as_euler("xyz").astype(np.float32)
    composed_xyz = (tcp_pose[:3] + global_arm_action[:3]).astype(np.float32)
    return np.concatenate([composed_xyz, composed_rpy]).astype(np.float32)


def _compose_gripper_action(gripper_state, gripper_global_action):
    return np.float32(1.0 - (gripper_state + 0.1 * gripper_global_action))


def build_abs_action(state, global_action):
    """Reproduce sim_bimanual_assembly_config.parse_episode's 14-D `action`."""
    left_arm = _compose_arm_action(state["left/tcp_pose"], global_action[0:6])
    right_arm = _compose_arm_action(state["right/tcp_pose"], global_action[7:13])
    left_grip = _compose_gripper_action(float(state["left/gripper_pos"][0]), global_action[6])
    right_grip = _compose_gripper_action(float(state["right/gripper_pos"][0]), global_action[13])
    return np.concatenate([left_arm, [left_grip], right_arm, [right_grip]]).astype(np.float32)


# --- eval conversion (verbatim from the user's _abs_action_to_env_delta) ------
def _abs_action_to_env_delta(abs_action: np.ndarray, obs: dict) -> np.ndarray:
    left_pose = np.asarray(obs["state"]["left/tcp_pose"], dtype=np.float32)
    right_pose = np.asarray(obs["state"]["right/tcp_pose"], dtype=np.float32)
    cur_lg = float(obs["state"]["left/gripper_pos"][0])
    cur_rg = float(obs["state"]["right/gripper_pos"][0])

    lpos_delta = abs_action[0:3].astype(np.float32) - left_pose[:3]
    r_abs_l = Rotation.from_euler("xyz", abs_action[3:6])
    r_state_l = Rotation.from_quat(left_pose[3:7])
    lrpy_delta = (r_abs_l * r_state_l.inv()).as_euler("xyz").astype(np.float32)
    lgrip_env = np.float32((1.0 - cur_lg - float(abs_action[6])) / 0.1)

    rpos_delta = abs_action[7:10].astype(np.float32) - right_pose[:3]
    r_abs_r = Rotation.from_euler("xyz", abs_action[10:13])
    r_state_r = Rotation.from_quat(right_pose[3:7])
    rrpy_delta = (r_abs_r * r_state_r.inv()).as_euler("xyz").astype(np.float32)
    rgrip_env = np.float32((1.0 - cur_rg - float(abs_action[13])) / 0.1)

    return np.concatenate([
        lpos_delta, lrpy_delta, [lgrip_env],
        rpos_delta, rrpy_delta, [rgrip_env],
    ]).astype(np.float32)


# --- mocap-anchored eval conversion (the proper fix for finding O1) ------------
def _abs_mocap_to_env_delta(abs_mocap, live_mocap, obs):
    """Convert a mocap-anchored absolute action to a base-env delta.

    abs_mocap : 14-D [l_pos(3), l_euler(3), l_grip(1), r_pos(3), r_euler(3), r_grip(1)]
    live_mocap: (l_pos, l_quat, r_pos, r_quat) read from env._data.mocap_* (quats w-first)

    env.step integrates the delta onto mocap_pos/mocap_quat, so anchoring on the
    live mocap (not tcp) makes `mocap += delta` land exactly on abs_mocap.
    """
    l_pos, l_quat, r_pos, r_quat = live_mocap
    lpos_d = abs_mocap[0:3] - np.asarray(l_pos, dtype=np.float64)
    lrot_d = (R.from_euler("xyz", abs_mocap[3:6])
              * R.from_quat(l_quat, scalar_first=True).inv()).as_euler("xyz")
    lgrip = (abs_mocap[6] - float(obs["state"]["left/gripper_pos"][0])) / 0.1
    rpos_d = abs_mocap[7:10] - np.asarray(r_pos, dtype=np.float64)
    rrot_d = (R.from_euler("xyz", abs_mocap[10:13])
              * R.from_quat(r_quat, scalar_first=True).inv()).as_euler("xyz")
    rgrip = (abs_mocap[13] - float(obs["state"]["right/gripper_pos"][0])) / 0.1
    return np.concatenate([lpos_d, lrot_d, [lgrip], rpos_d, rrot_d, [rgrip]]).astype(np.float32)


def rot_angle(euler_a, euler_b):
    """Geodesic angle (rad) between two xyz-euler rotations."""
    ra = R.from_euler("xyz", euler_a)
    rb = R.from_euler("xyz", euler_b)
    return (ra * rb.inv()).magnitude()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_path")
    parser.add_argument("--render", choices=["human", "rgb_array"], default="rgb_array")
    parser.add_argument("--mode", choices=["all", "gt", "naive", "fixed", "mocap", "chaos"],
                        default="all",
                        help="which replay; naive=WITH bug, fixed=bug fixed, mocap=mocap-anchored")
    args = parser.parse_args()

    data = np.load(args.npz_path, allow_pickle=True)
    if "seed" not in data.files:
        raise KeyError(f"{args.npz_path} has no 'seed'; cannot reproduce deterministically.")
    seed = int(data["seed"])
    obses = data["obses"]
    rews = np.asarray(data["rews"], dtype=float)
    N = len(obses)
    print(f"Loaded {N} steps, env seed = {seed}, recorded max reward = {rews.max()}")

    # global_action[k] = concat(og_action) of obses[k]  -- exactly data_format.py
    states = [obses[k]["state"] for k in range(N)]
    global_action = np.stack([
        np.concatenate([s["left/og_action"], s["right/og_action"]]).astype(np.float32)
        for s in states
    ])
    rec_left_tcp = np.stack([s["left/tcp_pose"] for s in states])
    rec_right_tcp = np.stack([s["right/tcp_pose"] for s in states])

    # ---- Test B: is _abs_action_to_env_delta the exact inverse of the builder?
    pos_err = rot_err = grip_err = 0.0
    for k in range(N):
        g = global_action[k]
        abs_a = build_abs_action(states[k], g)
        rec = _abs_action_to_env_delta(abs_a, {"state": states[k]})
        pos_err = max(pos_err, np.abs(rec[[0, 1, 2, 7, 8, 9]] - g[[0, 1, 2, 7, 8, 9]]).max())
        rot_err = max(rot_err, rot_angle(rec[3:6], g[3:6]), rot_angle(rec[10:13], g[10:13]))
        grip_err = max(grip_err, abs(rec[6] - g[6]), abs(rec[13] - g[13]))
    print("\n[Test B] round-trip identity (same state for build + invert):")
    print(f"  max |pos delta err| = {pos_err:.3e} m   max rot err = {rot_err:.3e} rad"
          f"   max grip err = {grip_err:.3e}")
    print("  -> PASS (functions are mutual inverses)" if max(pos_err, rot_err, grip_err) < 1e-4
          else "  -> FAIL")

    # pre-action states: state seen *before* action k (reset state for k=0)
    env = DoubleInsertDualXarmsGymEnv(control_freq=60, time_limit=2 * 60, render_mode=args.render)
    np.random.seed(seed)
    reset_obs, _ = env.reset(seed=seed)
    pre_states = [copy.deepcopy(reset_obs["state"])] + states[:-1]

    abs_naive = np.stack([build_abs_action(states[k], global_action[k]) for k in range(N)])
    abs_fixed = np.stack([build_abs_action(pre_states[k], global_action[k]) for k in range(N)])

    # mocap-anchored absolute action: recovered mocap target pose + recorded gripper
    mocap = reconstruct_mocap(global_action.astype(np.float64))
    abs_mocap = np.zeros((N, 14))
    for k in range(N):
        le = R.from_quat(mocap["left/mocap_quat"][k], scalar_first=True).as_euler("xyz")
        re = R.from_quat(mocap["right/mocap_quat"][k], scalar_first=True).as_euler("xyz")
        abs_mocap[k] = np.concatenate([
            mocap["left/mocap_pos"][k], le, [states[k]["left/gripper_pos"][0]],
            mocap["right/mocap_pos"][k], re, [states[k]["right/gripper_pos"][0]],
        ])

    human_rate = RateLimiter(60, name="Replay", warn=False) if args.render == "human" else None

    def run_replay(name, action_fn):
        np.random.seed(seed)
        obs, _ = env.reset(seed=seed)
        tcp_err, rep_rews = [], []
        ended = None
        for k in tqdm(range(N), desc=name, leave=False):
            delta = np.asarray(action_fn(k, obs), dtype=np.float32)
            obs, rew, done, trunc, _ = env.step(delta)
            le = np.linalg.norm(obs["state"]["left/tcp_pose"][:3] - rec_left_tcp[k][:3])
            re = np.linalg.norm(obs["state"]["right/tcp_pose"][:3] - rec_right_tcp[k][:3])
            tcp_err.append(le + re)
            rep_rews.append(rew)
            if (done or trunc) and ended is None:
                ended = k
            if human_rate is not None:
                human_rate.sleep()
        tcp_err = np.asarray(tcp_err)
        rep_rews = np.asarray(rep_rews)
        rew_match = np.mean(rep_rews == rews[:N]) if N else float("nan")
        print(f"\n[{name}]  tcp pos err vs recorded:  mean={tcp_err.mean():.4e}  "
              f"max={tcp_err.max():.4e} m")
        print(f"          replay max reward={rep_rews.max()}  "
              f"reward-step match={rew_match*100:.1f}%"
              + (f"  (env done/trunc at step {ended})" if ended is not None else ""))
        return tcp_err

    print("\n--- Replays (base env, no wrappers) ---")
    rng = np.random.RandomState(0)
    replays = {
        "gt": ("Replay GT    (recorded global delta -- exact reference)",
               lambda k, o: global_action[k]),
        "naive": ("Replay NAIVE (pipeline's abs_action -- WITH the off-by-one bug)",
                  lambda k, o: _abs_action_to_env_delta(abs_naive[k], o)),
        "fixed": ("Replay FIXED (off-by-one corrected, but still tcp-anchored)",
                  lambda k, o: _abs_action_to_env_delta(abs_fixed[k], o)),
        "mocap": ("Replay MOCAP (mocap-anchored abs_action, recovered -- the proper fix)",
                  lambda k, o: _abs_mocap_to_env_delta(
                      abs_mocap[k],
                      (env._data.mocap_pos[0].copy(), env._data.mocap_quat[0].copy(),
                       env._data.mocap_pos[1].copy(), env._data.mocap_quat[1].copy()),
                      o)),
        # chaos probe: GT + 1e-7 noise; isolates numerical drift from real bugs.
        "chaos": ("Replay GT+1e-7 noise (chaos probe)",
                  lambda k, o: global_action[k] + rng.normal(0, 1e-7, 14).astype(np.float32)),
    }
    selected = (["gt", "naive", "fixed", "mocap", "chaos"]
                if args.mode == "all" else [args.mode])
    for key in selected:
        run_replay(*replays[key])
    env.close()


if __name__ == "__main__":
    main()
