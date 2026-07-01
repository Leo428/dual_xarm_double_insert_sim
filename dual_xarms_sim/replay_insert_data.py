"""Replay a trajectory collected by collect_insert_data.py inside the sim env.

The collected .npz stores actions already expressed in the wrist/relative frame
(they are info["intervene_action"] after RelativeFrame's inverse transform), plus
the env `seed` used at collection time. Seeding the global np.random with that
seed and replaying the actions through a RelativeFrame-wrapped env reproduces the
exact scene and trajectory.

This script also dumps the IK mocap target AFTER each env.step (taken from
env.unwrapped.data.mocap_pos / mocap_quat) into a sidecar .npz. Indexing matches
the actions / obses arrays: entry k is the mocap target produced by action k.
Quaternions are stored scalar-first (w, x, y, z), matching MuJoCo's convention.

Usage:
    python -m dual_xarms_sim.replay_insert_data <path_to.npz> \\
        [--render human|rgb_array] [--out <out.npz>]
"""

import argparse
import os
from pathlib import Path

import numpy as np
from loop_rate_limiters import RateLimiter
from tqdm import tqdm

from dual_xarms_sim.fix_dual_xarms_sim import DoubleInsertDualXarmsGymEnv
from dual_xarms_sim.relative_frame import RelativeFrame

# Sample episode bundled with the repo (state-only "slim" episode, ~2 MB, committed to git).
DEFAULT_NPZ = Path(__file__).resolve().parent.parent / 'data' / 'sim_dual_xarms_double_insert_20260518_121749_slim.npz'


def _default_out_path(npz_path):
    stem, ext = os.path.splitext(npz_path)
    return f'{stem}_mocap{ext}'


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        'npz_path',
        nargs = '?',
        default = str(DEFAULT_NPZ),
        help = 'Path to a .npz collected by collect_insert_data.py (default: bundled sample episode)',
    )
    parser.add_argument('--render', choices = ['human', 'rgb_array'], default = 'rgb_array')
    parser.add_argument(
        '--max-steps',
        type = int,
        default = None,
        help = 'Replay only the first N steps (quick smoke test); default replays the whole episode',
    )
    parser.add_argument(
        '--out',
        default = None,
        help = 'Output .npz path for the recorded mocap state (default: <input>_mocap.npz)',
    )
    args = parser.parse_args()

    out_path = args.out or _default_out_path(args.npz_path)

    data = np.load(args.npz_path, allow_pickle = True)
    actions = data['actions']
    saved_rews = data['rews']
    if args.max_steps is not None:
        actions = actions[:args.max_steps]
        saved_rews = saved_rews[:args.max_steps]
    if 'seed' not in data.files:
        raise KeyError(
            f"{args.npz_path} has no 'seed' key; it predates seed logging and "
            'cannot be reproduced deterministically.'
        )
    seed = int(data['seed'])
    print(f'Loaded {len(actions)} steps; env seed = {seed}')

    # Reproduce the scene: the env samples peg/socket poses with the global
    # np.random, so seed that; also pass seed= through to env.reset() for Gym.
    np.random.seed(seed)
    env = DoubleInsertDualXarmsGymEnv(
        control_freq = 60, time_limit = 2 * 60, render_mode = args.render
    )
    env = RelativeFrame(env)  # actions were saved in the relative frame

    human_rate = RateLimiter(60, name = 'Replay Rate', warn = False)
    bar = tqdm(total = len(actions), desc = 'Replay')

    n_total = len(actions)
    left_mocap_pos   = np.zeros((n_total, 3), dtype = np.float64)
    left_mocap_quat  = np.zeros((n_total, 4), dtype = np.float64)
    right_mocap_pos  = np.zeros((n_total, 3), dtype = np.float64)
    right_mocap_quat = np.zeros((n_total, 4), dtype = np.float64)

    obs, info = env.reset(seed = seed)
    replay_rews = []
    max_reward = 0.0
    n_steps = 0
    try:
        for t, action in enumerate(actions):
            obs, rew, done, truncated, info = env.step(np.asarray(action))
            # After step, mocap_pos[0/1] holds the left/right IK target the env
            # just integrated onto. Copy so subsequent steps don't overwrite.
            sim_data = env.unwrapped.data
            left_mocap_pos[t]   = sim_data.mocap_pos[0].copy()
            left_mocap_quat[t]  = sim_data.mocap_quat[0].copy()
            right_mocap_pos[t]  = sim_data.mocap_pos[1].copy()
            right_mocap_quat[t] = sim_data.mocap_quat[1].copy()
            replay_rews.append(rew)
            max_reward = max(max_reward, rew)
            n_steps = t + 1
            bar.update(1)
            if args.render == 'human':
                human_rate.sleep()
            if done or truncated:
                print(f'Env ended early at step {t} (done={done}, truncated={truncated})')
                break
    finally:
        env.close()

    # Trim to actual rollout length (in case the episode ended early).
    left_mocap_pos   = left_mocap_pos[:n_steps]
    left_mocap_quat  = left_mocap_quat[:n_steps]
    right_mocap_pos  = right_mocap_pos[:n_steps]
    right_mocap_quat = right_mocap_quat[:n_steps]

    # Reproducibility check: replayed rewards should match the saved ones.
    replay_rews = np.asarray(replay_rews)
    n = len(replay_rews)
    saved = np.asarray(saved_rews[:n], dtype = float)
    max_diff = float(np.abs(replay_rews - saved).max()) if n else float('nan')
    print(f'Replayed {n} steps | max reward = {max_reward}')
    print(f'Max |replay_reward - saved_reward| over {n} steps = {max_diff:.6g}')
    if n == len(actions) and max_diff < 1e-6:
        print('Reproducible: replayed rewards match saved rewards.')
    else:
        print('WARNING: replay diverged from the saved trajectory.')

    np.savez(
        out_path,
        left_mocap_pos   = left_mocap_pos,
        left_mocap_quat  = left_mocap_quat,
        right_mocap_pos  = right_mocap_pos,
        right_mocap_quat = right_mocap_quat,
        actions          = actions[:n_steps],
        seed             = seed,
        source_npz       = args.npz_path,
    )
    print(f'Wrote mocap state for {n_steps} steps to {out_path}')


if __name__ == '__main__':
    main()
