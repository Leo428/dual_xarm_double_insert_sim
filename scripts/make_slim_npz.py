"""Make a slim, git-committable copy of a collected episode .npz.

The full episode is ~200 MB because each per-step obs stores four JPEG-encoded camera images
(left/right x top/wrist, ~60 KB/step). Neither replay tool needs those:

  * replay_insert_data.py re-simulates and reads only actions / rews / seed;
  * recover_mocap.py reads obses[*]/state/{left,right}/og_action + seed.

This drops the `images` entry from every obs (keeping the full `state`) and re-saves compressed,
shrinking the file to ~1 MB while keeping both tools working unchanged.

Usage:
    python scripts/make_slim_npz.py <full.npz> <out_slim.npz>
"""

import argparse

import numpy as np


def slim_obs(obs):
    """Return the obs dict without the bulky JPEG `images` blob (everything else kept)."""
    return {k: v for k, v in obs.items() if k != "images"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="full episode .npz")
    ap.add_argument("out", help="output slim .npz")
    args = ap.parse_args()

    z = np.load(args.src, allow_pickle=True)
    obses_slim = np.array([slim_obs(o) for o in z["obses"]], dtype=object)

    np.savez_compressed(
        args.out,
        obses=obses_slim,
        actions=z["actions"],
        rews=z["rews"],
        dones=z["dones"],
        truncateds=z["truncateds"],
        infos=z["infos"],
        seed=z["seed"],
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
