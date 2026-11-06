"""
Filters a robomimic-format image.hdf5 (output of dataset_states_to_obs.py,
vision-based observations included) down to the "best" demos based on:

  1. No object drop during/after lifting.
  2. Gripper is actuated (opened/closed) FEWER than MAX_ACTUATIONS times
     (i.e. a clean, decisive grasp rather than repeated fumbling/regrasping).

Does NOT use jerk/smoothness — that metric is intentionally excluded here.

The script reads only low-dim keys (cube position, actions) to compute
metrics -- it does NOT decode/process the RGB image data, just copies it
through unchanged for demos that pass the filter. This keeps it fast even
on a large image dataset.

Usage:
    python extract_expert_data.py \
        --dataset image.hdf5 \
        --output image_filtered.hdf5 \
        --table_height 0.8 \
        --lift_threshold 0.05 \
        --max_actuations 3 \
        --n_keep 175
"""

import argparse
import h5py
import numpy as np
import pandas as pd


def detect_drop(cube_z, table_height, lift_threshold=0.05, drop_ratio=0.5):
    """
    Returns True if the demo shows a "drop" pattern: the cube is lifted
    above lift_threshold at some point, then falls back below
    (lift_threshold * drop_ratio) afterwards.

    Returns True (treated as a failure) also if the cube is never lifted
    above lift_threshold at all, since that's arguably worse than a drop.
    """
    above = (cube_z - table_height) > lift_threshold
    if not above.any():
        return True  # never lifted -- exclude

    first_lift_idx = np.argmax(above)
    post_lift = cube_z[first_lift_idx:]
    dropped = ((post_lift - table_height) < (lift_threshold * drop_ratio)).any()
    return bool(dropped)


def count_gripper_actuations(gripper_action, threshold=0.0):
    """
    Counts the number of open<->close transitions in the gripper action
    channel. gripper_action is expected to be a 1D array (last action dim),
    where positive/negative (or > / < threshold) indicates close/open.
    """
    binary = (gripper_action > threshold).astype(int)
    transitions = np.diff(binary)
    return int(np.sum(np.abs(transitions)))


def find_cube_key(obs_group):
    """
    Tries to find the object/cube position key in the obs group.
    Adjust CANDIDATE_KEYS if your obs uses a different name.
    """
    candidate_keys = ["cube_free_pos", "object", "cube_pos", "object-state", "tool"]
    for k in candidate_keys:
        if k in obs_group:
            return k
    raise KeyError(
        f"Could not find a cube/object position key in obs. "
        f"Available keys: {list(obs_group.keys())}. "
        f"Edit CANDIDATE_KEYS in find_cube_key() to match your obs schema."
    )


def main(args):
    f = h5py.File(args.dataset, "r")
    demo_keys = list(f["data"].keys())
    print(f"Found {len(demo_keys)} demos in {args.dataset}")

    results = []
    for demo_key in demo_keys:
        d = f["data"][demo_key]

        if "obs" not in d:
            print(f"  [skip] {demo_key}: no 'obs' group (did conversion run correctly?)")
            continue

        obs = d["obs"]
        cube_key = find_cube_key(obs)
        cube_pos = obs[cube_key][:]
        # cube_pos may be (T, 3) position or (T, 10) object-state (pos+quat+rel);
        # assume z is index 2 in either case (first 3 dims = xyz position)
        cube_z = cube_pos[:, 2]

        actions = d["actions"][:]
        gripper_action = actions[:, -1]  # assumes gripper is last action dim -- verify for your controller

        dropped = detect_drop(cube_z, args.table_height, args.lift_threshold)
        n_actuations = count_gripper_actuations(gripper_action)

        results.append({
            "demo": demo_key,
            "dropped": dropped,
            "n_actuations": n_actuations,
            "length": actions.shape[0],
        })

    f.close()

    df = pd.DataFrame(results)
    print("\n=== Summary across all demos ===")
    print(df[["dropped", "n_actuations", "length"]].describe())
    print(f"\nDemos flagged as dropped/never-lifted: {df['dropped'].sum()} / {len(df)}")
    print(f"Demos at/above max_actuations ({args.max_actuations}): "
          f"{(df['n_actuations'] >= args.max_actuations).sum()} / {len(df)}")

    # hard filters
    passed = df[(~df["dropped"]) & (df["n_actuations"] < args.max_actuations)]
    print(f"\nDemos passing both filters: {len(passed)} / {len(df)}")

    if len(passed) < args.n_keep:
        print(
            f"WARNING: only {len(passed)} demos passed the filters, which is "
            f"fewer than --n_keep={args.n_keep}. Keeping all {len(passed)} that passed "
            f"rather than padding with worse demos. Consider relaxing "
            f"--max_actuations or --lift_threshold if you need more."
        )
        best_demos = passed["demo"].tolist()
    else:
        # among those that passed, rank by actuation count (ascending) as a
        # tiebreaker/quality proxy since jerk is excluded -- fewer actuations
        # is treated as "cleaner", so take the lowest-actuation top n_keep.
        # Feel free to change this secondary sort if you have another
        # preference for ranking demos that already passed both hard filters.
        ranked = passed.sort_values("n_actuations", ascending=True)
        best_demos = ranked.head(args.n_keep)["demo"].tolist()

    print(f"\nKeeping {len(best_demos)} demos -> writing to {args.output}")

    # write filtered file (full copy, including images, for kept demos only)
    src = h5py.File(args.dataset, "r")
    dst = h5py.File(args.output, "w")

    data_grp = dst.create_group("data")
    for k, v in src["data"].attrs.items():
        data_grp.attrs[k] = v
    data_grp.attrs["total"] = len(best_demos)

    for demo_key in best_demos:
        src.copy(f"data/{demo_key}", data_grp)

    src.close()
    dst.close()

    # also dump the per-demo metrics to CSV for your own inspection/threshold tuning
    csv_path = args.output.replace(".hdf5", "_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Per-demo metrics written to {csv_path}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to input image.hdf5")
    parser.add_argument("--output", required=True, help="Path to write filtered output hdf5")
    parser.add_argument("--table_height", type=float, default=0.8,
                         help="z-height of the table surface (table_offset[2] in your env)")
    parser.add_argument("--lift_threshold", type=float, default=0.05,
                         help="height above table_height counted as 'lifted'")
    parser.add_argument("--max_actuations", type=int, default=3,
                         help="maximum number of gripper open/close transitions allowed (exclusive)")
    parser.add_argument("--n_keep", type=int, default=175,
                         help="max number of demos to keep among those that pass filters")
    args = parser.parse_args()
    main(args)