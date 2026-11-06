"""
Combine multiple robosuite demo.hdf5 files (as produced by collect_human_demonstrations.py)
into a single hdf5 dataset.

Usage:
    python merge_hdf5_demos.py demo_a.hdf5 demo_b.hdf5 -o merged.hdf5

    # more than two files also works
    python merge_hdf5_demos.py a.hdf5 b.hdf5 c.hdf5 -o merged.hdf5

    # merge even if the env_info of the files differs (not recommended)
    python merge_hdf5_demos.py a.hdf5 b.hdf5 -o merged.hdf5 --force
"""

import argparse
import datetime
import json
import os
import re
import sys

import h5py


def demo_sort_key(name):
    """Sort demo_1, demo_2, ..., demo_10 numerically rather than lexicographically."""
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else float("inf")


def decode(x):
    return x.decode("utf-8") if isinstance(x, bytes) else x


def main():
    parser = argparse.ArgumentParser(description="Merge robosuite hdf5 demo files into one.")
    parser.add_argument("inputs", nargs="+", help="Input hdf5 files, in the order they should be merged.")
    parser.add_argument("-o", "--output", required=True, help="Output hdf5 path.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Merge even if env / env_info differ between files (first file's metadata is used).",
    )
    args = parser.parse_args()

    if len(args.inputs) < 2:
        sys.exit("Provide at least two input files.")
    for p in args.inputs:
        if not os.path.isfile(p):
            sys.exit(f"Input not found: {p}")
    if os.path.abspath(args.output) in [os.path.abspath(p) for p in args.inputs]:
        sys.exit("Output path must differ from the input paths.")

    # ---- Check metadata consistency before writing anything ----
    ref_env, ref_env_info, ref_version = None, None, None
    for p in args.inputs:
        with h5py.File(p, "r") as f:
            if "data" not in f:
                sys.exit(f"{p} has no 'data' group.")
            attrs = f["data"].attrs
            env = decode(attrs.get("env"))
            env_info = decode(attrs.get("env_info"))
            version = decode(attrs.get("repository_version"))

            if ref_env_info is None:
                ref_env, ref_env_info, ref_version = env, env_info, version
                continue

            problems = []
            if env != ref_env:
                problems.append(f"env: {env!r} vs {ref_env!r}")
            if env_info != ref_env_info:
                try:
                    a, b = json.loads(env_info), json.loads(ref_env_info)
                    diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
                except Exception:
                    diff = "env_info strings differ"
                problems.append(f"env_info differences (this file, first file): {diff}")
            if version != ref_version:
                print(f"Warning: repository_version differs in {p} ({version} vs {ref_version})")

            if problems:
                msg = f"{p} is not compatible with {args.inputs[0]}:\n  " + "\n  ".join(problems)
                if args.force:
                    print("Warning (--force): " + msg)
                else:
                    sys.exit(msg + "\nUse --force to merge anyway.")

    # ---- Merge ----
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    total = 0
    with h5py.File(args.output, "w") as out:
        out_grp = out.create_group("data")

        for p in args.inputs:
            with h5py.File(p, "r") as f:
                demos = sorted(f["data"].keys(), key=demo_sort_key)
                for name in demos:
                    total += 1
                    # copy() preserves datasets, attributes (incl. model_file) and any subgroups
                    f.copy(f["data"][name], out_grp, name=f"demo_{total}")
                print(f"{p}: copied {len(demos)} demos")

        # Metadata (same layout as the collection script)
        now = datetime.datetime.now()
        out_grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
        out_grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
        out_grp.attrs["repository_version"] = ref_version
        out_grp.attrs["env"] = ref_env
        out_grp.attrs["env_info"] = ref_env_info

    print(f"Done. {total} demos written to {args.output}")


if __name__ == "__main__":
    main()