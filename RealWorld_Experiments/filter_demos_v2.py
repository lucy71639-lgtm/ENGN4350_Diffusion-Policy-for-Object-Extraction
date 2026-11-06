#!/usr/bin/env python3
"""
Filter human demonstrations for diffusion-policy imitation learning.

Expected layout:
    <input_dir>/trajectory_*/actions.npy       (T, 10)  absolute target pose
    <input_dir>/trajectory_*/states.npy        (T, 10)  xyz | 6D rotation | gripper
    <input_dir>/trajectory_*/images.npy        (T, 3, H, W)
    <input_dir>/trajectory_*/traj_lengths.npy  (1,)

Task: approach tool in holder -> two-stage gripper close -> small move down -> lift.

What it does
  REJECT (demo not copied):
    * wrong_direction : the first motion heads away from the eventual grasp point
    * wrong_way_axis  : during the approach, any axis backtracks away from the
                        grasp point by more than --max-backtrack-mm
    * still_too_long  : the end effector is still for --reject-still frames or more
    * long_alignment  : too much time / dithering near the tool before the grasp
    * no_grasp        : gripper never reaches the "closed" level
  FLAG ONLY (demo still copied, listed for manual review):
    * long_still      : the end effector is still for more than --max-still frames
                        (no frames are trimmed)
    * no_post_grasp_down : the small post-grasp downward move was not detected

Outputs
  <output_dir>/trajectory_*/        unchanged copies of every non-rejected demo
  <report_dir>/demo_report.csv      one row per demo: metrics, flags, reasons
  <report_dir>/still_runs.csv       every still run longer than --max-still
  <report_dir>/thresholds.json      thresholds actually used
  <report_dir>/metric_histograms.png
  <report_dir>/plots/<status>/<demo>.png   per-demo plots for manual review

Usage
  python filter_demos.py /path/to/dataset --output-dir /path/to/filtered_dataset
  python filter_demos.py /path/to/dataset --dry-run          # reports only, copy nothing
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

POS = slice(0, 3)
ROT = slice(3, 9)
GRIP = 9
FILES = ("actions.npy", "states.npy", "images.npy", "traj_lengths.npy")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def runs(mask):
    """Return (start, length) for each run of True values in a 1-D bool array."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return [(int(s), int(e - s)) for s, e in zip(edges[::2], edges[1::2])]


def robust_upper(values, k, floor):
    """median + k * MAD (scaled to std), never below `floor`."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return floor
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    return float(max(med + k * mad, floor))


def load_demo(folder):
    states = np.load(folder / "states.npy")
    actions = np.load(folder / "actions.npy")
    tl = np.load(folder / "traj_lengths.npy")
    n_img = np.load(folder / "images.npy", mmap_mode="r").shape[0]
    T = states.shape[0]
    problems = []
    if actions.shape[0] != T or n_img != T:
        problems.append(f"length mismatch states={T} actions={actions.shape[0]} images={n_img}")
    if int(np.ravel(tl)[0]) != T:
        problems.append(f"traj_lengths={int(np.ravel(tl)[0])} but states has {T} frames")
    return states, actions, problems


# --------------------------------------------------------------------------- #
# Per-demo analysis
# --------------------------------------------------------------------------- #
def analyse(states, a):
    """Compute motion segmentation and all metrics for one demo."""
    T = states.shape[0]
    pos = states[:, POS]
    rot = states[:, ROT]
    g = states[:, GRIP]

    dpos = np.diff(pos, axis=0)
    step = np.linalg.norm(dpos, axis=1)                         # (T-1,)
    drot = np.linalg.norm(np.diff(rot, axis=0), axis=1)
    dgrip = np.abs(np.diff(g))

    moving = (step > a.move_eps) | (drot > a.rot_eps)           # step t -> t+1
    grip_change = dgrip > a.grip_eps

    m = {"n_frames": T}

    # ---- still runs: no translation, rotation or gripper change ----------- #
    # A run of k still steps means k+1 consecutive identical frames.
    still_step = ~moving & ~grip_change
    still_runs = [(s, k + 1) for s, k in runs(still_step)]     # (start_frame, n_frames)
    long_still = [(s, n) for s, n in still_runs if n > a.max_still]
    m["n_reject_still_runs"] = sum(n >= a.reject_still for _, n in still_runs)
    m["max_still_run"] = max([n for _, n in still_runs], default=0)
    m["n_long_still_runs"] = len(long_still)
    m["frames_in_long_still_runs"] = int(sum(n for _, n in long_still))

    # ---- motion start ------------------------------------------------------ #
    mv_idx = np.flatnonzero(moving)
    m["motion_start"] = int(mv_idx[0]) if mv_idx.size else -1

    # ---- grasp: first frame the gripper reaches the "closed" level --------- #
    closed = np.flatnonzero(g <= a.grip_closed)
    half = np.flatnonzero(g <= a.grip_half)
    m["half_close_frame"] = int(half[0]) if half.size else -1
    m["grasp_frame"] = int(closed[0]) if closed.size else -1

    info = dict(moving=moving, still_runs=still_runs, long_still=long_still)
    if m["grasp_frame"] < 0 or m["motion_start"] < 0 or m["motion_start"] >= m["grasp_frame"]:
        for k in ("init_cos", "init_dist_change_mm", "align_frames", "align_moving_frames",
                  "reversals", "path_ratio", "n_segments", "post_grasp_down_mm", "wrong_way_mm"):
            m[k] = np.nan
        return m, info

    s0, gf = m["motion_start"], m["grasp_frame"]
    grasp_pt = pos[gf]
    start_pt = pos[s0]
    info.update(grasp_pt=grasp_pt)

    # ---- 1. initial direction --------------------------------------------- #
    # Window = from motion start until a.init_dist metres of path have been
    # travelled (speed-independent), capped at the grasp frame.
    cum = np.concatenate([[0.0], np.cumsum(step[s0:gf])])
    w_end = s0 + int(np.searchsorted(cum, a.init_dist))
    w_end = min(max(w_end, s0 + 1), gf)
    init_vec = pos[w_end] - start_pt
    goal_vec = grasp_pt - start_pt
    if np.linalg.norm(init_vec) < 1e-9 or np.linalg.norm(goal_vec) < a.min_goal_dist:
        m["init_cos"] = np.nan          # cannot judge (start already at the tool)
    else:
        m["init_cos"] = float(init_vec @ goal_vec /
                              (np.linalg.norm(init_vec) * np.linalg.norm(goal_vec)))
    # Did the distance to the grasp point grow at any point in the window?
    d_goal = np.linalg.norm(pos[s0:w_end + 1] - grasp_pt, axis=1)
    m["init_dist_change_mm"] = float((d_goal.max() - d_goal[0]) * 1000)
    info.update(init_window=(s0, w_end))

    # ---- 2. alignment behaviour near the tool ----------------------------- #
    dist_to_grasp = np.linalg.norm(pos[:gf + 1] - grasp_pt, axis=1)
    inside = np.flatnonzero(dist_to_grasp[s0:] <= a.align_radius)
    a0 = s0 + int(inside[0]) if inside.size else gf
    m["align_start"] = a0
    m["align_frames"] = gf - a0                                 # all frames, incl. pauses
    m["align_moving_frames"] = int(moving[a0:gf].sum())

    # Direction reversals per axis inside the alignment window
    rev = 0
    for ax in range(3):
        d = dpos[a0:gf, ax]
        sgn = np.sign(d[np.abs(d) > a.move_eps])
        rev += int(np.count_nonzero(np.diff(sgn)))
    m["reversals"] = rev

    # Path efficiency for the whole approach (motion start -> grasp)
    path = float(step[s0:gf].sum())
    net = float(np.linalg.norm(goal_vec))
    m["path_ratio"] = path / net if net > a.min_goal_dist else np.nan
    m["n_segments"] = len(runs(moving[s0:gf]))                  # separate motion bursts
    info.update(align_window=(a0, gf))

    # ---- 1b. per-axis wrong-way travel during the approach ---------------- #
    # The approach moves one axis at a time, so a wrong-way move can happen at
    # the start of any axis's motion, not just the first one. For each axis
    # that has to move by at least --min-axis-move to reach the grasp point,
    # track progress toward the grasp point and measure the largest backtrack
    # (best progress so far minus current progress) before the end effector
    # enters the alignment zone. Dithering inside the zone is left to the
    # alignment checks.
    worst_mm, worst_ax, worst_frame = 0.0, "", -1
    for ax in range(3):
        need = goal_vec[ax]
        if abs(need) < a.min_axis_move:
            continue
        prog = np.sign(need) * (pos[s0:a0 + 1, ax] - start_pt[ax])
        backtrack = np.maximum.accumulate(np.maximum(prog, 0.0)) - prog
        i = int(np.argmax(backtrack))
        if backtrack[i] * 1000 > worst_mm:
            worst_mm, worst_ax, worst_frame = float(backtrack[i] * 1000), "xyz"[ax], s0 + i
    m["wrong_way_mm"] = worst_mm
    m["wrong_way_axis"] = worst_ax
    m["wrong_way_frame"] = worst_frame

    # ---- post-grasp small downward move (reported, not filtered) ---------- #
    z = pos[:, 2]
    after = z[gf:]
    lowest = gf + int(np.argmin(after))
    m["post_grasp_down_mm"] = float((z[gf] - z[lowest]) * 1000)
    mv_after = np.flatnonzero(moving[gf:lowest])
    down_start = gf + int(mv_after[0]) if mv_after.size else lowest
    m["post_grasp_down_frames"] = f"{down_start}-{lowest}"
    m["post_down_start"] = down_start
    info.update(post_down=(down_start, lowest))
    return m, info


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_demo(path, name, states, m, info, reasons, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos, g = states[:, POS], states[:, GRIP]
    t = np.arange(len(states))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8),
                                   gridspec_kw={"width_ratios": [1.6, 1]})

    for i, lab in enumerate("xyz"):
        ax1.plot(t, pos[:, i] - pos[0, i], label=f"{lab} - {lab}0")
    ax1.plot(t, g - g[0], "k--", lw=1, label="gripper - g0")
    for s, n in info["still_runs"]:
        ax1.axvspan(s, s + n - 1, color="red" if n > a.max_still else "grey",
                    alpha=(0.4 if n >= a.reject_still else 0.18) if n > a.max_still else 0.08, lw=0)
    if "init_window" in info:
        ax1.axvspan(*info["init_window"], color="orange", alpha=0.3, lw=0, label="initial window")
    if "align_window" in info:
        ax1.axvspan(*info["align_window"], color="purple", alpha=0.12, lw=0, label="alignment window")
    if "post_down" in info:
        ax1.axvspan(*info["post_down"], color="green", alpha=0.2, lw=0, label="post-grasp down")
    if m.get("wrong_way_mm", 0) > a.max_backtrack_mm:
        ax1.axvline(m["wrong_way_frame"], color="red", ls="--", lw=1.2, label="worst backtrack")
    if m["grasp_frame"] >= 0:
        ax1.axvline(m["grasp_frame"], color="k", lw=1)
    ax1.set_xlabel("frame")
    ax1.set_ylabel("change from first frame (m)")
    ax1.legend(fontsize=7, loc="best")
    ax1.set_title("light red = still > %d frames (flag), dark red = still >= %d frames (reject)" % (a.max_still, a.reject_still), fontsize=9)

    # Spatial view: the two axes with the most motion
    span = pos.max(0) - pos.min(0)
    i, j = sorted(np.argsort(span)[-2:])
    sc = ax2.scatter(pos[:, i], pos[:, j], c=t, cmap="viridis", s=10)
    ax2.plot(pos[:, i], pos[:, j], color="grey", lw=0.5)
    ax2.plot(pos[0, i], pos[0, j], "go", ms=9, label="start")
    if "grasp_pt" in info:
        ax2.plot(info["grasp_pt"][i], info["grasp_pt"][j], "r*", ms=14, label="grasp")
        circ = plt.Circle((info["grasp_pt"][i], info["grasp_pt"][j]), a.align_radius,
                          fill=False, ls=":", color="purple")
        ax2.add_patch(circ)
    ax2.set_xlabel("xyz"[i]); ax2.set_ylabel("xyz"[j])
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.legend(fontsize=7)
    fig.colorbar(sc, ax=ax2, label="frame")

    def f(x, fmt):
        return "nan" if not np.isfinite(x) else format(x, fmt)
    fig.suptitle(
        f"{name}  |  {', '.join(reasons) or 'ok'}\n"
        f"init_cos={f(m['init_cos'], '.2f')}  align_frames={m['align_frames']}  "
        f"reversals={m['reversals']}  path_ratio={f(m['path_ratio'], '.2f')}  "
        f"wrong_way={f(m['wrong_way_mm'], '.1f')}mm({m.get('wrong_way_axis', '')})  "
        f"max_still={m['max_still_run']}  post_down={f(m['post_grasp_down_mm'], '.1f')}mm",
        fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_histograms(path, rows, thr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["init_cos", "align_frames", "reversals", "path_ratio", "max_still_run", "post_grasp_down_mm",
            "wrong_way_mm"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, k in zip(axes.ravel(), keys):
        v = np.array([r[k] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            ax.hist(v, bins=40, color="#4878a8")
        if k in thr:
            ax.axvline(thr[k], color="red", ls="--", label=f"threshold {thr[k]:.2f}")
            ax.legend(fontsize=8)
        ax.set_title(k)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_dir", type=Path)
    p.add_argument("--output-dir", type=Path, default=None, help="default: <input_dir>/../filtered_dataset")
    p.add_argument("--report-dir", type=Path, default=None, help="default: <output_dir>/../filter_report")
    p.add_argument("--dry-run", action="store_true", help="write reports/plots but copy no demos")
    p.add_argument("--overwrite", action="store_true", help="delete an existing output dir first")
    p.add_argument("--plots", choices=["none", "flagged", "all"], default="flagged")

    g = p.add_argument_group("motion detection")
    g.add_argument("--move-eps", type=float, default=5e-4, help="m per frame counted as moving")
    g.add_argument("--rot-eps", type=float, default=1e-3, help="6D-rotation change counted as moving")
    g.add_argument("--grip-eps", type=float, default=5e-3, help="gripper change counted as activity")
    g.add_argument("--grip-half", type=float, default=0.225, help="gripper <= this = first-stage close")
    g.add_argument("--grip-closed", type=float, default=0.09, help="gripper <= this = grasped")

    g = p.add_argument_group("still frames (flag only)")
    g.add_argument("--max-still", type=int, default=8, help="flag still runs longer than this many frames")
    g.add_argument("--reject-still", type=int, default=30,
                   help="reject demos with a still run of this many frames or more")

    g = p.add_argument_group("wrong initial direction (reject)")
    g.add_argument("--init-dist", type=float, default=0.02, help="path length (m) of the initial window")
    g.add_argument("--min-cos", type=float, default=0.0,
                   help="reject if cos(initial motion, direction to grasp) is below this")
    g.add_argument("--max-away-mm", type=float, default=10.0,
                   help="reject if distance to grasp point grows by more than this in the initial window")
    g.add_argument("--max-backtrack-mm", type=float, default=5.0,
                   help="reject if any axis moves this far away from the grasp point during the approach")
    g.add_argument("--min-axis-move", type=float, default=0.005,
                   help="only check axes that must move at least this far (m) to reach the grasp point")
    g.add_argument("--min-goal-dist", type=float, default=0.01, help="skip direction check if start is this close to grasp")

    g = p.add_argument_group("long alignment (reject)")
    g.add_argument("--align-radius", type=float, default=0.02, help="radius (m) around grasp point = 'near the tool'")
    g.add_argument("--k-mad", type=float, default=3.0, help="auto threshold = median + k*MAD over the dataset")
    g.add_argument("--max-align-frames", type=float, default=None, help="fixed threshold (overrides auto)")
    g.add_argument("--max-reversals", type=float, default=None, help="fixed threshold (overrides auto)")
    g.add_argument("--max-path-ratio", type=float, default=None, help="fixed threshold (overrides auto)")
    a = p.parse_args()

    inp = a.input_dir.resolve()
    out = (a.output_dir or inp.parent / "filtered_dataset").resolve()
    rep = (a.report_dir or out.parent / "filter_report").resolve()
    if out == inp:
        sys.exit("output dir must differ from input dir")

    demos = sorted(d for d in inp.glob("trajectory_*") if d.is_dir())
    if not demos:
        sys.exit(f"no trajectory_* folders in {inp}")
    print(f"Found {len(demos)} demos in {inp}")

    # ---- pass 1: metrics -------------------------------------------------- #
    rows, cache, still_rows = [], {}, []
    for d in demos:
        missing = [f for f in FILES if not (d / f).exists()]
        if missing:
            rows.append({"demo": d.name, "status": "rejected", "reasons": "missing:" + "|".join(missing)})
            continue
        states, actions, problems = load_demo(d)
        m, info = analyse(states, a)
        m["demo"] = d.name
        m["problems"] = "; ".join(problems)
        rows.append(m)
        cache[d.name] = (states, m, info)
        for s, n in info["long_still"]:
            gf = m["grasp_frame"]
            pd = m.get("post_down_start", -1)
            phase = ("before_motion" if s + n - 1 <= max(m["motion_start"], 0) else
                     "approach" if gf < 0 or s < m["half_close_frame"] else
                     "between_closes" if s < gf else
                     "after_grasp_before_down" if pd >= 0 and s + n - 1 <= pd else "after_down")
            still_rows.append({"demo": d.name, "start_frame": s, "end_frame": s + n - 1,
                               "n_frames": n, "phase": phase})

    valid = [r for r in rows if "align_frames" in r]

    # ---- thresholds ------------------------------------------------------- #
    thr = {
        "init_cos": a.min_cos,
        "align_frames": a.max_align_frames if a.max_align_frames is not None
        else robust_upper([r["align_frames"] for r in valid], a.k_mad, floor=10),
        "reversals": a.max_reversals if a.max_reversals is not None
        else robust_upper([r["reversals"] for r in valid], a.k_mad, floor=4),
        "path_ratio": a.max_path_ratio if a.max_path_ratio is not None
        else robust_upper([r["path_ratio"] for r in valid], a.k_mad, floor=1.3),
        "max_still_run": a.reject_still,
        "wrong_way_mm": a.max_backtrack_mm,
    }

    # ---- pass 2: decide --------------------------------------------------- #
    for r in rows:
        if "status" in r:
            continue
        rej, flg = [], []
        if r["problems"]:
            rej.append("data_problem")
        if r["grasp_frame"] < 0:
            rej.append("no_grasp")
        elif r["motion_start"] < 0 or r["motion_start"] >= r["grasp_frame"]:
            rej.append("no_motion_before_grasp")
        else:
            if (np.isfinite(r["init_cos"]) and r["init_cos"] < thr["init_cos"]) or \
                    r["init_dist_change_mm"] > a.max_away_mm:
                rej.append("wrong_direction")
            if r["wrong_way_mm"] > a.max_backtrack_mm:
                rej.append(f"wrong_way_axis({r['wrong_way_axis']})")
            if r["align_frames"] > thr["align_frames"]:
                rej.append("long_alignment(time)")
            if r["reversals"] > thr["reversals"]:
                rej.append("long_alignment(reversals)")
            if np.isfinite(r["path_ratio"]) and r["path_ratio"] > thr["path_ratio"]:
                rej.append("long_alignment(path)")
            if not (r["post_grasp_down_mm"] > 1.0):
                flg.append("no_post_grasp_down")
        if r["n_reject_still_runs"] > 0:
            rej.append("still_too_long")
        elif r["n_long_still_runs"] > 0:
            flg.append("long_still")
        r["status"] = "rejected" if rej else ("flagged" if flg else "ok")
        r["reasons"] = ", ".join(rej + flg)

    # ---- write reports ---------------------------------------------------- #
    rep.mkdir(parents=True, exist_ok=True)
    cols = ["demo", "status", "reasons", "n_frames", "motion_start", "half_close_frame", "grasp_frame",
            "init_cos", "init_dist_change_mm", "wrong_way_mm", "wrong_way_axis", "wrong_way_frame", "align_start", "align_frames", "align_moving_frames",
            "reversals", "path_ratio", "n_segments", "max_still_run", "n_long_still_runs", "n_reject_still_runs",
            "frames_in_long_still_runs", "post_grasp_down_mm", "post_grasp_down_frames", "problems"]
    with open(rep / "demo_report.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    with open(rep / "still_runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["demo", "start_frame", "end_frame", "n_frames", "phase"])
        w.writeheader()
        w.writerows(still_rows)
    with open(rep / "thresholds.json", "w") as fh:
        json.dump({"thresholds_used": thr, "args": {k: str(v) for k, v in vars(a).items()}}, fh, indent=2)
    plot_histograms(rep / "metric_histograms.png", valid, thr)

    if a.plots != "none":
        for r in rows:
            if r["demo"] not in cache or (a.plots == "flagged" and r["status"] == "ok"):
                continue
            states, m, info = cache[r["demo"]]
            plot_demo(rep / "plots" / r["status"] / f"{r['demo']}.png", r["demo"], states, m, info,
                      r["reasons"].split(", ") if r["reasons"] else [], a)

    # ---- copy kept demos -------------------------------------------------- #
    kept = [r for r in rows if r["status"] != "rejected"]
    if not a.dry_run:
        if out.exists():
            if not a.overwrite:
                sys.exit(f"{out} exists; use --overwrite to replace it")
            shutil.rmtree(out)
        out.mkdir(parents=True)
        for r in kept:
            shutil.copytree(inp / r["demo"], out / r["demo"])

    # ---- summary ---------------------------------------------------------- #
    n = len(rows)
    count = lambda s: sum(r["status"] == s for r in rows)
    print(f"\nThresholds: align_frames>{thr['align_frames']:.1f}, reversals>{thr['reversals']:.1f}, "
          f"path_ratio>{thr['path_ratio']:.2f}, init_cos<{thr['init_cos']:.2f}, "
          f"away>{a.max_away_mm}mm, backtrack>{a.max_backtrack_mm}mm,\n            "
          f"flag still>{a.max_still} frames, reject still>={a.reject_still} frames")
    print(f"ok: {count('ok')}   flagged (kept): {count('flagged')}   rejected: {count('rejected')}   of {n}")
    reason_counts = {}
    for r in rows:
        for reason in filter(None, r["reasons"].split(", ")):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    for k, v in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:30s} {v}")
    print(f"\nReports: {rep}")
    print("Demos:   " + ("(dry run, nothing copied)" if a.dry_run else f"{len(kept)} copied to {out}"))


if __name__ == "__main__":
    main()
