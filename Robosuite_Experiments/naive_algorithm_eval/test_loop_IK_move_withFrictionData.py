"""
Run an automated series of trials that grasp and lift a cube constrained
between two fixed cubes, and collect + plot the vertical resistive friction
force acting on the free cube throughout each lift.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import mujoco
import imageio

from robosuite.controllers import load_composite_controller_config
from robosuite.utils.transform_utils import (
    mat2euler,
    quat_multiply,
    axisangle2quat,
    quat2mat,
)

from environments.LIFT_withFriction import LiftCubeBetweenFixedCubes
from utils.ik_nullspace_return import solve_ik_ret
from utils.move import move_to_hidden, close_gripper


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_TRIALS = 3
SAVE_FAILURE_IMAGES = False        # set to True to save images of failed trials
FAILURE_IMG_DIR = "ik_failures"
FORCE_PLOT_DIR = "force_plots_gap_neg0.002_fric0.5"

# geoms whose contact with the free cube we care about
# (these are the actual collision geoms in the model; the "_vis" siblings
# are visual-only and never appear in sim.data.contact)
CONTACT_PAIRS = {
    "left": ("cube_free_g0", "cube_left_g0"),
    "right": ("cube_free_g0", "cube_right_g0"),
}


# ---------------------------------------------------------------------------
# Force utilities
# ---------------------------------------------------------------------------
def get_contact_force_on(env, ref_geom_name, other_geom_name):
    """
    Sum the normal and friction forces (world frame, 3D vectors) acting ON
    ref_geom_name due to all current contacts with other_geom_name.

    MuJoCo's mj_contactForce reports the contact-frame force as acting on
    con.geom2 (pushed away from con.geom1 along the normal), and it decides
    internally which of the two geoms is geom1/geom2 for a given contact —
    not based on the order they're passed in here. Without correcting for
    that, summing forces from two different contact pairs (e.g. left and
    right neighbours of the same cube) can silently mix sign conventions
    and partially cancel instead of accumulate. This function always
    returns the force as it acts on ref_geom_name, regardless of which
    slot MuJoCo assigned it to, so results from multiple calls with the
    same ref_geom_name can be safely added together.

    Call after env.step() (or sim.forward()) so sim.data.contact reflects
    the current state.
    """
    sim = env.sim
    # robosuite wraps the raw mujoco.MjModel/MjData; fall back gracefully
    # if a given robosuite version doesn't use the underscore attrs.
    model = getattr(sim.model, "_model", sim.model)
    data = getattr(sim.data, "_data", sim.data)

    ref_id = sim.model.geom_name2id(ref_geom_name)
    other_id = sim.model.geom_name2id(other_geom_name)

    total_normal = np.zeros(3)
    total_friction = np.zeros(3)

    for i in range(data.ncon):
        con = data.contact[i]
        if {con.geom1, con.geom2} != {ref_id, other_id}:
            continue

        # 6D force in the contact frame: [normal, tangent1, tangent2, torsional, roll1, roll2]
        c_force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, c_force)

        frame = np.array(con.frame).reshape(3, 3)  # row0=normal, row1/2=tangents (world frame)
        normal_world = c_force[0] * frame[0]
        friction_world = c_force[1] * frame[1] + c_force[2] * frame[2]

        # mj_contactForce's convention is "force on geom2". Flip if
        # ref_geom_name ended up as geom1, so the returned vector always
        # acts on ref_geom_name.
        if con.geom1 == ref_id:
            normal_world = -normal_world
            friction_world = -friction_world

        total_normal += normal_world
        total_friction += friction_world

    return total_normal, total_friction


def log_contact_forces(env, force_log):
    """
    Append the current vertical (z) friction force acting on cube_free from
    each of its two fixed neighbours to force_log, in a consistent sign
    convention, so total_z is their true sum (the net vertical resistive
    force the manipulator has to overcome), not a partial cancellation.

    NOTE: this assumes move_to_hidden()/close_gripper() (utils/move.py)
    accept an optional `log_fn` callback that they invoke once per internal
    simulation step. If your current versions of those helpers don't yet
    support that, add a single line such as:

        if log_fn is not None:
            log_fn()

    inside their step loops (right after env.step(...)), and this script
    will work unmodified. Without that hook you'll only get one force
    sample per call to move_to_hidden/close_gripper rather than a dense
    time series.
    """
    _, f_left = get_contact_force_on(env, *CONTACT_PAIRS["left"])
    _, f_right = get_contact_force_on(env, *CONTACT_PAIRS["right"])

    t = len(force_log["t"])
    force_log["t"].append(t)
    force_log["left_z"].append(f_left[2])
    force_log["right_z"].append(f_right[2])
    force_log["total_z"].append(f_left[2] + f_right[2])


def plot_resistive_force_z(force_log, save_path):
    """
    Plot the z-component (vertical) of the friction force resisting the
    lift, from both cube_left and cube_right contacts plus their sum,
    over the course of one episode, and save it as a PNG.

    Args:
        force_log (dict): with keys "t", "left_z", "right_z", "total_z"
            (lists of floats, one entry per logged timestep).
        save_path (str): full path (including .png) to save the figure to.
    """
    if len(force_log["t"]) == 0:
        return  # nothing was logged (e.g. episode reset immediately)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(force_log["t"], force_log["left_z"], label="cube_free vs cube_left", alpha=0.7)
    ax.plot(force_log["t"], force_log["right_z"], label="cube_free vs cube_right", alpha=0.7)
    ax.plot(force_log["t"], force_log["total_z"], label="total", color="black", linewidth=2)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("timestep")
    ax.set_ylabel("friction force on cube_free, z (N)")
    ax.set_title("Vertical resistive friction force during lift")
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved resistive force plot to {save_path}")


def save_failure_images(env, trial, q_goal):
    """Render and save diagnostic images for a failed trial."""
    os.makedirs(FAILURE_IMG_DIR, exist_ok=True)

    # Option A: render the current (final) simulated state as-is
    img = env.sim.render(camera_name="frontview", width=512, height=512)
    imageio.imwrite(f"{FAILURE_IMG_DIR}/trial_{trial}_final_state.png", img[::-1])

    # Option B: force the robot into the raw IK-output joint config
    # (bypasses whatever the low-level controller actually achieved)
    if q_goal is not None:
        robot = env.robots[0]
        qpos_indices = robot._ref_joint_pos_indexes
        env.sim.data.qpos[qpos_indices] = q_goal
        env.sim.forward()  # propagate kinematics without stepping physics
        img_ik = env.sim.render(camera_name="frontview", width=512, height=512)
        imageio.imwrite(f"{FAILURE_IMG_DIR}/trial_{trial}_ik_target.png", img_ik[::-1])


# ---------------------------------------------------------------------------
# Controller / environment setup
# ---------------------------------------------------------------------------
arm_key = "right"  # confirm against the printed dict structure

lower_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
upper_limits = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

controller_config = load_composite_controller_config(robot="Panda")
arm_cfg = controller_config["body_parts"][arm_key]
arm_cfg["type"] = "JOINT_POSITION"
arm_cfg["input_type"] = "absolute"
arm_cfg["input_min"] = -1
arm_cfg["input_max"] = 1
arm_cfg["output_min"] = lower_limits
arm_cfg["output_max"] = upper_limits
arm_cfg["qpos_limits"] = [lower_limits, upper_limits]  # redundant safety clip, matches same range

env = LiftCubeBetweenFixedCubes(
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=True,  # keep off for testing speed
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
    cube_half_size=0.025,
    gap=-0.002,  # -ve gap gives slight squeeze
    friction=(0.5, 0.005, 0.0001),
)

os.makedirs(FORCE_PLOT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Trial loop
# ---------------------------------------------------------------------------
results = []  # list of dicts: {"trial": i, "success": bool}

for trial in range(N_TRIALS):
    obs = env.reset()
    force_log = {"t": [], "left_z": [], "right_z": [], "total_z": []}

    def log_fn():
        log_contact_forces(env, force_log)

    # -----------------------------------------------------------------
    # Per-trial setup
    # -----------------------------------------------------------------
    robot = env.robots[0]
    joint_idx = robot._ref_joint_pos_indexes
    dof_idx = robot._ref_joint_vel_indexes

    q_start = robot.sim.data.qpos[joint_idx].copy()

    # Min-Max joint ranges for the Panda robot, used to keep motion planning
    # within joint limits.
    joint_ranges = robot.sim.model.jnt_range[joint_idx]
    lower = joint_ranges[:, 0]
    upper = joint_ranges[:, 1]

    # Locate the free cube
    obj_pos = obs["cube_free_pos"].copy()
    target_pos = obj_pos
    eef_pos = obs["robot0_eef_pos"].copy()

    # Default top-down end-effector orientation, used as a reference for the
    # target orientation.
    down_quat = obs["robot0_eef_quat"].copy()
    down_mat = quat2mat(down_quat)

    cube_mat = env.sim.data.get_body_xmat("cube_free_main").copy()  # (3,3)
    _, _, yaw = mat2euler(cube_mat)
    yaw = yaw % (np.pi / 2)

    yaw_quat = axisangle2quat(np.array([0, 0, yaw]))
    target_quat = quat_multiply(yaw_quat, down_quat)

    # -----------------------------------------------------------------
    # Solve IK for the grasp pose
    # -----------------------------------------------------------------
    q_goal, ok = solve_ik_ret(
        sim=env.sim,
        target_pos=target_pos,
        joint_indexes=joint_idx,
        dof_indexes=dof_idx,
        target_quat=None,
        site_name="gripper0_right_grip_site",
        qpos_init=q_start,       # initial guess for the IK solver
        q_preferred=q_start,     # bias the configuration towards the start pose
        joint_limits=(lower, upper),
    )

    success = False
    if ok:
        # 1) reach down to the cube, gripper open
        move_to_hidden(q_goal, -1, env, k=0.1, max_steps=200, log_fn=log_fn)
        # 2) grasp
        close_gripper(env, log_fn=log_fn)
        # 3) lift back to the starting pose, gripper closed
        move_to_hidden(q_start, 1, env, k=0.1, max_steps=200, log_fn=log_fn)

        success = env._check_success()
    else:
        print(f"Trial {trial}: IK failed to find a grasp pose, skipping motion.")

    results.append({"trial": trial, "success": success})

    # -----------------------------------------------------------------
    # Save the resistive force plot for this episode
    # -----------------------------------------------------------------
    plot_resistive_force_z(
        force_log, os.path.join(FORCE_PLOT_DIR, f"episode_{trial + 1}_resistive_force_z.png")
    )

    # -----------------------------------------------------------------
    # On failure, optionally save diagnostic images
    # -----------------------------------------------------------------
    if not success and SAVE_FAILURE_IMAGES:
        save_failure_images(env, trial, q_goal)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
n_success = sum(r["success"] for r in results)
print(f"Success rate: {n_success}/{N_TRIALS} = {100 * n_success / N_TRIALS:.1f}%")