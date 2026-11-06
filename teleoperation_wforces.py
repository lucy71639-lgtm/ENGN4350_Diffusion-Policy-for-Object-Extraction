#TELEOPERATION DEMO FROM ROBOSUITE : https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/demos/demo_device_control.py
#MODIFIED TO SET UP 'SQUEEZED' CUBE FRICTION ENVIRONMENT
#FURTHER MODIFIED TO: (1) run a fixed number of teleoperated demonstrations,
#                      (2) log gripper squeeze force + vertical (z) contact
#                          force during each demonstration, and
#                      (3) save one force plot per demonstration.
#
# Force-logging code (get_contact_force_on / log_contact_forces /
# plot_resistive_force_z / CONTACT_PAIRS / GRIPPER_CONTACT_PAIRS) is carried
# over from the automated test-trial script almost unchanged, since the
# environment and geom names are the same.

"""Teleoperate robot with keyboard or SpaceMouse, for N demonstrations,
while logging contact forces on the free cube.

***Choose user input option with the --device argument***

Keyboard:
    We use the keyboard to control the end-effector of the robot.
    The keyboard provides 6-DoF control commands through various keys.
    The commands are mapped to joint velocities through an inverse kinematics
    solver from Bullet physics.

    Note:
        To run this script with macOS, you must run it with root access.

SpaceMouse:

    We use the SpaceMouse 3D mouse to control the end-effector of the robot.
    The mouse provides 6-DoF control commands. The commands are mapped to joint
    velocities through an inverse kinematics solver from Bullet physics.

    The two side buttons of SpaceMouse are used for controlling the grippers.

Additionally, --pos_sensitivity and --rot_sensitivity provide relative gains for increasing / decreasing the user input
device sensitivity.

***New in this version***

    --num-demos: how many demonstrations to collect before the script exits
        (default 5). Each time the device sends a "reset" signal (input2action
        returns None) or the task succeeds, the current demonstration ends,
        its force log is saved as a PNG under --force-plot-dir, and the next
        demonstration begins automatically, until --num-demos have been
        completed.

    --force-plot-dir: directory to save the per-demonstration force plots to
        (default "teleop_force_plots").

Examples:

    $ python demo_device_control_with_forces.py --device keyboard --num-demos 5
"""

import argparse
import os
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import mujoco
import numpy as np

import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper

from robosuite.controllers import load_composite_controller_config
from environments.LIFT_withFriction import LiftCubeBetweenFixedCubes


# ---------------------------------------------------------------------------
# Force utilities (carried over from the automated test-trial script)
# ---------------------------------------------------------------------------
# geoms whose contact with the free cube we care about
# (these are the actual collision geoms in the model; the "_vis" siblings
# are visual-only and never appear in sim.data.contact)
CONTACT_PAIRS = {
    "left": ("cube_free_g0", "cube_left_g0"),
    "right": ("cube_free_g0", "cube_right_g0"),
}

# gripper pad geoms that actually contact the cube (the non-"_pad"
# "_collision" geoms are the rigid finger bodies further back and typically
# don't touch the object)
GRIPPER_CONTACT_PAIRS = {
    "finger1": ("cube_free_g0", "gripper0_right_finger1_pad_collision"),
    "finger2": ("cube_free_g0", "gripper0_right_finger2_pad_collision"),
}


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

    Call after env.step() so sim.data.contact reflects the current state.
    """
    sim = env.sim
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

        c_force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, c_force)

        frame = np.array(con.frame).reshape(3, 3)  # row0=normal, row1/2=tangents (world frame)
        normal_world = c_force[0] * frame[0]
        friction_world = c_force[1] * frame[1] + c_force[2] * frame[2]

        if con.geom1 == ref_id:
            normal_world = -normal_world
            friction_world = -friction_world

        total_normal += normal_world
        total_friction += friction_world

    return total_normal, total_friction


def new_force_log():
    """Return a fresh, empty force log dict for one demonstration."""
    return {
        "t": [],
        "left_z": [],
        "right_z": [],
        "total_z": [],
        "grip_finger1_normal": [],
        "grip_finger2_normal": [],
        "grip_normal_total": [],
        "grip_friction_z_total": [],
    }


def log_contact_forces(env, force_log):
    """
    Append the current contact forces acting on cube_free to force_log.
    Call this once per env.step() during teleoperation.

    - left_z / right_z / total_z: vertical (z) friction from the two fixed
      neighbour cubes, in a consistent sign convention (see
      get_contact_force_on), so total_z is their true sum.
    - grip_finger1_normal / grip_finger2_normal / grip_normal_total: the
      magnitude of the squeeze (normal) force each gripper pad applies to
      the cube, and their sum.
    - grip_friction_z_total: the vertical friction the gripper pads
      themselves provide, i.e. the force actually holding the cube up
      against gravity via the grip.
    """
    _, f_left = get_contact_force_on(env, *CONTACT_PAIRS["left"])
    _, f_right = get_contact_force_on(env, *CONTACT_PAIRS["right"])

    n_f1, f_f1 = get_contact_force_on(env, *GRIPPER_CONTACT_PAIRS["finger1"])
    n_f2, f_f2 = get_contact_force_on(env, *GRIPPER_CONTACT_PAIRS["finger2"])

    t = len(force_log["t"])
    force_log["t"].append(t)
    force_log["left_z"].append(f_left[2])
    force_log["right_z"].append(f_right[2])
    force_log["total_z"].append(f_left[2] + f_right[2])

    force_log["grip_finger1_normal"].append(np.linalg.norm(n_f1))
    force_log["grip_finger2_normal"].append(np.linalg.norm(n_f2))
    force_log["grip_normal_total"].append(np.linalg.norm(n_f1) + np.linalg.norm(n_f2))
    force_log["grip_friction_z_total"].append(f_f1[2] + f_f2[2])


def plot_resistive_force_z(force_log, save_path):
    """
    Plot two panels for one demonstration and save as a single PNG:

    1. The z-component (vertical) of the friction force resisting the lift
       from both cube_left and cube_right contacts, plus their sum.
    2. The gripper's squeeze (normal) force per finger and total, plus the
       vertical friction the gripper itself provides to support the cube.
    """
    if len(force_log["t"]) == 0:
        print(f"Nothing logged for {save_path}, skipping plot (demo may have been reset immediately).")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.plot(force_log["t"], force_log["left_z"], label="cube_free vs cube_left", alpha=0.7)
    ax1.plot(force_log["t"], force_log["right_z"], label="cube_free vs cube_right", alpha=0.7)
    ax1.plot(force_log["t"], force_log["total_z"], label="total", color="black", linewidth=2)
    ax1.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("friction force on cube_free, z (N)")
    ax1.set_title("Vertical resistive friction force from neighbour cubes")
    ax1.legend()

    ax2.plot(force_log["t"], force_log["grip_finger1_normal"], label="finger1 squeeze", alpha=0.7)
    ax2.plot(force_log["t"], force_log["grip_finger2_normal"], label="finger2 squeeze", alpha=0.7)
    ax2.plot(
        force_log["t"], force_log["grip_normal_total"], label="total squeeze",
        color="black", linewidth=2,
    )
    ax2.plot(
        force_log["t"], force_log["grip_friction_z_total"], label="grip support (z)",
        color="red", linewidth=2, linestyle="--",
    )
    ax2.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("timestep")
    ax2.set_ylabel("gripper force (N)")
    ax2.set_title("Gripper squeeze force and vertical support from grip friction")
    ax2.legend()

    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved resistive force plot to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--switch-on-grasp",
        action="store_true",
        help="Switch gripper control on gripper action",
    )
    parser.add_argument(
        "--toggle-camera-on-grasp",
        action="store_true",
        help="Switch camera angle on gripper action",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_MINK_IK') or json file (see robosuite/controllers/config for examples) or None to get the robot's default controller if it exists",
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simluation runs faster than specified frame rate; 20 fps is real time.",
    )
    parser.add_argument(
        "--reverse_xy",
        type=bool,
        default=False,
        help="(DualSense Only)Reverse the effect of the x and y axes of the joystick.It is used to handle the case that the left/right and front/back sides of the view are opposite to the LX and LY of the joystick(Push LX up but the robot move left in your view)",
    )
    parser.add_argument(
        "--num-demos",
        type=int,
        default=5,
        help="Number of teleoperated demonstrations to collect before exiting.",
    )
    parser.add_argument(
        "--force-plot-dir",
        type=str,
        default="teleop_force_plots",
        help="Directory to save the per-demonstration force plots to.",
    )
    args = parser.parse_args()

    # Get controller config
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None

    # Create environment
    env = LiftCubeBetweenFixedCubes(
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
        cube_half_size=0.025,
        gap=-0.002,               # cubes start touching
        friction=(0.5, 0.005, 0.0001),
    )

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    # env._check_success() relies on the wrapper's __getattr__ forwarding
    # to the underlying LiftCubeBetweenFixedCubes instance. That should
    # work, but to avoid depending on it silently, grab the raw base env
    # once and call _check_success() on that directly instead of on the
    # wrapper.
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env

    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    os.makedirs(args.force_plot_dir, exist_ok=True)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
        env.viewer.add_keypress_callback(device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "dualsense":
        from robosuite.devices import DualSense

        device = DualSense(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            reverse_xy=args.reverse_xy,
        )
    elif args.device == "mjgui":
        from robosuite.devices.mjgui import MJGUI

        device = MJGUI(env=env)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard', 'dualsense', 'spacemouse', 'dualsense' or 'mjgui'.")

    # Collect a fixed number of demonstrations, each with its own force log
    # and its own saved plot. A demonstration ends when the device sends a
    # reset signal (input2action returns None); the next one starts right
    # after, until num_demos have been completed.
    demo_idx = 0
    all_results = []

    print(f"Collecting {args.num_demos} demonstrations. Force plots will be saved to '{args.force_plot_dir}'.")

    while demo_idx < args.num_demos:
        print(f"\n--- Starting demonstration {demo_idx + 1}/{args.num_demos} ---")

        # Reset the environment
        obs = env.reset()

        # Fresh force log for this demonstration
        force_log = new_force_log()

        # Setup rendering
        cam_id = 0
        num_cam = len(env.sim.model.camera_names)
        env.render()

        # Initialize variables that should be maintained between resets
        last_grasp = 0

        # Initialize device control
        device.start_control()
        all_prev_gripper_actions = [
            {
                f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                for robot_arm in robot.arms
                if robot.gripper[robot_arm].dof > 0
            }
            for robot in env.robots
        ]

        task_success = False

        # Loop until we get a reset from the input device or the task completes
        while True:
            start = time.time()

            # Set active robot
            active_robot = env.robots[device.active_robot]

            # Get the newest action
            input_ac_dict = device.input2action()

            # If action is none, then this a reset so we should break
            if input_ac_dict is None:
                break

            action_dict = deepcopy(input_ac_dict)  # {}
            # set arm actions
            for arm in active_robot.arms:
                if isinstance(active_robot.composite_controller, WholeBody):  # input type passed to joint_action_policy
                    controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
                else:
                    controller_input_type = active_robot.part_controllers[arm].input_type

                if controller_input_type == "delta":
                    action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                elif controller_input_type == "absolute":
                    action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                else:
                    raise ValueError

            # Maintain gripper state for each robot but only update the active robot with action
            env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
            env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
            env_action = np.concatenate(env_action)
            for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

            env.step(env_action)
            env.render()

            # Log forces for this timestep (after env.step, so sim.data.contact
            # reflects the state that was just stepped to)
            log_contact_forces(env, force_log)

            # Track task success as we go. The env is created with
            # ignore_done=True (so env.step()'s own done flag never fires
            # and cuts your teleop control short mid-motion), which means
            # success has to be polled and acted on here explicitly rather
            # than relied on to end the episode by itself.
            #
            # Call _check_success() on the unwrapped base env rather than
            # on the VisualizationWrapper, to avoid depending on the
            # wrapper's attribute-forwarding for this.
            step_count = len(force_log["t"])
            height_above_table = (
                base_env.sim.data.body_xpos[base_env.cube_free_body_id][2]
                - base_env.table_offset[2]
            )
            if step_count % 20 == 0:
                # Throttled debug print so you can watch the actual lift
                # height approach the 0.04 m success threshold in real time.
                print(f"  step {step_count}: height above table = {height_above_table:.4f} m (need > 0.04 m)")

            if base_env._check_success():
                task_success = True
                print(f"Success detected at step {step_count} (height={height_above_table:.4f} m) -- ending demonstration.")
                break

            # limit frame rate if necessary
            if args.max_fr is not None:
                elapsed = time.time() - start
                diff = 1 / args.max_fr - elapsed
                if diff > 0:
                    time.sleep(diff)

        # Demonstration ended (reset signal received) -- save its force plot
        plot_resistive_force_z(
            force_log,
            os.path.join(args.force_plot_dir, f"demo_{demo_idx + 1}_force.png"),
        )
        all_results.append({"demo": demo_idx + 1, "success": task_success, "n_steps": len(force_log["t"])})
        print(f"Demonstration {demo_idx + 1} finished. Success: {task_success}. Steps logged: {len(force_log['t'])}.")

        demo_idx += 1

    print("\n=== Summary ===")
    for r in all_results:
        print(r)
    n_success = sum(r["success"] for r in all_results)
    print(f"Success rate: {n_success}/{args.num_demos} = {100 * n_success / args.num_demos:.1f}%")