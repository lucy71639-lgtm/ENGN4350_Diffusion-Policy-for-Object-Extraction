#Structure is AI generated. Template for running a sweep of IK - PD control trials in robosuite.

import numpy as np
import imageio
import robosuite as suite
#from robosuite.controllers import load_controller_config

from utils.inverse_kin_nullspace import solve_ik
from utils.ik_nullspace_return import solve_ik_ret

from utils.move import initialize_robot_2, move_to_hidden, close_gripper

#import helper functions from robosuite. Not all of these are used in this script, but they are kept for reference. 
from robosuite.utils.transform_utils import mat2euler, euler2mat, mat2quat, quat_multiply, convert_quat, axisangle2quat, quat2mat
import numpy as np

from robosuite.controllers import load_composite_controller_config

# ---------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------
N_TRIALS = 100
SAVE_FAILURE_IMAGES = False        # set to True to save images of failed trials
FAILURE_IMG_DIR = "ik_failures"

#-----------------------------------------------------------
# Setup Controller
#-----------------------------------------------------------
controller_config = load_composite_controller_config(robot="Panda")

arm_key = "right"  # confirm against the printed dict structure

lower_limits =  np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
upper_limits = np.array([2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

arm_cfg = controller_config["body_parts"][arm_key]
arm_cfg["type"] = "JOINT_POSITION"
arm_cfg["input_type"] = "absolute"
arm_cfg["input_min"] = -1
arm_cfg["input_max"] = 1
arm_cfg["output_min"] = lower_limits   # per-joint list, from Step 1
arm_cfg["output_max"] = upper_limits
arm_cfg["qpos_limits"] = [lower_limits, upper_limits]  # redundant safety clip, matches same range

env = suite.make(
    env_name="Lift",
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=False,           # keep off for speed during the sweep
    has_offscreen_renderer=True,  # needed to grab images on failure
    use_camera_obs=False,         # you don't need obs images every step, only on failure
    camera_names="frontview",
    ignore_done=True,
    control_freq=20,
)

results = []  # list of dicts: {"trial": i, "success": bool}

# ---------------------------------------------------------
# 2. Trial loop
# ---------------------------------------------------------
for trial in range(N_TRIALS):
    obs = env.reset()   # <-- new random cube pose each time (placement_initializer)
    success = False
    #final_ik_qpos = None   # store what your IK *asked for* on the last step

    #---------------------------------------------------------
    # Setup Variables for this trial
    #---------------------------------------------------------
    robot = env.robots[0]

    joint_idx = robot._ref_joint_pos_indexes
    dof_idx = robot._ref_joint_vel_indexes

    q_start = robot.sim.data.qpos[joint_idx].copy()

    #Min-Max Joint Ranges for Panda Robot. These are used to ensure that the robot does not exceed its joint limits during motion planning.
    joint_ranges = robot.sim.model.jnt_range[joint_idx]
    lower = joint_ranges[:, 0]
    upper = joint_ranges[:, 1]

    #Locate the Cube
    obj_pos = obs["cube_pos"].copy()
    obj_quat = obs["cube_quat"].copy()

    target_pos = obj_pos
    eef_pos = obs["robot0_eef_pos"].copy()

    #starting position of the robot's end effector, which defaults to a top-down orientation. This is used as a reference for the target orientation of the end effector.
    down_quat =obs["robot0_eef_quat"].copy()
    down_mat = quat2mat(down_quat)

    cube_mat = env.sim.data.get_body_xmat("cube_main").copy()  # (3,3)
    roll, pitch, yaw = mat2euler(cube_mat) 
    yaw = yaw % (np.pi / 2)

    yaw_quat = axisangle2quat(np.array([0, 0, yaw]))
    target_quat = quat_multiply(yaw_quat, down_quat)

    #---------------------------------------------------------
    # Solve IK for the target pose
    #--------------------------------------------------------
    q_goal, ok = solve_ik_ret(
    sim=env.sim,
    target_pos=target_pos,
    joint_indexes=joint_idx,
    dof_indexes=dof_idx, 
    target_quat=target_quat,
    site_name="gripper0_right_grip_site",
    qpos_init=q_start, #initial guess for the IK solver. This is set to the current robot configuration.
    q_preferred=q_start, #bias the configuration towards the initial robot configuration.
    joint_limits=(lower, upper),
    )
    
    #------------------------------------------------
    move_to_hidden(q_goal, -1, env=env, k=0.1, max_steps=200)
    close_gripper(env)
    move_to_hidden(q_start,grip= 1, env=env, k=1, max_steps=200)

    success = env._check_success()
    print("Trial", trial, "Success:", success)
    results.append({"trial": trial, "success": success})

    # -----------------------------------------------------
    # 3. On failure, visualize
    # -----------------------------------------------------
    if not success and SAVE_FAILURE_IMAGES:
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

# ---------------------------------------------------------
# 4. Report
# ---------------------------------------------------------
n_success = sum(r["success"] for r in results)
print(f"Success rate: {n_success}/{N_TRIALS} = {100*n_success/N_TRIALS:.1f}%")