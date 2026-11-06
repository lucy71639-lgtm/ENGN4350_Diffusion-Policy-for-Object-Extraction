#This file contains functions necessary to control the robot manipulator in the robosuite enviornment. 
#Control is compatible with a 7DoF manipulator with a 1DoF gripper, such as the Panda robot. 
#Control in implemented in joint space, with the option to control the gripper separately.

import numpy as np

def move_to_hidden(target_joints, grip, env, k, tol=1e-3, max_steps=200, log_fn=None):
    global obs
    robot = env.robots[0]
    joint_idx = robot._ref_joint_pos_indexes
    n_joints = len(joint_idx)

    for step in range(max_steps):
        q_current = env.sim.data.qpos[joint_idx].copy()
        delta = target_joints - q_current

        if np.linalg.norm(delta) < tol:
            break  # converged, stop early

        q_desired = q_current + k * delta

        action = np.zeros(env.action_dim)
        action[:n_joints] = q_desired
        action[-1] = grip

        obs, reward, done, info = env.step(action)

    if log_fn is not None:
            log_fn()

    #else:
        #print(f"Warning: move_to did not converge, final error = {np.linalg.norm(delta):.5f}")

    #print('actual pos', env.sim.data.qpos[joint_idx].copy())
    
    
def initialize_robot_2(env):
    #VISUALIZE THE ROBOT IN ITS CURRENT POSITION. THE ROBOT IS ALREADY IN SOME POSITION, THIS FUNCTION JUST RENDERS IT.
    #THE CREATION OF THE ACTION VECTOR IS TECHNICALLY REDUNDANT, BUT IT IS NECESSARY TO CALL ENV.STEP() TO UPDATE THE RENDERING.
    global obs
    robot = env.robots[0]
    joint_idx = robot._ref_joint_pos_indexes
    n_joints = len(joint_idx)
    grip = -1 #gripper is open

    current_pos = env.sim.data.qpos[joint_idx].copy()

     # Build action vector
    action = np.zeros(env.action_dim)
    action[:n_joints] = current_pos      # arm joints
    action[-1] = grip                  # gripper control

    print('actual pos', env.sim.data.qpos[joint_idx].copy())
        
    # Send command
    obs, reward, done, info = env.step(action)
    
    # Render
    env.render()

def close_gripper(env, log_fn=None):
    global obs
    robot = env.robots[0]
    joint_idx = robot._ref_joint_pos_indexes
    n_joints = len(joint_idx)


    for _ in range(20):  
        # Build action vector
        action = np.zeros(env.action_dim)
        action[:n_joints] = env.sim.data.qpos[joint_idx].copy()  # Keep current joint positions
        action[-1] = 1  # Close gripper

        # Send command
        obs, reward, done, info = env.step(action)

        # Render
        #env.render()

        if log_fn is not None:
            log_fn()