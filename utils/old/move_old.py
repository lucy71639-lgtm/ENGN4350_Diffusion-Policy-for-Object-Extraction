import numpy as np

#THESE ARE OLD VERSIONS OF THE MOVE FUNCTIONS, WHICH ARE NOW IN move.py. THEY ARE KEPT HERE FOR REFERENCE, BUT ARE NOT USED IN THE MAIN NOTEBOOK.

def move_to(target_joints, grip, env, k): #k determines number of steps to reach target, smaller k = more steps
    global obs
    robot = env.robots[0]
    joint_idx = robot._ref_joint_pos_indexes
    n_joints = len(joint_idx)

    for _ in range(50):
        q_current = env.sim.data.qpos[joint_idx].copy()

        # Error in joint space
        delta = target_joints - q_current
        #print("current", q_current)
        #print("target", target_joints)
        #print("delta", delta)
        # Desired joint positions (smoothed toward target)
        q_desired = q_current + k * delta

        # Build action vector
        action = np.zeros(env.action_dim)
        action[:n_joints] = q_desired      # arm joints
        action[-1] = grip                  # gripper

        # Send command
        obs, reward, done, info = env.step(action)

        # Render
        env.render()


def move_to_2(target_joints, grip, env, k, tol=1e-3, max_steps=200):
    global obs
    robot = env.robots[0]
    joint_idx = robot._ref_joint_pos_indexes
    n_joints = len(joint_idx)

    for step in range(max_steps):
        q_current = env.sim.data.qpos[joint_idx].copy()
        #print("current", q_current)
        #print("target", target_joints)
        delta = target_joints - q_current
        #print("delta", delta)

        if np.linalg.norm(delta) < tol:
            break  # converged, stop early

        q_desired = q_current + k * delta

        action = np.zeros(env.action_dim)
        action[:n_joints] = q_desired
        action[-1] = grip

        obs, reward, done, info = env.step(action)
        env.render()
    else:
        print(f"Warning: move_to did not converge, final error = {np.linalg.norm(delta):.5f}")


def initialize_robot(env, q_init):
    # Reset the environment and visualize. Goes AFTER IK
    global obs
    robot = env.robots[0]
    joint_idx = robot._ref_joint_pos_indexes
    n_joints = len(joint_idx)

    # Reset the environment to get initial observation
    #obs = env.reset()

    print('input pos', q_init)

    # Build action vector
    action = np.zeros(env.action_dim)
    action[:n_joints] = q_init      # arm joints
    action[-1] = -1                  # gripper is open

    print('actual pos', env.sim.data.qpos[joint_idx].copy())
    
    # Send command
    obs, reward, done, info = env.step(action)
    
    # Render
    env.render()