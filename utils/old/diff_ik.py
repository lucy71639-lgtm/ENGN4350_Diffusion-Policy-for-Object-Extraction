# Partially based on 
# Kevin Zakkas' mjctrl github (https://github.com/kevinzakka/mjctrl/blob/main/diffik_nullspace.py) AND
# Robosuite IK Controller (https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/controllers/parts/arm/ik.py)


import numpy as np
import mujoco
import mujoco.viewer
import time

def diff_ik(
    sim,
    q_start,
    target_pos,
    joint_indexes,
    dof_indexes,
    target_quat,          # (w, x, y, z) or None to ignore orientation
    site_name="gripper0_grip_site",
    max_iters=200,
    tol=1e-4,
    damping=1e-2,
    step_size=1.0,
    joint_limits=None,         # (lower, upper) arrays, len 7, or None
):
    # Integration timestep in seconds. This corresponds to the amount of time the joint
    # velocities will be integrated for to obtain the desired joint positions.
    integration_dt: float = 0.1

    # Damping term for the pseudoinverse. This is used to prevent joint velocities from
    # becoming too large when the Jacobian is close to singular.
    damping: float = 1e-4

    # Gains for the twist computation. These should be between 0 and 1. 0 means no
    # movement, 1 means move the end-effector to the target in one integration step.
    Kpos: float = 0.95
    Kori: float = 0.95

    # Whether to enable gravity compensation.
    gravity_compensation: bool = True

    # Simulation timestep in seconds.
    dt: float = 0.002

    # Nullspace P gain.
    Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])

    # Maximum allowable joint velocity in rad/s.
    max_angvel = 0.785

    model = sim.model
    data = sim.data
    site_id = model.site(site_name).id
    
    qpos_idx = joint_indexes
    dof_idx = dof_indexes 

    q0 = q_start
    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    eye = np.eye(model.nv)
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    # Spatial velocity (aka twist).
    dx = target_pos - data.site(site_id).xpos
    twist[:3] = Kpos * dx / integration_dt
    mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
    mujoco.mju_negQuat(site_quat_conj, site_quat)
    mujoco.mju_mulQuat(error_quat, target_quat, site_quat_conj)
    mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
    twist[3:] *= Kori / integration_dt
    
    # Jacobian.
    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
    
    # Damped least squares.
    dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, twist)
    
    # Nullspace control biasing joint velocities towards the home configuration.
    dq += (eye - np.linalg.pinv(jac) @ jac) @ (Kn * (q0 - data.qpos[dof_indexes]))
    
    # Clamp maximum joint velocity.
    dq_abs_max = np.abs(dq).max()
    if dq_abs_max > max_angvel:
        dq *= max_angvel / dq_abs_max
    
    # Integrate joint velocities to obtain joint positions.
    q = data.qpos.copy()  # Note the copy here is important.
    mujoco.mj_integratePos(model, q, dq, integration_dt)
    np.clip(q, *model.jnt_range.T, out=q)
    print(q)
    
    return q
