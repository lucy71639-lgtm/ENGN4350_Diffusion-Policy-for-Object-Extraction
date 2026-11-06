'''
This is based on:  
Kevin Zakkas' mjctrl github (https://github.com/kevinzakka/mjctrl/blob/main/diffik_nullspace.py) AND
Robosuite IK Controller (https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/controllers/parts/arm/ik.py)
With AI assistance from Claude to help comprehend and apply the code. 
'''

import numpy as np
import mujoco
from robosuite.utils.transform_utils import mat2quat, quat_multiply, quat_conjugate, quat2axisangle

def solve_ik(
    sim,
    target_pos,
    joint_indexes,
    dof_indexes,
    target_quat=None,
    site_name="gripper0_grip_site",
    qpos_init=None,
    q_preferred=None,          # nominal "nice" posture to bias toward, e.g. Panda ready pose
    max_iters=500,
    tol=1e-3, #changed from 1e-4
    damping=1e-2,
    step_size=0.5,             # smaller default -- more stable near limits/singularities
    ori_weight=0.3,            # de-weight orientation vs. position (meters vs radians mismatch)
    nullspace_gain=2.0,
    joint_limits=None,
):
    model = sim.model
    data = sim.data
    qpos_idx = joint_indexes
    dof_idx = dof_indexes

    qpos_copy = data.qpos[qpos_idx].copy()  # Store a copy of the original qpos
    qvel_copy = data.qvel[dof_idx].copy()  # Store a copy of the original qvel

    if qpos_init is not None:
        data.qpos[qpos_idx] = qpos_init
    if q_preferred is None:
        q_preferred = qpos_init if qpos_init is not None else data.qpos[qpos_idx].copy()
    sim.forward()

    for i in range(max_iters):
        sim.forward()
        cur_pos = data.get_site_xpos(site_name).copy()
        pos_err = target_pos - cur_pos

        if target_quat is not None:
            cur_mat = data.get_site_xmat(site_name)
            cur_quat = mat2quat(cur_mat)
            if np.dot(target_quat, cur_quat) < 0:      # double-cover fix
                cur_quat = -cur_quat
            err_quat = quat_multiply(target_quat, quat_conjugate(cur_quat))
            ori_err = quat2axisangle(err_quat)
            err = np.concatenate([pos_err, ori_weight * ori_err])
        else:
            err = pos_err

        if np.linalg.norm(err) < tol:
            return data.qpos[qpos_idx].copy(), True

        jacp = data.get_site_jacp(site_name).reshape(3, model.nv)
        if target_quat is not None:
            jacr = data.get_site_jacr(site_name).reshape(3, model.nv)
            J = np.vstack([jacp[:, dof_idx], jacr[:, dof_idx]])
        else:
            J = jacp[:, dof_idx]

        JJt = J @ J.T
        lam_sq = damping ** 2 * np.eye(JJt.shape[0])
        dq_primary = J.T @ np.linalg.solve(JJt + lam_sq, err)

        # Secondary task: pull toward q_preferred, projected into the nullspace
        # so it doesn't fight the primary pose objective.
        q_cur = data.qpos[qpos_idx].copy() #CHANGED THIS TO COPY
        dq_secondary = nullspace_gain * (q_preferred - q_cur)
        N = nullspace_projection(J, damping)
        dq = dq_primary + N @ dq_secondary

        # Soft joint-limit avoidance, blended in before any hard clip
        if joint_limits is not None:
            lower, upper = joint_limits
            dq_limit = -joint_limit_avoidance_grad(q_cur, lower, upper)
            dq = dq + 0.1 * (N @ dq_limit)

        q_new = q_cur + step_size * dq
        if joint_limits is not None:
            q_new = np.clip(q_new, lower, upper)  # final safety net only

        data.qpos[qpos_idx] = q_new

    return data.qpos[qpos_idx].copy(), False

def joint_limit_avoidance_grad(q, lower, upper, margin=0.1):
    """Push q away from limits smoothly, zero effect far from them."""
    grad = np.zeros_like(q)
    for i in range(len(q)):
        range_i = upper[i] - lower[i]
        dist_lower = q[i] - lower[i]
        dist_upper = upper[i] - q[i]
        if dist_lower < margin * range_i:
            grad[i] = (margin * range_i - dist_lower) / (margin * range_i)
        elif dist_upper < margin * range_i:
            grad[i] = -(margin * range_i - dist_upper) / (margin * range_i)
    return grad

def nullspace_projection(J, damping=1e-2):
    """Projector onto the nullspace of J (damped, for stability)."""
    J_pinv = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(J.shape[0]))
    return np.eye(J.shape[1]) - J_pinv @ J