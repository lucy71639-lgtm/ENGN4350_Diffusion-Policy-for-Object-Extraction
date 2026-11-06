import numpy as np
import mujoco
from robosuite.utils.transform_utils import mat2quat, quat_multiply, quat_conjugate, quat2axisangle

def inverse_kin(
    sim,
    target_pos,
    joint_indexes,
    dof_indexes,
    target_quat=None,          # (w, x, y, z) or None to ignore orientation
    site_name="gripper0_grip_site",
    max_iters=200,
    tol=1e-4,
    damping=1e-2,
    step_size=1.0,
    joint_limits=None,         # (lower, upper) arrays, len 7, or None
        
):
    model = sim.model
    data = sim.data

    qpos_idx = joint_indexes
    dof_idx = dof_indexes

    site_id = model.site_name2id(site_name)
 
    for i in range(max_iters):
        sim.forward()

        cur_pos = data.get_site_xpos(site_name).copy()
        pos_err = target_pos - cur_pos
 
        if target_quat is not None:
            cur_mat = data.site_xmat[site_id].reshape(3, 3)
            cur_quat = np.zeros(4)
            # mju_mat2Quat equivalent via mujoco_py/mujoco bindings
            #from mujoco_py import functions as mjf
            #mjf.mju_mat2Quat(cur_quat, cur_mat.flatten())
            mujoco.mju_mat2Quat(cur_quat, cur_mat.flatten())
            ori_err = quat_diff_to_rotvec(target_quat, cur_quat)
            err = np.concatenate([pos_err, ori_err])
        else:
            err = pos_err

        #Check if the the effector is already in the target position (i.e error is less than the tolerance)
        if np.linalg.norm(err) < tol:
            return data.qpos[qpos_idx].copy(), True
 
        jacp = data.get_site_jacp(site_name).reshape(3, model.nv)
        if target_quat is not None:
            jacr = data.get_site_jacr(site_name).reshape(3, model.nv)
            J = np.vstack([jacp[:, dof_idx], jacr[:, dof_idx]])
        else:
            J = jacp[:, dof_idx]
 
        # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 err
        JJt = J @ J.T
        lam_sq = damping ** 2 * np.eye(JJt.shape[0])
        dq = J.T @ np.linalg.solve(JJt + lam_sq, err)
 
        q_new = data.qpos[qpos_idx] + step_size * dq
 
        if joint_limits is not None:
            lower, upper = joint_limits
            q_new = np.clip(q_new, lower, upper)
 
        data.qpos[qpos_idx] = q_new
 
    return data.qpos[qpos_idx].copy(), False
 

def quat_diff_to_rotvec(target_quat, cur_quat):
    """Small-angle orientation error between two quaternions (wxyz), as a rotvec."""
    # cur_quat_conj = [w, -x, -y, -z]
    w0, x0, y0, z0 = cur_quat
    cur_conj = np.array([w0, -x0, -y0, -z0])
    w1, x1, y1, z1 = target_quat
 
    # quaternion multiply: target * conj(cur)
    w = w1 * cur_conj[0] - x1 * cur_conj[1] - y1 * cur_conj[2] - z1 * cur_conj[3]
    x = w1 * cur_conj[1] + x1 * cur_conj[0] + y1 * cur_conj[3] - z1 * cur_conj[2]
    y = w1 * cur_conj[2] - x1 * cur_conj[3] + y1 * cur_conj[0] + z1 * cur_conj[1]
    z = w1 * cur_conj[3] + x1 * cur_conj[2] - y1 * cur_conj[1] + z1 * cur_conj[0]
 
    return 2 * np.array([x, y, z]) * np.sign(w)
