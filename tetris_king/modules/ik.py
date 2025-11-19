"""
Module with all functions relating to the Inverse Kinematics of the robot
"""

import modern_robotics as mr
from tetris_king.modules.fk import forward_kinematics
import numpy as np

M = np.array(
    [
        [0, 0, 1, 0.0436],
        [-1, 0, 0, -0.06639],
        [0, -1, 0, 0.41114],
        [0, 0, 0, 1],
    ]
)

S_list = np.transpose(
    np.array(
        [
            np.array([0, 0, 1, 0, 0, 0]),
            np.array([0, -1, 0, 0.13156, 0, 0]),
            np.array([0, -1, 0, 0.24196, 0, 0]),
            np.array([0, -1, 0, 0.33796, 0, 0]),
            np.array([0, 0, 1, -0.06639, 0, 0]),
            np.array([1, 0, 0, 0, 0.41114, 0.06639]),
        ]
    )
)


def line_search(theta, dtheta, p_desired, max_trials=10):
    alpha = 1.0
    for _ in range(max_trials):
        theta_new = theta + alpha * dtheta

        # Compute FK at new angles
        p_new, T = forward_kinematics(theta_new)
        error_new = np.array(
            [
                0,
                0,
                0,
                p_desired[0] - p_new[0],
                p_desired[1] - p_new[1],
                p_desired[2] - p_new[2],
            ]
        )
        err_norm = np.linalg.norm(error_new)

        # Compute FK at current angles for comparison
        p_current, T = forward_kinematics(theta)
        error_current = np.array(
            [
                0,
                0,
                0,
                p_desired[0] - p_current[0],
                p_desired[1] - p_current[1],
                p_desired[2] - p_current[2],
            ]
        )
        err_current_norm = np.linalg.norm(error_current)

        if err_norm < err_current_norm:
            return alpha  # Accept step size

        alpha *= 0.5  # Try smaller step

    return 0.0  # Step rejected — no improvement


def inverse_kinematics(
    current_angle_pos, desired_ee, tol=1e-3, max_iters=200, damping=5e-4
):
    """
    Given the current_angle_pos and desired_ee pose, return angles of the robot mapping to that
    desired end-effector transformation.

    In case where no such desired_ee exists, function will return False

    Args:
        current_angle_pos: [list] (radians)
            0: theta 1
            1: theta 2
            2: theta 3
            3: theta 4
            4: theta 5
            5: theta 6
        desired_ee: (tuple)
            0: x displacement
            1: y displacement
            2: z displacement
        tol: (float)
        max_iters: (int)
        damping: (float)

    Returns:
        Soln exists: [list], Boolean
            0: theta 1
            1: theta 2
            2: theta 3
            3: theta 4
            4: theta 5
            5: theta 6

    """
    best_ans = (None, 100)
    theta = current_angle_pos.copy()
    p_desired = np.array(desired_ee)
    joint_limits = [
        (-np.radians(168), np.radians(168)),
        (-np.radians(135), np.radians(135)),
        (-np.radians(150), np.radians(150)),
        (-np.radians(145), np.radians(145)),
        (-np.radians(165), np.radians(165)),
        (-np.radians(180), np.radians(180)),
    ]
    lower_bounds = np.array([low for low, _ in joint_limits])
    upper_bounds = np.array([high for _, high in joint_limits])

    for i in range(max_iters):
        # Compute forward kinematics position
        p_current, T = forward_kinematics(theta)

        # Compute linear error
        error = p_desired - np.array(p_current)

        # Computer Angular error
        z_current = T[0:3, 2]
        z_desired = np.array([0, 0, -1])

        # Orientation error using angle-axis (only constrain z-axis alignment)
        axis = np.cross(z_current, z_desired)
        sin_theta = np.linalg.norm(axis)
        cos_theta = np.dot(z_current, z_desired)
        theta_ = np.arctan2(sin_theta, cos_theta)

        if sin_theta > 1e-6:
            axis = axis / sin_theta
        else:
            axis = np.zeros(3)

        orientation_error = theta_ * axis

        # Full error
        error = np.array(
            [
                orientation_error[0],
                orientation_error[1],
                orientation_error[2],
                error[0],
                error[1],
                error[2],
            ]
        )
        err_norm = np.linalg.norm(error)

        if err_norm < best_ans[1]:
            best_ans = (theta.copy(), err_norm)

        if err_norm < tol:
            return theta, err_norm  # Success

        # Compute Jacobian and take position part
        Js = mr.JacobianSpace(S_list, theta)

        # Damped least squares inverse
        JT = Js.T
        J_damped_inv = JT @ np.linalg.inv(Js @ JT + (damping) * np.eye(6))

        dtheta = J_damped_inv @ error
        max_step = 2 * err_norm
        dtheta = np.clip(dtheta, -max_step, max_step)

        alpha = line_search(theta, dtheta, p_desired)
        if alpha > 0:
            theta_temp = theta + alpha * dtheta
            theta_temp = np.clip(theta_temp, lower_bounds, upper_bounds)
            if np.linalg.norm(theta - theta_temp) < 0.001:
                theta += np.random.uniform(-1, 1, size=theta.shape)
            else:
                theta = theta_temp
        else:
            theta += np.random.uniform(-1, 1, size=theta.shape)

    return best_ans[0], err_norm  # Did not converge
