"""
Module with all functions relating to the Forward Kinematics of the robot
"""

import modern_robotics as mr
import numpy as np

M = np.array(
    [
        [0, 0, 1, 0.0436],
        [-1, 0, 0, -0.06639],
        [0, -1, 0, 0.41114],
        [0, 0, 0, 1],
    ]
)

S_list = [
    np.array([0, 0, 1, 0, 0, 0]),
    np.array([0, -1, 0, 0.13156, 0, 0]),
    np.array([0, -1, 0, 0.24196, 0, 0]),
    np.array([0, -1, 0, 0.33796, 0, 0]),
    np.array([0, 0, 1, -0.06639, 0, 0]),
    np.array([1, 0, 0, 0, 0.41114, 0.06639]),
]


def forward_kinematics(theta_list):
    """
    Given a list of angles (theta_list) in radians of the robot, return the
    spatial position of the robot mapping to the given theta_list.

    NOTE: if angles are not within bound, the function will return False
        theta 1: -168 ~ +168
        theta 2: -135 ~ +135
        theta 3: -150 ~ +150
        theta 4: -145 ~ +145
        theta 5: -165 ~ +165
        theta 6: -180 ~ +180

    Args:
        theta_list: [list] (rad)
            0: theta 1
            1: theta 2
            2: theta 3
            3: theta 4
            4: theta 5
            5: theta 6

    Returns:
        Angles within bounds: (tuple) (m)
            0: x displacement,
            1: y displacement,
            2: z displacement
        Angles out of bounds: False
    """
    no_of_joints = len(theta_list)
    T = []
    for i in range(0, no_of_joints):
        T.append(
            mr.MatrixExp6(mr.VecTose3(S_list[i]) * theta_list[i]),
        )

    result = M
    for i in range(0, no_of_joints):
        result = T[no_of_joints - 1 - i] @ result

    return list(result[0:3, 3]), result
