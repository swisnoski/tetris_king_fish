from modules.ik import inverse_kinematics
import modules.util as util
import numpy as np
import time
from pymycobot import MyCobot280

# POS 1

print("Starting IK")

success_count = 0
count = 0

total_time = 0

# for i in range(2):

start_time = time.perf_counter()

UP = [0.21, 0.005, 0.045]
DOWN = [0.19, 0, 0.045]
LEFT = [0.2, -0.013, 0.045]
RIGHT = [0.2, 0.01, 0.045]
HOME = [0.2, 0, 0.06]

pos_list = [HOME, UP, DOWN, LEFT, RIGHT]

for pos in pos_list:
    desired_ee = pos
    soln, err = inverse_kinematics(
        util.deg2rad([0, 0, 0, 0, 0, 0]), desired_ee, tol=0.001
    )

    print(f"Solution (radians): {soln}")

    soln = util.rad2deg(soln)
    soln[5] = 0

    end_time = time.perf_counter()

    print(f"Solution: {soln}")
    print(f"Error: {err}")

    elapsed_time = end_time - start_time
    total_time = total_time + elapsed_time

    if err < 0.01:
        success_count = success_count + 1
    count = count + 1

    avg_time = total_time / count
    success_rate = success_count / count

    print(f"Success rate: {success_rate}")
    print(f"Average time: {avg_time}")

    print()
    print()
    print()
