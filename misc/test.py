from modules.ik import inverse_kinematics
import modules.util as util
import numpy as np
import time

# POS 1

print("Starting IK")

success_count = 0
count = 0

total_time = 0

# for i in range(2):

start_time = time.perf_counter()

desired_ee = [0.2, 0, 0.06]
soln, err = inverse_kinematics(util.deg2rad([0, 0, 0, 0, 0, 0]), desired_ee, tol=0.01)

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
