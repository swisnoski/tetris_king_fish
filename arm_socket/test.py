from modules.ik import inverse_kinematics
import modules.util as util
import numpy as np
import time
from pymycobot import MyCobot280

# mc = MyCobot280("/dev/ttyAMA0", baudrate=1000000)

# POS 1

print("Starting IK")

# for i in range(2):

start_time = time.perf_counter()

HOME = [0.2, 0, 0.06]

while True:

    x = input("x: ")
    y = input("y: ")
    z = input("z: ")

    desired_ee = [float(x), float(y), float(z)]
    soln, err = inverse_kinematics(
        util.deg2rad([0, 0, 0, 0, 0, 0]), desired_ee, tol=0.001
    )

    print(f"Solution (radians): {soln}")

    soln = util.rad2deg(soln)
    soln[5] = 0

    print(f"Solution: {soln}")
    print(f"Error: {err}")
