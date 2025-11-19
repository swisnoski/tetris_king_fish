from pymycobot import MyCobot280
from modules.ik import inverse_kinematics
import modules.util as util
import time

print("Connecting to arm")

mc = MyCobot280("/dev/ttyAMA0", 1000000)

print("Connected")

# Gets the current angle of all joints
angles = mc.get_angles()

print(angles)

mc.send_angles([0, 0, 0, 0, 0, 0], 30)
print("Arm reset!")

print("Starting IK")

start_time = time.perf_counter()

desired_ee = [0.1, 0.15, 0.1]
soln, err = inverse_kinematics([0, 0, 0, 0, 0, 0], desired_ee, tol=0.01)
soln = util.rad2deg(soln)
soln[5] = 0

end_time = time.perf_counter()

print(err)
print(soln)
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")

mc.send_angles(soln, 30)
