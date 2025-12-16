from modules.ik import inverse_kinematics
import modules.util as util
import numpy as np
import time
from pymycobot import MyCobot280
import threading

# mc = MyCobot280("/dev/ttyAMA0", baudrate=1000000)
mc = MyCobot280("/dev/ttyAMA0", baudrate=1000000)
mc.set_fresh_mode(0)
mc2 = MyCobot280("/dev/ttyAMA0", baudrate=1000000)

action = {
    "rotate": [],
    "home": [24.997305570185226, -49.63050036158377, -110.44722133723108, 70.07767431067634, 3.950022876936672e-05, 0],
    "home2": [
        34.879476733238256,
        -49.057293065782225,
        -109.83631226701569,
        68.89365481510154,
        -2.5010854119063205e-05,
        0,
    ],
    "home3": [
        32.23013184139671,
        -45.77797295655054,
        -123.21291804054101,
        78.99091355242291,
        -1.5380681461781622e-05,
        0,
    ],
}


def move_thread(instr, mc):
    """
    Thread to move arm
    """
    print("thread 1")
    processed = False
    while not processed:
        try:
            mc.send_angles(action[instr], 70)
        except Exception:
            processed = True
        else:
            processed = True
    print("thread 1 finished")


def home_thread(mc2):
    """
    Thread to check the status of arm
    """
    time.sleep(0.1)
    print("thread 2")
    processed = False
    while not processed:
        try:
            mc2.send_angles(action["home"], 70)
        except Exception:
            pass
        else:
            processed = True
    print("thread 2 finished")


def move(instr, mc, mc2):
    """
    Move arm based on instruction
    """
    start = time.perf_counter()
    thread1 = threading.Thread(target=move_thread, args=(instr, mc))
    thread2 = threading.Thread(target=home_thread, args=(mc2,))

    thread1.start()
    thread2.start()

    thread2.join()
    thread1.join()
    print(f"Threads join: {instr}")
    print(f"Time: {time.perf_counter() - start}")


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

    action["rotate"] = soln

    print(f"Solution: {soln}")
    print(f"Error: {err}")

    move("rotate", mc, mc2)
