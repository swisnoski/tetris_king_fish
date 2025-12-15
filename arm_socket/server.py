import socket
import ast
from pymycobot import MyCobot280
import numpy as np
import time
import threading

HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 5000  # Arbitrary port >1024

action = {
    "rotate": [
        18.70698233348091,
        -62.812084700082664,
        -102.3756482611303,
        75.18775242605099,
        -7.779765331039758e-06,
        0,
    ],
    "drop": [
        20.452150451187027,
        -62.76464772691289,
        -113.34551195680278,
        86.11017243216526,
        -4.06473253008484e-06,
        0,
    ],
    "left": [
        22.224245716142704,
        -64.16523573718025,
        -105.9438513284137,
        80.10907275844538,
        1.2220142031219308e-05,
        0,
    ],
    "right": [
        16.499338057714052,
        -64.30823824197729,
        -105.83911545445714,
        80.1473309816758,
        2.1453826671196892e-05,
        0,
    ],
    "home": [
        19.38766022032492,
        -53.72804223828324,
        -110.45016002179023,
        74.17824998575445,
        -2.1409152024860533e-05,
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


def main():
    """
    Starts a server and moves arm based on messages receieved through socket TCP
    """

    mc = MyCobot280("/dev/ttyAMA0", baudrate=1000000)
    mc.set_fresh_mode(0)
    mc2 = MyCobot280("/dev/ttyAMA0", baudrate=1000000)

    mc.send_angles(action["home"], 30)

    """data = [3, 3]

    rotations = data[0]
    movement = data[1]

    # Decide whether movement is left or right
    direction = None
    if movement < 0:
        direction = "left"
    elif movement > 0:
        direction = "right"

    # Move arm based on instructions
    if not mc.is_moving():
        # Rotate
        if rotations != 0:
            for _ in range(int(rotations)):
                move("rotate", mc, mc2)
        # Move
        if direction is not None:
            for _ in range(int(abs(movement))):
                move(direction, mc, mc2)
    move("drop", mc, mc2)"""

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    print(f"Server listening on {HOST}:{PORT}")

    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    while True:
        data = conn.recv(1024)
        if not data:
            break

        decoded = data.decode()
        lst = ast.literal_eval(decoded)
        print(f"Client says: {lst}")
        print(f"Typeof: {type(lst)}")

        data = lst

        rotations = data[0]
        movement = data[1]

        # Decide whether movement is left or right
        direction = None
        if movement < 0:
            direction = "left"
        elif movement > 0:
            direction = "right"

        # Move arm based on instructions
        if not mc.is_moving():
            # Rotate
            if rotations != 0:
                for _ in range(int(rotations)):
                    move("rotate", mc, mc2)
            # Move
            if direction is not None:
                for _ in range(int(abs(movement))):
                    move(direction, mc, mc2)
        move("drop", mc, mc2)

    conn.close()
    server_socket.close()


if __name__ == "__main__":
    main()
