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
        16.942364909303294,
        -61.42429237195168,
        -110.60229048120522,
        82.02660179430062,
        -7.683645828234532e-06,
        0,
    ],
    "drop": [
        20.679581038060547,
        -61.08386261684743,
        -115.37226054332203,
        86.45613451999829,
        -5.657667012159038e-06,
        0,
    ],
    "left": [40.66641389903865, -59.80598033221904, -101.19645342513901, 71.00246810420892, -1.3729825531349562e-05, 0],
    "right": [
        16.942364909303294,
        -61.42429237195168,
        -110.60229048120522,
        82.02660179430062,
        -7.683645828234532e-06,
        0,
    ],
    "home": [
        19.38766022032492,
        -48.53838,
        -111.69046,
        70.22889,
        0,
        0,
    ],
    "home2": [
        34.879476733238256,
        -49.057293065782225,
        -109.83631226701569,
        68.89365481510154,
        -2.5010854119063205e-05,
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
            mc.send_angles(action[instr], 100)
        except Exception:
            processed = True
        else:
            processed = True
    print("thread 1 finished")


def home_thread(instr, mc2):
    """
    Thread to check the status of arm
    """
    time.sleep(0.1)
    print("thread 2")
    processed = False
    while not processed:
        try:
            if instr == "right":
                mc2.send_angles(action["home2"], 70)
            else:
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
    thread2 = threading.Thread(target=home_thread, args=(instr, mc2))

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

            if direction == "right":
                mc.send_angles(action["home2"], 100)
            # Move
            if direction is not None:
                for _ in range(int(abs(movement))):
                    move(direction, mc, mc2)

            if direction == "right":
                mc.send_angles(action["home"], 100)

            conn.sendall(("finished").encode())

    conn.close()
    server_socket.close()


if __name__ == "__main__":
    main()
