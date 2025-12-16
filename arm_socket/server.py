import socket
import ast
from pymycobot import MyCobot280
import numpy as np
import time
import threading

HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 5000  # Arbitrary port >1024

# Angles for each position
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
        25.162762448642738,
        -61.42368721486929,
        -92.69525712948317,
        64.11891164330632,
        2.046404914793443e-05,
        0,
    ],
    "left": [
        38.43664745258475,
        -58.79594585322086,
        -104,
        72.72079251924849,
        -1.1450604527398737e-05,
        0,
    ],
    "right": [
        16.942364909303294,
        -61.42429237195168,
        -107,
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
    Thread to move arm to position
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
    Thread to move arm back to home position.
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
    Move arm sequence based on instruction
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

    # Connect to arm
    mc = MyCobot280("/dev/ttyAMA0", baudrate=1000000)
    mc.set_fresh_mode(0)
    mc2 = MyCobot280("/dev/ttyAMA0", baudrate=1000000)

    mc.send_angles(action["home"], 30)

    # Create socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    print(f"Server listening on {HOST}:{PORT}")

    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    while True:
        # Wait for message from client
        data = conn.recv(1024)
        if not data:
            break

        decoded = data.decode()
        lst = ast.literal_eval(decoded)
        print(f"Client says: {lst}")
        print(f"Typeof: {type(lst)}")

        data = lst

        # Decode message into rotations and movement
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

            # Drop the tetris block
            mc.send_angles(action["home3"], 100)
            move("drop", mc, mc2)

            # Tell client it is finished with moving
            conn.sendall(("finished").encode())

    conn.close()
    server_socket.close()


if __name__ == "__main__":
    main()
