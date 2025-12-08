"""
ROS node to move myCobot 280 arm
"""

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Float32MultiArray

from pymycobot import MyCobot280
import numpy as np
import time
from tetris_king.modules.ik import inverse_kinematics
import tetris_king.modules.util as util

import threading


class TetrisArm(Node):
    """
    Class to initialize and establish subscriber/publisher interaction
    """

    action = {
        "rotate": [
            18.42945704984153,
            -64.09078549771468,
            -99.60140781108544,
            73.69215836314227,
            3.0031403880330304e-05,
            0,
        ],
        "drop": [
            20.120293863338166,
            -61.4062793708456,
            -112.01524476494939,
            83.4215353526735,
            -5.958138336667897e-06,
            0,
        ],
        "left": [
            22.224149230856394,
            -62.683497431891574,
            -106.61694234639404,
            79.30041567484217,
            2.1517009588529715e-05,
            0,
        ],
        "right": [
            16.499995231431722,
            -61.75319033436939,
            -107.2852511676767,
            79.0384707772517,
            -8.59293671689396e-06,
            0,
        ],
        "home": [
            19.38432522180462,
            -48.91106561440083,
            -109.46801740270132,
            68.37870373124753,
            0.0011162704024856528,
            0,
        ],
    }

    def __init__(self):
        super().__init__("tetris_arm")
        # Connect to mycobot arm
        self.get_logger().info("Connecting to arm...")
        self.mc = MyCobot280("/dev/ttyAMA0", baudrate=1000000)
        self.mc.set_free_mode(1)
        self.mc2 = MyCobot280("/dev/ttyAMA0", baudrate=1000000)
        self.get_logger().info("Connected to arm")

        # Reset arm to location [0, 0, 0, 0, 0, 0]
        self.mc.send_angles(self.action["home"], 30)
        self.get_logger().info("Reset arm")

        # Subscribe to action
        self.subscription = self.create_subscription(
            Float32MultiArray, "/action", self.listener_callback, 10
        )

    def listener_callback(self, msg):
        """
        Get desired pose from topic, perform IK, then move arm
        """
        # Extract information from topic into rotations and movement
        data = msg.data

        rotations = data[0]
        movement = data[1]

        # Decide whether movement is left or right
        direction = None
        if movement < 0:
            direction = "left"
        elif movement > 0:
            direction = "right"

        # Move arm based on instructions
        if not self.mc.is_moving():
            # Rotate
            if rotations != 0:
                for _ in range(int(rotations)):
                    self.move("rotate")
            # Move
            if direction is not None:
                for _ in range(int(abs(movement))):
                    self.move(direction)

        self.move("drop")

    def move_thread(self):
        """
        Thread to move arm
        """
        self.mc.sync_send_angles(self.action["rotate"], 100, timeout=0.1)

    def status_thread(self):
        """
        Thread to check the status of arm
        """
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < 0.5:
            print(f"Angle plans: {time.perf_counter()}")
        self.mc2.send_angles(self.action["home"], 100)

    def move(self, instr):
        """
        Move arm based on instruction
        """
        thread1 = threading.Thread(target=self.move_thread)
        thread2 = threading.Thread(target=self.status_thread)

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()
        time.sleep(0.5)


def main(args=None):
    rclpy.init(args=args)

    node = TetrisArm()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
