"""
ROS node to move myCobot 280 arm
"""

import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from pymycobot import MyCobot280
import numpy as np
import time
from tetris_king.modules.ik import inverse_kinematics
import tetris_king.modules.util as util


class TetrisArm(Node):
    """
    Class to initialize and establish subscriber/publisher interaction
    """

    action = {
        "rotate": [
            19.390781149202574,
            -65.71144706580844,
            -106.36085117234512,
            82.07242236680308,
            0.0005176930105520582,
            0,
        ],
        "hold": [],
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
        "down": [],
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
        self.mc = MyCobot280("/dev/ttyAMA0", 1000000)
        self.get_logger().info("Connected to arm")

        # Reset arm to location [0, 0, 0, 0, 0, 0]
        self.mc.send_angles(self.action["home"], 30)
        self.mc.set_fresh_mode(1)
        self.get_logger().info("Reset arm")

        # Subscribe to action
        self.subscription = self.create_subscription(
            String, "/action", self.listener_callback, 10
        )

    def listener_callback(self, msg):
        """
        Get desired pose from topic, perform IK, then move arm
        """
        start_time = time.perf_counter()

        # Get desired end-effector based on action
        desired_action = msg.data
        desired_ee = self.action[desired_action].copy()

        # Move arm down and up
        if not self.mc.is_moving():
            self.mc.sync_send_angles(self.action["left"], 100, timeout=0.3)
            self.mc.sync_send_angles(self.action["home"], 100, timeout=0.3)
            self.mc.sync_send_angles(self.action["right"], 100, timeout=0.3)
            self.mc.sync_send_angles(self.action["home"], 100, timeout=0.3)
            self.mc.sync_send_angles(self.action["left"], 100, timeout=0.3)
            self.mc.sync_send_angles(self.action["home"], 100, timeout=0.3)
            self.mc.sync_send_angles(self.action["right"], 100, timeout=0.3)
            self.mc.sync_send_angles(self.action["home"], 100, timeout=0.3)

        end_time = time.perf_counter()
        self.get_logger().info(f"Elapsed Time: {end_time - start_time}")


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
