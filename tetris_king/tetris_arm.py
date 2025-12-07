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
            13.579377778196362,
            -62.428506532852964,
            -107.047695372942,
            79.47642858191479,
            0.00011509651056966509,
            0,
        ],
        "right": [
            13.574220868342843,
            -63.007718209874454,
            -105.27181107181333,
            78.27925744871303,
            0.0006085890561915545,
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
            self.mc.sync_send_angles(self.action["right"], 100, timeout=0.3)
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
