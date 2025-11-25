"""
ROS node to move myCobot 280 arm
"""

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray

from pymycobot import MyCobot280
import numpy as np
import time
from tetris_king.modules.ik import inverse_kinematics
import tetris_king.modules.util as util


class TetrisArm(Node):
    """
    Class to initialize and establish subscriber/publisher interaction
    """

    def __init__(self):
        super().__init__("tetris_arm")
        # Connect to mycobot arm
        self.get_logger().info("Connecting to arm...")
        self.mc = MyCobot280("/dev/ttyAMA0", 1000000)
        self.get_logger().info("Connected to arm")

        # Reset arm to location [0, 0, 0, 0, 0, 0]
        self.mc.send_angles([0, 0, 0, 0, 0, 0], 30)
        self.get_logger().info("Reset arm")

        # Subscribe to desired pose
        self.subscription = self.create_subscription(
            Float32MultiArray, "/desired_pose", self.listener_callback, 10
        )

    def listener_callback(self, msg):
        """
        Get desired pose from topic, perform IK, then move arm
        """
        desired_ee = np.array(msg.data)
        self.get_logger().info(f"Desired pose: {desired_ee}")

        if not self.mc.is_moving():
            print("Starting IK")

            self.get_logger().info(f"Current angles: {self.mc.get_angles()}")

            # track IK time for future optimization
            start_time = time.perf_counter()

            # IK
            soln, err = inverse_kinematics(
                util.deg2rad(self.mc.get_angles()), desired_ee, tol=0.01
            )
            soln = util.rad2deg(soln)
            soln[5] = 0

            ik_done_time = time.perf_counter()

            self.get_logger().info(
                f"IK time: {(ik_done_time - start_time):.6f} seconds"
            )

            # log results
            self.get_logger().info(f"Error: {err}")
            self.get_logger().info(f"Soln: {soln}")

            # Move arm
            if err < 0.01:
                arm_move_start = time.perf_counter()
                self.mc.send_angles(soln, 60)
                arm_move_end = time.perf_counter()
                self.get_logger().info(
                    f"Move time: {(arm_move_end - arm_move_start):.6f} seconds"
                )
            else:
                self.get_logger().info("Error too high!")

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            self.get_logger().info(f"Full process time: {elapsed_time:.6f} seconds")


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
