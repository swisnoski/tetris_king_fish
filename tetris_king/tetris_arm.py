"""
ROS node to move myCobot 280 arm
"""

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray

from pymycobot import MyCobot280


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
        self.get_logger().info(f"Desired pose: {msg.data}")

        if not self.mc.is_moving():
            pass


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
