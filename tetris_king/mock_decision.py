"""
ROS node to move myCobot 280 arm
"""

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
import numpy as np


class MockDecision(Node):
    """
    Class to initialize and establish subscriber/publisher interaction
    """

    def __init__(self):
        super().__init__("mock_decision")

        # Publisher to action
        self.publisher = self.create_publisher(String, "/action", 10)

        self.timer = self.create_timer(5, self.timer_callback)

        self.button = True

    def timer_callback(self):
        """
        Publishes random action every second
        """
        tasks = ["rotate", "hold", "left", "right", "down"]
        # random_number = int(np.random.uniform(low=0, high=4, size=1)[0])

        msg = String()
        msg.data = tasks[int(self.button)]

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = MockDecision()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
