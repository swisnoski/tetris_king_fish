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
        super().__init__("move_arm")
        self.mc = MyCobot280("/dev/ttyAMA0", 1000000)
        self.joint_angles_queue = []
        self.subscription = self.create_subscription(
            Float32MultiArray, "joint_angles", self.listener_callback, 10
        )
        self.publisher = self.create_publisher(Float32MultiArray, "current_angles", 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
        self.joint_angles_queue.append(list(msg.data))

    def timer_callback(self):
        msg = Float32MultiArray()
        msg.data = self.mc.get_angles()
        self.publisher.publish(msg)
        if not self.mc.is_moving() and len(self.joint_angles_queue) != 0:
            self.get_logger().info(
                (
                    f"Joint angle queue is {len(self.joint_angles_queue)} items"
                    f"long, moving to next location {self.joint_angles_queue[0]}"
                )
            )
            self.mc.send_angles(self.joint_angles_queue.pop(0), 30)


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
