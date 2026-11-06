#!/usr/bin/env python3
"""
Mecanum inverse kinematics for AURIGA.

Subscribes: /cmd_vel                (geometry_msgs/Twist, body frame)
Publishes : /wheel_velocity_controller/commands
                                    (std_msgs/Float64MultiArray, rad/s)
            order: front_left, front_right, rear_left, rear_right

Wheel equations for an X-configuration mecanum (rollers at 45 deg, front-left
and rear-right sharing one chirality):

    w_fl = ( vx - vy - (lx + ly) * wz ) / r
    w_fr = ( vx + vy + (lx + ly) * wz ) / r
    w_rl = ( vx + vy - (lx + ly) * wz ) / r
    w_rr = ( vx - vy + (lx + ly) * wz ) / r

Two ways to use it:

  * With libgazebo_ros_planar_move driving the body (the default in
    auriga.urdf), this node only makes the wheels spin correctly on screen.
  * With the physics path (wheel joints under ros2_control + anisotropic
    friction), this node is what actually moves the robot.

Run:
    ros2 run auriga_description mecanum_cmd_vel.py
    ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {y: 0.3}}"
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray


class MecanumCmdVel(Node):

    def __init__(self):
        super().__init__('mecanum_cmd_vel')

        self.declare_parameter('wheel_radius', 0.05)      # metres
        self.declare_parameter('lx', 0.175)               # half wheelbase
        self.declare_parameter('ly', 0.205)               # half track
        self.declare_parameter('max_wheel_speed', 25.0)   # rad/s
        self.declare_parameter('command_topic',
                               '/wheel_velocity_controller/commands')

        self.r = self.get_parameter('wheel_radius').value
        self.l = (self.get_parameter('lx').value +
                  self.get_parameter('ly').value)
        self.w_max = self.get_parameter('max_wheel_speed').value

        self.pub = self.create_publisher(
            Float64MultiArray,
            self.get_parameter('command_topic').value, 10)
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 10)

        self.get_logger().info(
            f'mecanum IK ready: r={self.r} m, lx+ly={self.l} m')

    def on_cmd(self, msg: Twist):
        vx, vy, wz = msg.linear.x, msg.linear.y, msg.angular.z
        k = self.l * wz
        w = [(vx - vy - k) / self.r,
             (vx + vy + k) / self.r,
             (vx + vy - k) / self.r,
             (vx - vy + k) / self.r]

        # scale the whole vector rather than clipping wheels one by one,
        # so the commanded direction of travel is preserved
        peak = max(abs(v) for v in w)
        if peak > self.w_max:
            w = [v * self.w_max / peak for v in w]

        out = Float64MultiArray()
        out.data = w
        self.pub.publish(out)


def main():
    rclpy.init()
    node = MecanumCmdVel()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
