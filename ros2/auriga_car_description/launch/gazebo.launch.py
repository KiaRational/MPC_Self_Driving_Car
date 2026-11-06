"""Bring up AURIGA in Gazebo Classic + robot_state_publisher (+ optional RViz).

    ros2 launch auriga_car_description gazebo.launch.py
    ros2 launch auriga_car_description gazebo.launch.py rviz:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('auriga_car_description')
    urdf = os.path.join(pkg, 'urdf', 'auriga_car.urdf')
    with open(urdf, 'r') as f:
        robot_description = f.read()

    use_rviz = LaunchConfiguration('rviz')
    world = LaunchConfiguration('world')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')),
        launch_arguments={'world': world, 'verbose': 'true'}.items(),
    )

    rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description,
                     'use_sim_time': True}],
    )

    spawn = Node(
        package='gazebo_ros', executable='spawn_entity.py', output='screen',
        arguments=['-topic', 'robot_description', '-entity', 'auriga_car',
                   '-x', '0', '-y', '0', '-z', '0.06'],
    )

    rviz = Node(
        package='rviz2', executable='rviz2', output='screen',
        condition=IfCondition(use_rviz),
        arguments=['-d', os.path.join(pkg, 'rviz', 'auriga_car.rviz')],
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('world', default_value='empty.world'),
        gazebo, rsp, spawn, rviz,
    ])
