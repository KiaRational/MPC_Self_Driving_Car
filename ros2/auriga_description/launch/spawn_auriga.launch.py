#!/usr/bin/env python3
"""
Spawn AURIGA in Gazebo Classic (ROS 2 Humble).

Package layout:

    auriga_description/
      urdf/auriga.urdf
      config/auriga_controllers.yaml
      media/materials/scripts/auriga.material
      media/materials/textures/auriga_label.png
      launch/spawn_auriga.launch.py
      scripts/mecanum_cmd_vel.py

The AppendEnvironmentVariable below is what makes the AURIGA plate texture
show up — Gazebo looks for material scripts under GAZEBO_RESOURCE_PATH.

    colcon build --packages-select auriga_description
    source install/setup.bash
    ros2 launch auriga_description spawn_auriga.launch.py

Drive it sideways (this is the fun part of a mecanum base):
    ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {y: 0.35}}"
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription, RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = 'auriga_description'


def generate_launch_description():
    pkg_share = get_package_share_directory(PKG)
    urdf_path = os.path.join(pkg_share, 'urdf', 'auriga.urdf')

    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    # Gazebo resolves 'Auriga/Label' by scanning media/materials/scripts here
    materials_path = AppendEnvironmentVariable('GAZEBO_RESOURCE_PATH', pkg_share)

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')),
        launch_arguments={'world': LaunchConfiguration('world'),
                          'verbose': 'false'}.items(),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description,
                     'use_sim_time': True}],
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description',
                   '-entity', 'auriga',
                   '-x', '0.0', '-y', '0.0', '-z', '0.08'],
        output='screen',
    )

    mecanum_ik = Node(
        package=PKG,
        executable='mecanum_cmd_vel.py',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    joint_state_broadcaster = Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster',
                   '--controller-manager', '/controller_manager'])

    arm_controller = Node(
        package='controller_manager', executable='spawner',
        arguments=['arm_controller', '--controller-manager', '/controller_manager'])

    gripper_controller = Node(
        package='controller_manager', executable='spawner',
        arguments=['gripper_controller', '--controller-manager', '/controller_manager'])

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='empty.world',
                              description='Gazebo world file'),
        materials_path,
        gazebo,
        robot_state_publisher,
        spawn_entity,
        mecanum_ik,
        RegisterEventHandler(OnProcessExit(target_action=spawn_entity,
                                           on_exit=[joint_state_broadcaster])),
        RegisterEventHandler(OnProcessExit(target_action=joint_state_broadcaster,
                                           on_exit=[arm_controller,
                                                    gripper_controller])),
    ])
