"""Сприйняття + сценарій піднімання книги.

  ros2 launch librarian_ros pick.launch.py [detector:=color|yolo]
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("librarian_ros")
    robot_info = os.path.join(pkg, "config", "robot_info.yaml")
    gripper_model = os.path.join(pkg, "config", "gripper_model.yaml")
    perception = Node(package="librarian_ros", executable="perception_node", output="screen",
                      parameters=[{"use_sim_time": True, "detector": LaunchConfiguration("detector")}])
    pick = Node(package="librarian_ros", executable="pick_node", output="screen",
                parameters=[robot_info, gripper_model, {"use_sim_time": True}])
    return LaunchDescription([DeclareLaunchArgument("detector", default_value="color"), perception, pick])
