"""move_group (MoveIt 2) для робота librarian_robot з використанням sim-часу.
 
  ros2 launch librarian_ros moveit.launch.py [rviz:=true]  (за замовчуванням вимкнено)
"""
import os
 
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
 
 
def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("librarian_robot", package_name="librarian_ros")
        .robot_description(file_path="urdf/librarian_robot.urdf")
        .robot_description_semantic(file_path="config/librarian.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_scene_monitor(publish_robot_description=True, publish_robot_description_semantic=True)
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    move_group = Node(package="moveit_ros_move_group", executable="move_group", output="screen",
                      parameters=[moveit_config.to_dict(), {"use_sim_time": True}])
    rviz = Node(package="rviz2", executable="rviz2", output="log", condition=IfCondition(LaunchConfiguration("rviz")),
                parameters=[moveit_config.robot_description, moveit_config.robot_description_semantic,
                            moveit_config.robot_description_kinematics, moveit_config.planning_pipelines,
                            moveit_config.joint_limits, {"use_sim_time": True}])
    return LaunchDescription([DeclareLaunchArgument("rviz", default_value="false", description="RViz у VM часто падає (SIGSEGV)"), move_group, rviz])
 
