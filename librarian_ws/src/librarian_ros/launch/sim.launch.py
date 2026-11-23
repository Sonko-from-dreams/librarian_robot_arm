"""Gazebo (gz sim) + робот + gz_ros2_control + bridge камер.
 
  ros2 launch librarian_ros sim.launch.py [book_pose:=flat|standing|lean] [headless:=true]
"""
import os
import tempfile
from pathlib import Path
 
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
 
from librarian_ros.scene_poses import book_poses
 
 
def spawner(name, params_file):
    """--param-file: спавнер сам завантажує параметри контролера (joints тощо) із controllers.yaml."""
    return Node(package="controller_manager", executable="spawner", output="screen",
                arguments=[name, "--controller-manager", "/controller_manager",
                           "--controller-manager-timeout", "60", "--param-file", params_file])
 
 
def launch_setup(context):
    pkg = Path(get_package_share_directory("librarian_ros"))
    urdf = pkg / "urdf" / "librarian_robot.urdf"
    if not urdf.exists():
        raise RuntimeError(f"Немає {urdf}. Спершу виконайте: ros2 run librarian_ros build_robot --arm-urdf <xarm7.urdf>")
    mode = LaunchConfiguration("book_pose").perform(context)
    headless = LaunchConfiguration("headless").perform(context).lower() == "true"
 
    book, wall = book_poses(mode)
    world_txt = (pkg / "worlds" / "library.sdf.in").read_text().replace("@BOOK_POSE@", book).replace("@WALL_POSE@", wall)
    world = Path(tempfile.gettempdir()) / "librarian_world.sdf"
    world.write_text(world_txt)
 
    gz_args = f"-r {world}" + (" -s --headless-rendering" if headless else "")
    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": gz_args, "on_exit_shutdown": "true"}.items())
 
    rsp = Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen",
               parameters=[{"robot_description": urdf.read_text(), "use_sim_time": True}])
    create = Node(package="ros_gz_sim", executable="create", output="screen",
                  arguments=["-topic", "robot_description", "-name", "librarian", "-x", "0", "-y", "0", "-z", "0.0"])
    bridge = Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
                  parameters=[{"config_file": str(pkg / "config" / "bridge.yaml"), "use_sim_time": True}])
 
    cfg = str(pkg / "config" / "controllers.yaml")
    jsb, arm, grip = (spawner("joint_state_broadcaster", cfg), spawner("xarm7_controller", cfg),
                      spawner("gripper_controller", cfg))
    chain = [
        RegisterEventHandler(OnProcessExit(target_action=create, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[arm, grip])),
    ]
    return [gz, rsp, create, bridge, *chain]
 
 
def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("book_pose", default_value="flat", description="flat | standing | lean"),
        DeclareLaunchArgument("headless", default_value="false", description="без вікна Gazebo (сервер + headless-рендеринг)"),
        OpaqueFunction(function=launch_setup),
    ])
 
