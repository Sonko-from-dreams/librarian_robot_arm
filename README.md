source /opt/ros/lyrical/setup.bash

source ~/xarm_ws/install/setup.bash

cd ~/librarian_ws

colcon build --symlink-install --packages-select librarian_ros

source install/setup.bash  

--------------------------------------------------------------

ros2 launch librarian_ros sim.launch.py book_pose:=flat

ros2 launch librarian_ros moveit.launch.py

--------------------------------------------------------------

Перевірка

ros2 run librarian_ros arm_cli status         # положення TCP і суглобів

ros2 run librarian_ros arm_cli ready          # рука в робочу позу

ros2 run librarian_ros arm_cli gripper close

ros2 run librarian_ros arm_cli gripper open

ros2 run librarian_ros arm_cli pose 0.45 0.0 0.50    # TCP згори над столом

--------------------------------------------------------------

ros2 launch librarian_ros pick.launch.py
