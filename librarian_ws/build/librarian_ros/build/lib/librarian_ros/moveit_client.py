"""Тонкий клієнт до move_group через стандартні інтерфейси MoveIt 2 (без moveit_py).

Дії/сервіси: /move_action, /compute_ik, /compute_cartesian_path, /execute_trajectory,
/apply_planning_scene + FollowJointTrajectory для захвата.
Усі виклики блокуючі: вузол має крутитися в іншому потоці (див. pick_node.main).
"""
import time

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (BoundingVolume, CollisionObject, Constraints, JointConstraint,
                             MoveItErrorCodes, OrientationConstraint, PlanningScene,
                             PositionConstraint)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetPositionIK
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectoryPoint


def make_pose(pos, quat):
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in pos)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (float(v) for v in quat)
    return p


class MoveItClient:
    def __init__(self, node, group, tcp_link, gripper_joint, world_frame="world"):
        self.node, self.group, self.tcp, self.gj, self.frame = node, group, tcp_link, gripper_joint, world_frame
        self.move_ac = ActionClient(node, MoveGroup, "move_action")
        self.exec_ac = ActionClient(node, ExecuteTrajectory, "execute_trajectory")
        self.grip_ac = ActionClient(node, FollowJointTrajectory, "/gripper_controller/follow_joint_trajectory")
        self.ik_cli = node.create_client(GetPositionIK, "compute_ik")
        self.cart_cli = node.create_client(GetCartesianPath, "compute_cartesian_path")
        self.scene_cli = node.create_client(ApplyPlanningScene, "apply_planning_scene")
        self.js = None
        node.create_subscription(JointState, "/joint_states", self._on_js, 10)

    # ── утиліти
    def _on_js(self, msg):
        self.js = msg

    def joint(self, name):
        if self.js and name in self.js.name:
            return self.js.position[self.js.name.index(name)]
        return None

    def wait(self, fut, timeout=60.0):
        t0 = time.time()
        while not fut.done():
            if time.time() - t0 > timeout:
                return None
            time.sleep(0.01)
        return fut.result()

    def wait_ready(self, timeout=60.0):
        ok = (self.move_ac.wait_for_server(timeout) and self.exec_ac.wait_for_server(timeout)
              and self.grip_ac.wait_for_server(timeout) and self.ik_cli.wait_for_service(timeout)
              and self.cart_cli.wait_for_service(timeout) and self.scene_cli.wait_for_service(timeout))
        t0 = time.time()
        while self.js is None and time.time() - t0 < timeout:
            time.sleep(0.05)
        return bool(ok and self.js)

    # ── планувальна сцена
    def add_box(self, name, center, size):
        co = CollisionObject()
        co.header.frame_id = self.frame
        co.id = name
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [float(s) for s in size]
        co.primitives = [prim]
        co.primitive_poses = [make_pose(center, [0, 0, 0, 1])]
        co.operation = CollisionObject.ADD
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [co]
        req = ApplyPlanningScene.Request()
        req.scene = scene
        res = self.wait(self.scene_cli.call_async(req), 10.0)
        return bool(res and res.success)

    # ── планування і виконання
    def _send_move(self, goal):
        gh = self.wait(self.move_ac.send_goal_async(goal), 10.0)
        if gh is None or not gh.accepted:
            self.node.get_logger().error("move_group відхилив ціль")
            return False
        res = self.wait(gh.get_result_async(), 90.0)
        ok = res is not None and res.result.error_code.val == MoveItErrorCodes.SUCCESS
        if not ok:
            code = res.result.error_code.val if res else "timeout"
            self.node.get_logger().warn(f"Планування/виконання не вдалося, код MoveIt: {code}")
        return ok

    def _goal(self, constraints, vel):
        goal = MoveGroup.Goal()
        r = goal.request
        r.group_name = self.group
        r.goal_constraints = [constraints]
        r.num_planning_attempts = 5
        r.allowed_planning_time = 5.0
        r.max_velocity_scaling_factor = vel
        r.max_acceleration_scaling_factor = vel
        goal.planning_options.plan_only = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        return goal

    def move_to_pose(self, pos, quat, vel=0.3):
        ps = PoseStamped()
        ps.header.frame_id = self.frame
        ps.pose = make_pose(pos, quat)
        sp = SolidPrimitive()
        sp.type = SolidPrimitive.SPHERE
        sp.dimensions = [0.002]
        bv = BoundingVolume()
        bv.primitives = [sp]
        bv.primitive_poses = [ps.pose]
        pc = PositionConstraint()
        pc.header, pc.link_name, pc.constraint_region, pc.weight = ps.header, self.tcp, bv, 1.0
        oc = OrientationConstraint()
        oc.header, oc.link_name, oc.orientation, oc.weight = ps.header, self.tcp, ps.pose.orientation, 1.0
        oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = 0.02
        c = Constraints()
        c.position_constraints, c.orientation_constraints = [pc], [oc]
        return self._send_move(self._goal(c, vel))

    def move_to_joints(self, names, values, vel=0.3):
        c = Constraints()
        for n, v in zip(names, values):
            jc = JointConstraint()
            jc.joint_name, jc.position = n, float(v)
            jc.tolerance_above = jc.tolerance_below = 0.01
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        return self._send_move(self._goal(c, vel))

    def cartesian(self, waypoints, step=0.005, min_fraction=0.98):
        """waypoints: список (pos, quat). Пряма в декартовому просторі й виконання."""
        req = GetCartesianPath.Request()
        req.header.frame_id = self.frame
        req.group_name, req.link_name = self.group, self.tcp
        req.waypoints = [make_pose(p, q) for p, q in waypoints]
        req.max_step = step
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        res = self.wait(self.cart_cli.call_async(req), 15.0)
        if res is None or res.fraction < min_fraction:
            self.node.get_logger().warn(f"Декартовий шлях побудовано лише на {0 if res is None else res.fraction:.0%}")
            return False
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = res.solution
        gh = self.wait(self.exec_ac.send_goal_async(goal), 10.0)
        if gh is None or not gh.accepted:
            return False
        out = self.wait(gh.get_result_async(), 60.0)
        return out is not None and out.result.error_code.val == MoveItErrorCodes.SUCCESS

    def ik_ok(self, pos, quat):
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name, r.ik_link_name, r.avoid_collisions = self.group, self.tcp, True
        r.pose_stamped.header.frame_id = self.frame
        r.pose_stamped.pose = make_pose(pos, quat)
        r.timeout = Duration(sec=0, nanosec=150_000_000)
        if self.js is not None:
            r.robot_state.joint_state = self.js
        res = self.wait(self.ik_cli.call_async(req), 5.0)
        return bool(res and res.error_code.val == MoveItErrorCodes.SUCCESS)

    # ── захват (контролер суглоба drive_joint)
    def set_gripper(self, position, seconds=1.5):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [self.gj]
        pt = JointTrajectoryPoint()
        pt.positions = [float(position)]
        pt.time_from_start = Duration(sec=int(seconds), nanosec=int((seconds % 1) * 1e9))
        goal.trajectory.points = [pt]
        gh = self.wait(self.grip_ac.send_goal_async(goal), 10.0)
        if gh is None or not gh.accepted:
            return False
        self.wait(gh.get_result_async(), 20.0)
        return True
