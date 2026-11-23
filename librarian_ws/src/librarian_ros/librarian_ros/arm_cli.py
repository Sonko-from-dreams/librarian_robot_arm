#!/usr/bin/env python3
"""Ручне керування рукою через MoveIt без RViz.
 
  ros2 run librarian_ros arm_cli status
  ros2 run librarian_ros arm_cli ready | home
  ros2 run librarian_ros arm_cli gripper open | close | 0.4
  ros2 run librarian_ros arm_cli pose 0.45 0.0 0.50            # TCP згори (пальці вздовж y світу)
  ros2 run librarian_ros arm_cli pose 0.45 0.0 0.50 --rpy 3.14 0 0
  ros2 run librarian_ros arm_cli joints 0 0.2 0 0.6 0 0.8 0
 
Координати: у кадрі world, метри. Стіл: x 0.33..0.95, висота 0.30.
Попередньо мають працювати sim.launch.py і moveit.launch.py (один екземпляр!).
"""
import argparse
import math
import sys
import threading
import time
 
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.utilities import remove_ros_args
from tf2_ros import Buffer, TransformListener
 
from librarian_ros.grasp_geometry import GripperModel, rot_to_quat, tcp_rotation
from librarian_ros.moveit_client import MoveItClient, load_gripper_cfg
 
ARM = [f"joint{i}" for i in range(1, 8)]
POSES = {"home": [0.0] * 7, "ready": [0.0, 0.2, 0.0, 0.6, 0.0, 0.8, 0.0]}
 
 
def rpy_to_quat(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), math.cos(y / 2), math.sin(y / 2)
    return [sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy]
 
 
def main():
    gc = load_gripper_cfg()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["status", "ready", "home", "gripper", "pose", "joints"])
    ap.add_argument("values", nargs="*")
    ap.add_argument("--rpy", nargs=3, type=float, default=None, help="орієнтація TCP (рад) замість «згори»")
    ap.add_argument("--tcp", default="link_tcp")
    ap.add_argument("--gripper-joint", default=gc["joint"])
    ap.add_argument("--open", type=float, default=gc["open"], dest="g_open")
    ap.add_argument("--closed", type=float, default=gc["closed"], dest="g_closed")
    ap.add_argument("--vel", type=float, default=0.3, help="масштаб швидкості 0..1")
    args = ap.parse_args(remove_ros_args(sys.argv)[1:])
 
    rclpy.init()
    node = Node("arm_cli", parameter_overrides=[Parameter("use_sim_time", value=True)])
    tfb = Buffer()
    TransformListener(tfb, node)
    mv = MoveItClient(node, "arm", args.tcp, args.gripper_joint, "world",
                      gripper_joints=gc["joints"], gripper_mult=gc["mult"], gripper_off=gc["off"])
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    ok = True
    try:
        if not mv.wait_ready(30.0):
            print("move_group/контролери не відповідають. Запущено moveit.launch.py (один раз)?")
            sys.exit(2)
 
        if args.cmd == "status":
            t0 = time.time()
            while time.time() - t0 < 5:
                try:
                    tf = tfb.lookup_transform("world", args.tcp, rclpy.time.Time())
                    p = tf.transform.translation
                    print(f"TCP ({args.tcp}) у world: x={p.x:.3f} y={p.y:.3f} z={p.z:.3f}")
                    break
                except Exception:
                    time.sleep(0.2)
            else:
                print("TF до TCP недоступний")
            for n in ARM + [args.gripper_joint]:
                v = mv.joint(n)
                print(f"  {n}: {'н/д' if v is None else f'{v:+.3f}'}")
        elif args.cmd in ("ready", "home"):
            ok = mv.move_to_joints(ARM, POSES[args.cmd], args.vel)
        elif args.cmd == "joints":
            if len(args.values) != 7:
                sys.exit("Потрібно 7 значень (рад).")
            ok = mv.move_to_joints(ARM, [float(v) for v in args.values], args.vel)
        elif args.cmd == "gripper":
            v = args.values[0] if args.values else "open"
            pos = args.g_open if v == "open" else args.g_closed if v == "close" else float(v)
            ok = mv.set_gripper(pos)
        elif args.cmd == "pose":
            if len(args.values) != 3:
                sys.exit("Потрібно x y z.")
            pos = [float(v) for v in args.values]
            if args.rpy:
                quat = rpy_to_quat(*args.rpy)
            else:
                quat = rot_to_quat(tcp_rotation([0, 0, -1], [0, 1, 0], GripperModel()))
            if not mv.ik_ok(pos, quat):
                print("IK не знаходить розв'язку (поза зоною досяжності або в колізії). Змініть координати.")
                sys.exit(3)
            ok = mv.move_to_pose(pos, quat, args.vel)
        print("OK" if ok else "НЕ ВДАЛОСЯ")
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    sys.exit(0 if ok else 1)
 
 
if __name__ == "__main__":
    main()
 
