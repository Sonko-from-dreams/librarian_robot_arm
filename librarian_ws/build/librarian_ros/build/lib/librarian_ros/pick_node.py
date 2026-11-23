#!/usr/bin/env python3
"""Сценарій «знайти й підняти книгу»: огляд -> уточнення з зап'ястя -> кандидати -> IK (MoveIt) -> захоплення.

Пласка книга: спершу підсуваємо її до краю столу (push_plan), потім беремо за звис.
Запускайте ПІСЛЯ sim.launch.py + moveit.launch.py + perception_node.
"""
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from visualization_msgs.msg import Marker

from librarian_ros.grasp_geometry import (Book, GripperModel, grasp_candidates, look_pose,
                                          push_plan, quat_to_rot, rot_to_quat)
from librarian_ros.moveit_client import MoveItClient

READY = [0.0, 0.2, 0.0, 0.6, 0.0, 0.8, 0.0]


class PickNode(Node):
    def __init__(self):
        super().__init__("pick_node")
        d = self.declare_parameter
        d("arm_group", "arm"); d("tcp_link", "link_tcp"); d("gripper_joint", "drive_joint")
        d("gripper_open", 0.0); d("gripper_closed", 0.85); d("world_frame", "world")
        d("table_min", [0.33, -0.45, 0.0]); d("table_max", [0.95, 0.45, 0.30])
        d("arm_joints", [f"joint{i}" for i in range(1, 8)])
        # габарит захвата: ПІДЛАШТУЙТЕ під офіційний xArm Gripper
        d("tcp_to_tip", 0.0); d("jaw_axis", "y"); d("finger_len", 0.07); d("palm_len", 0.10)
        d("look_height", 0.28); d("lift", 0.15); d("empty_margin", 0.05)
        p = lambda n: self.get_parameter(n).value
        self.p = p
        self.world = p("world_frame")
        self.table = (np.array(p("table_min"), float), np.array(p("table_max"), float))
        self.g = GripperModel(tcp_to_tip=p("tcp_to_tip"), jaw_axis=p("jaw_axis"),
                              finger_len=p("finger_len"), palm_len=p("palm_len"))
        self.mv = MoveItClient(self, p("arm_group"), p("tcp_link"), p("gripper_joint"), self.world)
        self.latest = {"ext": None, "wrist": None}
        for cam in self.latest:
            self.create_subscription(Marker, f"/book/{cam}", lambda m, c=cam: self._on_marker(c, m), 10)

    def _on_marker(self, cam, m):
        self.latest[cam] = m

    # ── допоміжне
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def sleep(self, sec):
        t0 = self.now()
        while self.now() - t0 < sec:
            time.sleep(0.05)

    def book_from(self, cam, newer_than, timeout=15.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            m = self.latest[cam]
            if m is not None and m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 > newer_than:
                q = m.pose.orientation
                R = quat_to_rot([q.x, q.y, q.z, q.w])
                c = m.pose.position
                return Book(np.array([c.x, c.y, c.z]), R[:, 2], R[:, 0], R[:, 1],
                            m.scale.x, m.scale.y, m.scale.z)
            time.sleep(0.05)
        return None

    def log(self, msg):
        self.get_logger().info(msg)

    # ── сценарій
    def run(self):
        L = self.get_logger()
        if not self.mv.wait_ready(120.0):
            L.error("move_group / контролери не готові. Запущено moveit.launch.py?")
            return False
        lo, hi = self.table
        # стіл у планувальну сцену трохи меншим, щоб дотик пальців не давав хибних колізій
        self.mv.add_box("table", (lo + hi) / 2, (hi - lo) - np.array([0.01, 0.01, 0.01]))
        self.mv.move_to_joints(self.p("arm_joints"), READY)
        self.mv.set_gripper(self.p("gripper_open"))

        for attempt in range(1, 4):
            self.log(f"=== спроба {attempt} ===")
            t = self.now()
            coarse = self.book_from("ext", t - 5.0, 10.0)     # зовнішня камера: рука складена
            if coarse is None:
                L.error("Зовнішня камера не бачить книгу (/book/ext). Перевірте bridge та детектор.")
                return False
            pos, R = look_pose(coarse.center, self.g, self.p("look_height"))
            if not self.mv.move_to_pose(pos, rot_to_quat(R)):
                L.error("Поза огляду недосяжна. Змініть look_height або положення книги.")
                return False
            self.sleep(1.5)
            book = self.book_from("wrist", self.now(), 10.0) or coarse
            self.log(f"Книга: центр {np.round(book.center, 3)}, розміри {book.e1:.3f}x{book.e2:.3f}, "
                     f"{'лежить пласко' if book.flat else 'нахилена/стоїть'}")

            cands = grasp_candidates(book, self.g, self.table)
            chosen = None
            for c in cands:
                q = rot_to_quat(c["R"])
                if self.mv.ik_ok(c["pre"], q) and self.mv.ik_ok(c["tcp"], q):
                    chosen = (c, q)
                    break
            if chosen is None:
                plan = push_plan(book, self.g, self.table) if book.flat else None
                if plan is None:
                    L.error(f"Жодного досяжного захоплення ({len(cands)} кандидатів відсіяно IK/колізіями).")
                    return False
                self.log(f"Пласка книга: підсуваю до краю на {plan['distance'] * 100:.1f} см")
                q = rot_to_quat(plan["R"])
                self.mv.set_gripper(self.p("gripper_closed"))
                ok = (self.mv.move_to_pose(plan["above"], q) and self.mv.cartesian([(plan["start"], q)])
                      and self.mv.cartesian([(plan["end"], q)]) and self.mv.cartesian([(plan["end_up"], q)]))
                if not ok:
                    L.error("Підсування не вдалося (шлях недосяжний або в колізії).")
                    return False
                self.mv.move_to_joints(self.p("arm_joints"), READY)
                continue

            c, q = chosen
            a = c["approach"]
            self.log(f"Захоплення: підхід {np.round(a, 2)}, TCP {np.round(c['tcp'], 3)}")
            self.mv.set_gripper(self.p("gripper_open"))
            if not self.mv.move_to_pose(c["pre"], q):
                L.error("Не дійшов до pre-grasp.")
                return False
            if not self.mv.cartesian([(c["tcp"], q)]):
                L.error("Підхід по прямій не вдався.")
                return False
            self.mv.set_gripper(self.p("gripper_closed"), 2.0)
            self.sleep(0.5)
            g = self.mv.joint(self.p("gripper_joint"))
            empty = g is None or g > self.p("gripper_closed") - self.p("empty_margin")
            gtxt = "н/д" if g is None else f"{g:.3f}"
            self.log(f"Суглоб захвата {gtxt} (закритий = {self.p('gripper_closed')}) -> {'ПОРОЖНЬО' if empty else 'книга в руці'}")
            if empty:
                self.mv.set_gripper(self.p("gripper_open"))
                self.mv.move_to_joints(self.p("arm_joints"), READY)
                continue
            lift = np.array([0, 0, self.p("lift")])
            ok = self.mv.cartesian([(c["tcp"] + lift, q)])
            self.log("УСПІХ: книгу піднято" if ok else "Підйом не вдався")
            return ok
        L.error("Вичерпано спроби.")
        return False


def main():
    rclpy.init()
    node = PickNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    try:
        ok = node.run()
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
