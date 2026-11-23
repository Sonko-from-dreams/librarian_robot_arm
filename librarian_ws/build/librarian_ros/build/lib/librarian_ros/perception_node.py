#!/usr/bin/env python3
"""Сприйняття книги: RGB-D -> маска -> хмара точок -> OBB у кадрі world -> visualization_msgs/Marker.

Публікує /book/ext та /book/wrist (CUBE: поза = осі a1, a2, n; розміри = e1, e2, товщина).
Детектор: `color` (червона книга, без ML, для симуляції) або `yolo` (ultralytics, клас COCO «book»).
Жодного cv_bridge: зображення декодуються з байтів напряму.
"""
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker

from librarian_ros.grasp_geometry import estimate_book, quat_to_rot, rot_to_quat

COCO_BOOK = 73


class Perception(Node):
    def __init__(self):
        super().__init__("perception_node")
        self.declare_parameter("detector", "color")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("thickness", 0.03)
        self.declare_parameter("max_rate", 2.0)
        self.declare_parameter("yolo_model", "yolo11n-seg.pt")
        self.declare_parameter("cameras", ["ext", "wrist"])
        self.world = self.get_parameter("world_frame").value
        self.thickness = self.get_parameter("thickness").value
        self.min_dt = 1.0 / self.get_parameter("max_rate").value
        self.detector = self.get_parameter("detector").value
        self.yolo = None
        if self.detector == "yolo":
            from ultralytics import YOLO          # імпорт лише за потреби
            self.yolo = YOLO(self.get_parameter("yolo_model").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.state = {}
        for cam in self.get_parameter("cameras").value:
            self.state[cam] = dict(depth=None, info=None, last=0.0,
                                   pub=self.create_publisher(Marker, f"/book/{cam}", 10),
                                   frame=f"{cam}_camera_optical")
            self.create_subscription(Image, f"/{cam}_camera/depth_image", lambda m, c=cam: self._set(c, "depth", m), qos_profile_sensor_data)
            self.create_subscription(CameraInfo, f"/{cam}_camera/camera_info", lambda m, c=cam: self._set(c, "info", m), qos_profile_sensor_data)
            self.create_subscription(Image, f"/{cam}_camera/image", lambda m, c=cam: self._on_rgb(c, m), qos_profile_sensor_data)
        self.get_logger().info(f"Сприйняття запущено, детектор={self.detector}, камери={list(self.state)}")

    def _set(self, cam, key, msg):
        self.state[cam][key] = msg

    # ── детектори
    def _detect(self, rgb):
        if self.detector == "yolo":
            res = self.yolo(rgb[:, :, ::-1], classes=[COCO_BOOK], verbose=False)[0]
            if res.masks is None or len(res.masks) == 0:
                return None
            best = int(res.boxes.conf.argmax())
            m = res.masks.data[best].cpu().numpy() > 0.5
            if m.shape != rgb.shape[:2]:
                import cv2
                m = cv2.resize(m.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            return m
        r, g, b = (rgb[:, :, i].astype(np.int16) for i in range(3))
        return (r > 90) & (r > 1.6 * g) & (r > 1.6 * b)        # червона книга

    @staticmethod
    def _decode_depth(msg):
        if msg.encoding == "32FC1":
            return np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
        if msg.encoding == "16UC1":
            return np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0
        raise ValueError(f"невідоме кодування глибини: {msg.encoding}")

    # ── головний обробник
    def _on_rgb(self, cam, msg):
        st = self.state[cam]
        now = time.monotonic()
        if st["depth"] is None or st["info"] is None or now - st["last"] < self.min_dt:
            return
        st["last"] = now
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3]
        if msg.encoding == "bgr8":
            img = img[:, :, ::-1]
        depth = self._decode_depth(st["depth"])
        mask = self._detect(img)
        if mask is None:
            return
        valid = mask & np.isfinite(depth) & (depth > 0.05) & (depth < 3.0)
        vs, us = np.nonzero(valid)
        if len(vs) < 80:
            return
        K = st["info"].k
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]
        z = depth[vs, us].astype(np.float64)
        pts_cam = np.stack([(us - cx) * z / fx, (vs - cy) * z / fy, z], axis=1)
        try:
            tf = self.tf_buffer.lookup_transform(self.world, st["frame"], rclpy.time.Time())
        except Exception as e:                              # TF ще не готовий
            self.get_logger().warn(f"TF {self.world}<-{st['frame']}: {e}", throttle_duration_sec=5.0)
            return
        q = tf.transform.rotation
        t = tf.transform.translation
        R = quat_to_rot([q.x, q.y, q.z, q.w])
        trans = np.array([t.x, t.y, t.z])
        pts = pts_cam @ R.T + trans
        book = estimate_book(pts, trans, self.thickness)

        mk = Marker()
        mk.header.frame_id = self.world
        mk.header.stamp = self.get_clock().now().to_msg()
        mk.ns, mk.id, mk.type, mk.action = cam, 0, Marker.CUBE, Marker.ADD
        mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = map(float, book.center)
        qx, qy, qz, qw = rot_to_quat(np.column_stack([book.a1, book.a2, book.n]))
        mk.pose.orientation.x, mk.pose.orientation.y, mk.pose.orientation.z, mk.pose.orientation.w = qx, qy, qz, qw
        mk.scale.x, mk.scale.y, mk.scale.z = book.e1, book.e2, book.thickness
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = 1.0, 0.2, 0.2, 0.5
        mk.lifetime.sec = 5
        st["pub"].publish(mk)


def main():
    rclpy.init()
    node = Perception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
