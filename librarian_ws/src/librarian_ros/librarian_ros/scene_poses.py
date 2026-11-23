"""Початкові пози книги й підпірки для світу Gazebo (SDF: x y z roll pitch yaw)."""
import math

TABLE_Z = 0.30
W, L, T = 0.145, 0.21, 0.03          # у SDF: x = ширина, y = довжина, z = товщина (коли лежить пласко)


def book_poses(mode: str):
    """Повертає (поза_книги, поза_підпірки). Підпірка поза столом, якщо вона не потрібна."""
    far = "0 3 0.06 0 0 0"
    if mode == "flat":
        return f"0.46 0 {TABLE_Z + T / 2 + 0.002:.4f} 0 0 0.5", far
    if mode == "standing":
        return f"0.46 0 {TABLE_Z + W / 2 + 0.002:.4f} 0 -1.5708 0.2", far
    if mode == "lean":
        phi = math.radians(70)
        z = TABLE_Z + 0.5 * (W * math.sin(phi) + T * math.cos(phi)) + 0.002
        d = 0.5 * W * math.cos(phi) + 0.5 * T * math.sin(phi) + 0.007
        return f"0.46 0 {z:.4f} 0 {-phi:.4f} 0", f"{0.46 + d:.4f} 0 {TABLE_Z + 0.06:.4f} 0 0 0"
    raise ValueError(f"невідомий book_pose: {mode} (flat | standing | lean)")
