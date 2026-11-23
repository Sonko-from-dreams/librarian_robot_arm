import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from librarian_ros import grasp_geometry as gg

TABLE = (np.array([0.33, -0.45, 0.0]), np.array([0.95, 0.45, 0.30]))


def test_quat_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(500):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        assert np.abs(gg.quat_to_rot(gg.rot_to_quat(Q)) - Q).max() < 1e-9


def flat_points(rng, yaw=0.6, c=(0.46, 0.0, 0.315)):
    L, W = 0.21, 0.145
    u = np.array([math.cos(yaw), math.sin(yaw), 0]); v = np.array([-math.sin(yaw), math.cos(yaw), 0])
    P = [np.array(c) + u * rng.uniform(-L / 2, L / 2) + v * rng.uniform(-W / 2, W / 2) + [0, 0, 0.015] for _ in range(4000)]
    return np.array(P) + rng.normal(0, 0.001, (4000, 3))


def test_estimate_flat():
    rng = np.random.default_rng(2)
    b = gg.estimate_book(flat_points(rng), [0.2, 0, 0.8])
    assert np.linalg.norm(b.center - [0.46, 0, 0.30 + 0.015]) < 0.005, b.center
    assert b.flat and abs(np.dot(np.cross(b.a1, b.a2), b.n) - 1) < 1e-6


def test_flat_has_no_grasp_then_push():
    rng = np.random.default_rng(3)
    g = gg.GripperModel()
    b = gg.estimate_book(flat_points(rng), [0.2, 0, 0.8])
    assert gg.grasp_candidates(b, g, TABLE) == [], "для пласкої книги на столі кандидатів бути не має"
    plan = gg.push_plan(b, g, TABLE)
    assert plan is not None and 0.02 < plan["distance"] < 0.30, plan
    assert plan["start"][0] > b.center[0] and plan["end"][0] < plan["start"][0]


def test_standing_book_top_grasp():
    g = gg.GripperModel()
    n = np.array([-1.0, 0, 0]); a1 = np.array([0, 1.0, 0]); a2 = np.cross(n, a1)   # стоїть: корінець униз
    b = gg.Book(np.array([0.46, 0.0, 0.30 + 0.0725]), n, a1, a2, 0.21, 0.145)
    c = gg.grasp_candidates(b, g, TABLE)
    assert c, "для стоячої книги мають бути кандидати"
    top = c[0]
    assert top["approach"][2] < -0.9, "перший кандидат: згори"
    assert abs(top["R"][:, 1] @ n) > 0.99, "пальці зводяться вздовж товщини"
    assert np.linalg.det(top["R"]) > 0.999


def test_overhang_allows_horizontal_grasp():
    g = gg.GripperModel()
    # пласка книга звисає за край столу (x_min = 0.29, край 0.33)
    n = np.array([0, 0, 1.0]); a1 = np.array([1.0, 0, 0]); a2 = np.cross(n, a1)
    b = gg.Book(np.array([0.29 + 0.0725, 0.0, 0.315]), n, a1, a2, 0.145, 0.21)
    c = gg.grasp_candidates(b, g, TABLE, depths=(0.02,))
    horiz = [x for x in c if abs(x["approach"][2]) < 0.1 and x["approach"][0] > 0.9]
    assert horiz, "підхід з боку робота горизонтально має бути можливий"


def test_jaw_axis_x():
    g = gg.GripperModel(jaw_axis="x")
    R = gg.tcp_rotation([0, 0, -1], [0, 1, 0], g)
    assert np.allclose(R[:, 0], [0, 1, 0]) and np.linalg.det(R) > 0.999


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print(name, "OK")
