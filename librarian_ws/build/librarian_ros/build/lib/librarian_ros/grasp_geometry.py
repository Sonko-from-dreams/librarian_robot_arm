"""Геометрія захоплення книги без залежностей від ROS (тестується окремо).

Умовності:
  * кадр TCP: z = напрям підходу (вперед від фланця), вісь зведення пальців = jaw_axis (y або x);
  * захоплюємо книгу за край, пальці зводяться вздовж товщини книги (нормалі до обкладинки n).
"""
import math
from dataclasses import dataclass, field

import numpy as np


# ───────────── кватерніони / матриці ─────────────
def quat_to_rot(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R):
    """Матриця обертання -> кватерніон (x, y, z, w)."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def box_points(lo, hi, n=3):
    g = [np.linspace(lo[i], hi[i], n) for i in range(3)]
    return np.array(np.meshgrid(*g)).reshape(3, -1).T


# ───────────── моделі ─────────────
@dataclass
class Book:
    center: np.ndarray
    n: np.ndarray            # нормаль до обкладинки (напрям товщини)
    a1: np.ndarray           # осі в площині обкладинки
    a2: np.ndarray
    e1: float                # розміри вздовж a1, a2
    e2: float
    thickness: float = 0.03

    @property
    def flat(self):
        return abs(self.n[2]) > 0.93

    @property
    def corners_xy(self):
        return [self.center + s1 * self.a1 * self.e1 / 2 + s2 * self.a2 * self.e2 / 2
                for s1 in (1, -1) for s2 in (1, -1)]


@dataclass
class GripperModel:
    """Грубий габарит захвата в кадрі TCP. ПІДЛАШТУЙТЕ під офіційний xArm Gripper (дивіться README)."""
    tcp_to_tip: float = 0.0      # кінчики пальців попереду TCP на цю відстань (0, якщо link_tcp на кінчиках)
    open_half: float = 0.0425    # половина ходу (хід 85 мм)
    finger_t: float = 0.012      # товщина пальця
    finger_w: float = 0.02       # ширина пальця
    finger_len: float = 0.07     # довжина пальців
    palm_half: tuple = (0.04, 0.06)
    palm_len: float = 0.10       # довжина корпусу позаду пальців
    jaw_axis: str = "y"          # вісь зведення пальців у кадрі TCP: 'y' або 'x'
    _pts: np.ndarray = field(default=None, repr=False)

    def points(self, n=3):
        if self._pts is None:
            tip, hw = self.tcp_to_tip, self.finger_w / 2
            oh, t = self.open_half, self.finger_t
            parts = [
                box_points([-hw, oh, tip - self.finger_len], [hw, oh + t, tip], n),
                box_points([-hw, -oh - t, tip - self.finger_len], [hw, -oh, tip], n),
                box_points([-self.palm_half[0], -self.palm_half[1], tip - self.finger_len - self.palm_len],
                           [self.palm_half[0], self.palm_half[1], tip - self.finger_len], n),
            ]
            pts = np.vstack(parts)
            self._pts = pts[:, [1, 0, 2]] if self.jaw_axis == "x" else pts
        return self._pts


def tcp_rotation(a, close_dir, g: GripperModel):
    """Кадр TCP: z = a (підхід); вісь зведення = close_dir (спроєктована на площину ⟂ a)."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    c = np.asarray(close_dir, float)
    c = c - a * np.dot(c, a)
    c /= np.linalg.norm(c)
    if g.jaw_axis == "y":
        y = c
        x = np.cross(y, a)
    else:
        x = c
        y = np.cross(a, x)
    return np.column_stack([x, y, a])


def hits_aabb(R, tcp, g: GripperModel, aabb, margin=1e-3):
    w = np.asarray(tcp) + g.points() @ R.T
    lo, hi = aabb
    return bool(np.any(np.all((w > lo + margin) & (w < hi - margin), axis=1)))


# ───────────── оцінка пози книги ─────────────
def estimate_book(points, cam_pos, thickness=0.03):
    """OBB через PCA. Найменша вісь = нормаль до видимої обкладинки, спрямована до камери."""
    points = np.asarray(points, float)
    c0 = points.mean(0)
    _, _, Vt = np.linalg.svd(points - c0, full_matrices=False)
    proj = (points - c0) @ Vt.T
    lo, hi = proj.min(0), proj.max(0)
    center = c0 + ((lo + hi) / 2) @ Vt
    n = Vt[2] if np.dot(Vt[2], np.asarray(cam_pos) - center) > 0 else -Vt[2]
    center = center - n * thickness / 2          # видимий бік -> центр тіла книги
    a1 = Vt[0]
    a2 = np.cross(n, a1)                         # права трійка: a1 x a2 = n
    return Book(center, n, a1, a2, float(hi[0] - lo[0]), float(hi[1] - lo[1]), thickness)


# ───────────── кандидати захоплення ─────────────
def grasp_candidates(book: Book, g: GripperModel, table_aabb, depths=(0.02, 0.03), pre=0.10, min_edge=0.08):
    """Список кандидатів, відсортований за оцінкою. Без IK (її перевіряє MoveIt)."""
    out = []
    for ax, ext in ((book.a1, book.e1), (book.a2, book.e2)):
        if ext < min_edge:
            continue
        for sgn in (1, -1):
            outward = sgn * ax
            a = -outward
            if a[2] > 0.5:                      # знизу не заходимо
                continue
            edge = book.center + outward * ext / 2
            for depth in depths:
                tcp = edge + a * (depth - g.tcp_to_tip)
                for flip in (1, -1):            # два повороти на 180°: різні конфігурації 7-осьової руки
                    R = tcp_rotation(a, flip * book.n, g)
                    pre_pos = tcp - a * pre
                    if hits_aabb(R, tcp, g, table_aabb) or hits_aabb(R, pre_pos, g, table_aabb):
                        continue
                    score = -a[2] + 0.3 * a[0] + 0.1 * depth / max(depths)
                    out.append(dict(score=score, tcp=tcp, pre=pre_pos, R=R, approach=a))
    out.sort(key=lambda c: -c["score"])
    return out


def look_pose(center, g: GripperModel, height=0.28):
    """Поза огляду: захват згори над книгою, пальці вздовж осі y світу."""
    a = np.array([0.0, 0.0, -1.0])
    R = tcp_rotation(a, [0.0, 1.0, 0.0], g)
    return np.array([center[0], center[1], center[2] + height]), R


def push_plan(book: Book, g: GripperModel, table_aabb, overhang=0.04, lift=0.12, clearance=0.008):
    """Закритий захват підсуває пласку книгу до краю столу (-x), щоб вона звисала на `overhang`."""
    xs = [c[0] for c in book.corners_xy]
    need = min(xs) - (table_aabb[0][0] - overhang)
    if need <= 0:
        return None
    table_z = table_aabb[1][2]
    a = np.array([0.0, 0.0, -1.0])
    R = tcp_rotation(a, [1.0, 0.0, 0.0], g)     # бокова грань пальців упирається в книгу
    z = table_z + clearance + g.tcp_to_tip
    start = np.array([max(xs) + 0.03, book.center[1], z])
    end = start - np.array([need, 0.0, 0.0])
    up = np.array([0.0, 0.0, lift])
    return dict(R=R, above=start + up, start=start, end=end, end_up=end + up, distance=float(need))
