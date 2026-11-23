#!/usr/bin/env python3
"""Збирає робота для Gazebo + MoveIt 2 з готових частин.
 
Вхід : розгорнутий URDF xArm7 З офіційним захватом (xacro xarm_device.urdf.xacro ... add_gripper:=true).
Склеює з базою (urdf/librarian_base.urdf.xacro) і додає:
  * кріплення до world (база нерухома), камеру на зап'ясті, плагін gz_ros2_control;
  * ros2_control-теги для суглобів руки й захвата (mimic-суглоби автоматично, за тегами <mimic> в URDF);
  * конфіги: controllers.yaml, joint_limits.yaml, kinematics.yaml, moveit_controllers.yaml,
    librarian.srdf (матриця колізій наближена, див. README), robot_info.yaml.
Усі імена беруться з вхідного URDF, а не вгадуються.
"""
import argparse
import itertools
import re
import sys
from collections import deque
from pathlib import Path
import xml.etree.ElementTree as ET
 
import yaml
 
PKG_DIR = Path(__file__).resolve().parent.parent
READY_Q = [0.0, 0.2, 0.0, 0.6, 0.0, 0.8, 0.0]
 
 
def share_dir(pkg, fallback=None):
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory(pkg))
    except Exception:
        if fallback:
            return Path(fallback).expanduser().resolve()
        sys.exit(f"Не знайшов пакет '{pkg}'. Виконайте source install/setup.bash або задайте шлях параметром.")
 
 
def expand_base():
    try:
        import xacro
    except ImportError:
        sys.exit("Потрібен модуль xacro (source /opt/ros/lyrical/setup.bash або pip install xacro).")
    doc = xacro.process_file(str(PKG_DIR / "urdf" / "librarian_base.urdf.xacro"))
    return ET.fromstring(doc.toxml())
 
 
def joint_children(robot):
    parent_of, children = {}, {}
    for j in robot.findall("joint"):
        p, c = j.find("parent").get("link"), j.find("child").get("link")
        parent_of[c] = p
        children.setdefault(p, []).append(c)
    return parent_of, children
 
 
def reachable(robot, root):
    _, children = joint_children(robot)
    seen, dq = set(), deque([root])
    while dq:
        cur = dq.popleft()
        if cur not in seen:
            seen.add(cur)
            dq.extend(children.get(cur, []))
    return seen
 
 
def tree_distances(robot):
    """Відстані між усіма ланками в дереві (кількість суглобів)."""
    adj = {}
    for j in robot.findall("joint"):
        p, c = j.find("parent").get("link"), j.find("child").get("link")
        adj.setdefault(p, set()).add(c)
        adj.setdefault(c, set()).add(p)
    dist = {}
    for s in adj:
        d, dq = {s: 0}, deque([s])
        while dq:
            cur = dq.popleft()
            for nb in adj[cur]:
                if nb not in d:
                    d[nb] = d[cur] + 1
                    dq.append(nb)
        dist[s] = d
    return dist
 
 
def wrist_camera_xml(parent, xyz, rpy):
    return ET.fromstring(f"""
<root>
  <link name="wrist_camera_link">
    <visual><geometry><box size="0.02 0.05 0.02"/></geometry></visual>
    <inertial><mass value="0.05"/><inertia ixx="1e-5" iyy="1e-5" izz="1e-5" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="eef_to_wrist_camera" type="fixed">
    <parent link="{parent}"/><child link="wrist_camera_link"/><origin xyz="{xyz}" rpy="{rpy}"/>
  </joint>
  <link name="wrist_camera_optical"/>
  <joint name="wrist_camera_optical_joint" type="fixed">
    <parent link="wrist_camera_link"/><child link="wrist_camera_optical"/>
    <origin xyz="0 0 0" rpy="-1.5708 0 -1.5708"/>
  </joint>
  <gazebo reference="wrist_camera_link">
    <sensor name="wrist_camera" type="rgbd_camera">
      <always_on>true</always_on><update_rate>10</update_rate><visualize>false</visualize>
      <topic>wrist_camera</topic>
      <camera>
        <horizontal_fov>1.3</horizontal_fov>
        <image><width>320</width><height>240</height><format>R8G8B8</format></image>
        <clip><near>0.05</near><far>2.0</far></clip>
        <depth_camera><clip><near>0.05</near><far>2.0</far></clip></depth_camera>
      </camera>
    </sensor>
  </gazebo>
</root>""")
 
 
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-urdf", required=True, help="розгорнутий URDF xArm7 з add_gripper:=true")
    ap.add_argument("--out-dir", default=str(PKG_DIR), help="куди писати urdf/ та config/ (вихідна тека пакета)")
    ap.add_argument("--arm-root", default="link_base")
    ap.add_argument("--tool-link", default="link_eef")
    ap.add_argument("--tcp-link", default="auto", help="'auto' = link_tcp, якщо є, інакше tool-link")
    ap.add_argument("--xarm-description-dir", default=None, help="шлях до share/xarm_description (якщо не знайдено)")
    ap.add_argument("--wrist-cam-xyz", default="0.09 0 0.04", help="відносно tool-link")
    ap.add_argument("--gripper-mode", choices=["actuated", "mimic"], default="actuated",
                    help="actuated: усі суглоби захвата керуються окремо за множниками з URDF (працює, коли рушій "
                         "фізики Gazebo не підтримує mimic); mimic: лише drive_joint, решта за <mimic>")
    ap.add_argument("--keep-wheel-joints", action="store_true",
                    help="лишити колеса рухомими (потрібен контролер і joint_states для них; для нерухомої бази не треба)")
    ap.add_argument("--wrist-cam-rpy", default="0 -1.9208 0", help="x камери = вперед; нахил ~20° до осі")
    a = ap.parse_args(argv)
 
    out = Path(a.out_dir)
    (out / "urdf").mkdir(parents=True, exist_ok=True)
    (out / "config").mkdir(parents=True, exist_ok=True)
 
    base = expand_base()
    arm = ET.parse(a.arm_urdf).getroot()
    link_names = {l.get("name") for l in arm.findall("link")}
    for need in (a.arm_root, a.tool_link):
        if need not in link_names:
            sys.exit(f"У URDF руки немає ланки '{need}'. Доступні: {sorted(link_names)}")
    keep = reachable(arm, a.arm_root)
 
    # mesh: package://xarm_description/ -> file://<share>/xarm_description/
    desc = share_dir("xarm_description", a.xarm_description_dir)
    for mesh in arm.iter("mesh"):
        fn = mesh.get("filename", "")
        mesh.set("filename", re.sub(r"^package://xarm_description/", f"file://{desc}/", fn))
        if fn.startswith("package://") and not fn.startswith("package://xarm_description/"):
            print(f"УВАГА: невідомий пакет у mesh: {fn}")
 
    # ── імена суглобів із URDF
    joints = {j.get("name"): j for j in arm.findall("joint")
              if j.find("parent").get("link") in keep and j.find("child").get("link") in keep}
    arm_joints = {}
    for name in joints:
        m = re.fullmatch(r"joint([1-7])", name)
        if m and joints[name].get("type") != "fixed":
            arm_joints[int(m.group(1))] = name
    if sorted(arm_joints) != list(range(1, 8)):
        sys.exit(f"Очікував joint1..joint7, знайшов {sorted(arm_joints.values())}")
    arm_names = [arm_joints[i] for i in range(1, 8)]
 
    mimic = {}      # joint -> (master, multiplier, offset)
    actuated_grip = []
    for name, j in joints.items():
        if name in arm_names or j.get("type") == "fixed":
            continue
        mm = j.find("mimic")
        if mm is not None:
            mimic[name] = (mm.get("joint"), float(mm.get("multiplier", 1)), float(mm.get("offset", 0)))
        else:
            actuated_grip.append(name)
    if len(actuated_grip) != 1:
        sys.exit(f"Очікував рівно один рушійний суглоб захвата, знайшов {actuated_grip}. "
                 f"Чи розгорнули ви URDF з add_gripper:=true?")
    grip_joint = actuated_grip[0]
    glim = joints[grip_joint].find("limit")
    g_lo, g_hi = float(glim.get("lower", 0)), float(glim.get("upper", 0.85))
 
    tcp = a.tool_link
    if a.tcp_link == "auto":
        if "link_tcp" in keep:
            tcp = "link_tcp"
    else:
        tcp = a.tcp_link
 
    # ── складання
    robot = ET.Element("robot", name="librarian_robot")
    seen_mat = set()
 
    def add(src, only=None):
        for el in src:
            if el.tag == "material":
                if el.get("name") not in seen_mat:
                    seen_mat.add(el.get("name"))
                    robot.append(el)
            elif el.tag == "link" and (only is None or el.get("name") in only):
                robot.append(el)
            elif el.tag == "joint" and (only is None or (
                    el.find("parent").get("link") in only and el.find("child").get("link") in only)):
                robot.append(el)
 
    add(base)
    add(arm, keep)
 
    def fixed(name, parent, child, xyz="0 0 0", rpy="0 0 0"):
        j = ET.SubElement(robot, "joint", name=name, type="fixed")
        ET.SubElement(j, "parent", link=parent)
        ET.SubElement(j, "child", link=child)
        ET.SubElement(j, "origin", xyz=xyz, rpy=rpy)
 
    if not a.keep_wheel_joints:
        # База нерухома (прикріплена до world), а MoveIt чекає joint_states УСІХ рухомих суглобів.
        for j in robot.findall("joint"):
            if j.get("name", "").endswith("_wheel_joint") and j.get("type") == "continuous":
                j.set("type", "fixed")
                for tag in ("axis", "limit", "dynamics"):
                    for el in j.findall(tag):
                        j.remove(el)
    ET.SubElement(robot, "link", name="world")
    fixed("world_to_base", "world", "base_footprint")
    fixed("arm_mount_to_arm", "arm_mount", a.arm_root)
    for el in wrist_camera_xml(a.tool_link, a.wrist_cam_xyz, a.wrist_cam_rpy):
        robot.append(el)
 
    # ── ros2_control (gz_ros2_control)
    rc = ET.SubElement(robot, "ros2_control", name="GazeboSimSystem", type="system")
    hw = ET.SubElement(rc, "hardware")
    ET.SubElement(hw, "plugin").text = "gz_ros2_control/GazeboSimSystem"
 
    def ctrl_joint(name, command=True, mimic_of=None):
        j = ET.SubElement(rc, "joint", name=name)
        if command:
            ET.SubElement(j, "command_interface", name="position")
        if mimic_of:
            ET.SubElement(j, "param", name="mimic").text = mimic_of[0]
            ET.SubElement(j, "param", name="multiplier").text = str(mimic_of[1])
        si = ET.SubElement(j, "state_interface", name="position")
        if command:
            ET.SubElement(si, "param", name="initial_value").text = "0.0"
        ET.SubElement(j, "state_interface", name="velocity")
 
    for n in arm_names:
        ctrl_joint(n)
    ctrl_joint(grip_joint)
    for n, m in mimic.items():
        if a.gripper_mode == "mimic":
            ctrl_joint(n, command=False, mimic_of=m)
        else:
            ctrl_joint(n)
    grip_ctrl = [grip_joint] + (list(mimic) if a.gripper_mode == "actuated" else [])
    grip_mult = [1.0] + [mimic[n][1] for n in grip_ctrl[1:]]
    grip_off = [0.0] + [mimic[n][2] for n in grip_ctrl[1:]]
    if a.gripper_mode == "actuated":
        # Суглоби керуються окремо, тож <mimic> прибираємо: інакше gz_ros2_control вважає їх mimic і
        # відмовляється від командних інтерфейсів (сервер Gazebo падає), а фізика й так не підтримує mimic.
        for n in mimic:
            mm = joints[n].find("mimic")
            if mm is not None:
                joints[n].remove(mm)
 
    controllers_cfg = out / "config" / "controllers.yaml"
    # шлях у URDF: share-теку пакета (при colcon build --symlink-install вказує на вихідні файли)
    pkg_share = share_dir("librarian_ros", str(out))
    gz = ET.SubElement(robot, "gazebo")
    plug = ET.SubElement(gz, "plugin", filename="libgz_ros2_control-system",
                         name="gz_ros2_control::GazeboSimROS2ControlPlugin")
    ET.SubElement(plug, "parameters").text = str(pkg_share / "config" / "controllers.yaml")
 
    ET.indent(robot)
    urdf_path = out / "urdf" / "librarian_robot.urdf"
    ET.ElementTree(robot).write(urdf_path, xml_declaration=True, encoding="utf-8")
 
    # ── перевірки дерева
    links = {l.get("name") for l in robot.findall("link")}
    kids = {j.find("child").get("link") for j in robot.findall("joint")}
    if links - kids != {"world"}:
        sys.exit(f"Дерево має кілька коренів: {links - kids}")
 
    # ── controllers.yaml
    def jtc(joints_):
        return {"ros__parameters": {
            "joints": joints_, "command_interfaces": ["position"],
            "state_interfaces": ["position", "velocity"],
            "allow_nonzero_velocity_at_trajectory_end": True}}
    ctrl = {
        "controller_manager": {"ros__parameters": {
            "update_rate": 200, "use_sim_time": True,
            "joint_state_broadcaster": {"type": "joint_state_broadcaster/JointStateBroadcaster"},
            "xarm7_controller": {"type": "joint_trajectory_controller/JointTrajectoryController"},
            "gripper_controller": {"type": "joint_trajectory_controller/JointTrajectoryController"}}},
        "xarm7_controller": jtc(arm_names),
        "gripper_controller": jtc(grip_ctrl),
    }
    controllers_cfg.write_text(yaml.safe_dump(ctrl, sort_keys=False), encoding="utf-8")
 
    # ── joint_limits.yaml (прискорення потрібні для тайм-параметризації)
    jl = {}
    for n in arm_names + [grip_joint]:
        vel = float(joints[n].find("limit").get("velocity", 3.14))
        jl[n] = {"has_velocity_limits": True, "max_velocity": vel,
                 "has_acceleration_limits": True, "max_acceleration": 5.0 if n in arm_names else 2.0}
    (out / "config" / "joint_limits.yaml").write_text(
        yaml.safe_dump({"joint_limits": jl}, sort_keys=False), encoding="utf-8")
 
    (out / "config" / "kinematics.yaml").write_text(yaml.safe_dump({"arm": {
        "kinematics_solver": "kdl_kinematics_plugin/KDLKinematicsPlugin",
        "kinematics_solver_search_resolution": 0.005,
        "kinematics_solver_timeout": 0.05}}, sort_keys=False), encoding="utf-8")
 
    def fjt(joints_):
        return {"type": "FollowJointTrajectory", "action_ns": "follow_joint_trajectory",
                "default": True, "joints": joints_}
    (out / "config" / "moveit_controllers.yaml").write_text(yaml.safe_dump({
        "moveit_manage_controllers": True,
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        "moveit_simple_controller_manager": {
            "controller_names": ["xarm7_controller"],      # захват керуємо напряму, не через MoveIt
            "xarm7_controller": fjt(arm_names)}}, sort_keys=False), encoding="utf-8")
 
    # ── SRDF
    dist = tree_distances(robot)
    pairs = set()
    for l1, l2 in itertools.combinations(sorted(links), 2):
        if dist[l1].get(l2, 99) <= 2:
            pairs.add((l1, l2, "Adjacent"))
    static = {"world", "base_footprint", "base_link", "mast", "arm_mount", a.arm_root,
              "ext_camera_link", "ext_camera_optical"} | {l for l in links if l.endswith("_wheel")}
    grip_links = reachable(robot, a.tool_link) | {"link5", "link6", "link7"}
    for group in (static, grip_links):
        for l1, l2 in itertools.combinations(sorted(group & links), 2):
            pairs.add((l1, l2, "Never"))
    srdf = ET.Element("robot", name="librarian_robot")
    g_arm = ET.SubElement(srdf, "group", name="arm")
    ET.SubElement(g_arm, "chain", base_link=a.arm_root, tip_link=tcp)
    g_gr = ET.SubElement(srdf, "group", name="gripper")
    ET.SubElement(g_gr, "joint", name=grip_joint)
    for sname, q in (("home", [0.0] * 7), ("ready", READY_Q)):
        gs = ET.SubElement(srdf, "group_state", name=sname, group="arm")
        for n, v in zip(arm_names, q):
            ET.SubElement(gs, "joint", name=n, value=str(v))
    for sname, v in (("open", g_lo), ("closed", g_hi)):
        gs = ET.SubElement(srdf, "group_state", name=sname, group="gripper")
        ET.SubElement(gs, "joint", name=grip_joint, value=str(v))
    ET.SubElement(srdf, "end_effector", name="gripper", parent_link=a.tool_link,
                  group="gripper", parent_group="arm")
    for l1, l2, why in sorted(pairs):
        ET.SubElement(srdf, "disable_collisions", link1=l1, link2=l2, reason=why)
    ET.indent(srdf)
    ET.ElementTree(srdf).write(out / "config" / "librarian.srdf", xml_declaration=True, encoding="utf-8")
 
    # ── параметри для вузлів
    info = {"pick_node": {"ros__parameters": {
        "arm_group": "arm", "tcp_link": tcp, "gripper_joint": grip_joint,
        "gripper_open": g_lo, "gripper_closed": g_hi,
        "gripper_joints": grip_ctrl, "gripper_multipliers": [float(v) for v in grip_mult],
        "gripper_offsets": [float(v) for v in grip_off]}}}
    (out / "config" / "robot_info.yaml").write_text(yaml.safe_dump(info, sort_keys=False), encoding="utf-8")
 
    print(f"OK: {urdf_path}")
    print(f"    рука: {arm_names}")
    print(f"    захват: рушійний {grip_joint} [{g_lo}, {g_hi}], mimic: {sorted(mimic)}, режим {a.gripper_mode}")
    print(f"    суглоби gripper_controller: {grip_ctrl}")
    print(f"    TCP-ланка для планування: {tcp}")
    print(f"    SRDF: {len(pairs)} вимкнених пар колізій (наближення)")
 
 
if __name__ == "__main__":
    main()
 
