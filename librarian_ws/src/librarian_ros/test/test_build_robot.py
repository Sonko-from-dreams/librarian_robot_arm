"""Тест генератора на фіктивній руці з «офіційним» захватом (без ROS і xacro)."""
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from librarian_ros import build_robot as b


def fake_expand():
    ns = "{http://ros.org/wiki/xacro}"
    root = ET.parse(b.PKG_DIR / "urdf" / "librarian_base.urdf.xacro").getroot()
    macro = root.find(ns + "macro")
    body = list(macro)
    out = ET.Element("robot", name="librarian_base")
    for el in list(root):
        if el.tag == ns + "macro":
            continue
        if el.tag == ns + "wheel":
            s = {k: el.get(k) for k in ("name", "x", "y")}
            for tmpl in body:
                txt = ET.tostring(tmpl, encoding="unicode")
                for k, v in s.items():
                    txt = txt.replace("${%s}" % k, v)
                out.append(ET.fromstring(txt))
        else:
            out.append(el)
    return out


def fake_arm(path):
    r = ET.Element("robot", name="UF_ROBOT")
    ET.SubElement(r, "link", name="world")
    names = ["link_base"] + [f"link{i}" for i in range(1, 8)] + ["link_eef"]
    for n in names:
        l = ET.SubElement(r, "link", name=n)
        g = ET.SubElement(ET.SubElement(l, "visual"), "geometry")
        ET.SubElement(g, "mesh", filename=f"package://xarm_description/meshes/{n}.stl")
    j = ET.SubElement(r, "joint", name="world_joint", type="fixed")
    ET.SubElement(j, "parent", link="world"); ET.SubElement(j, "child", link="link_base")
    for i in range(1, 8):
        j = ET.SubElement(r, "joint", name=f"joint{i}", type="revolute")
        ET.SubElement(j, "parent", link=names[i - 1]); ET.SubElement(j, "child", link=names[i])
        ET.SubElement(j, "limit", lower="-3", upper="3", velocity="3.14", effort="50")
    j = ET.SubElement(r, "joint", name="joint_eef", type="fixed")
    ET.SubElement(j, "parent", link="link7"); ET.SubElement(j, "child", link="link_eef")
    # захват: base + drive + 5 mimic + tcp
    for n in ["xarm_gripper_base_link", "left_outer_knuckle", "left_finger", "right_outer_knuckle", "right_finger", "link_tcp"]:
        ET.SubElement(r, "link", name=n)
    j = ET.SubElement(r, "joint", name="gripper_fix", type="fixed")
    ET.SubElement(j, "parent", link="link_eef"); ET.SubElement(j, "child", link="xarm_gripper_base_link")
    j = ET.SubElement(r, "joint", name="drive_joint", type="revolute")
    ET.SubElement(j, "parent", link="xarm_gripper_base_link"); ET.SubElement(j, "child", link="left_outer_knuckle")
    ET.SubElement(j, "limit", lower="0", upper="0.85", velocity="3.14", effort="10")
    for jn, parent, child, mult in [("left_finger_joint", "left_outer_knuckle", "left_finger", "-1"),
                                    ("right_outer_knuckle_joint", "xarm_gripper_base_link", "right_outer_knuckle", "1"),
                                    ("right_finger_joint", "right_outer_knuckle", "right_finger", "-1")]:
        j = ET.SubElement(r, "joint", name=jn, type="revolute")
        ET.SubElement(j, "parent", link=parent); ET.SubElement(j, "child", link=child)
        ET.SubElement(j, "limit", lower="0", upper="0.85", velocity="3.14", effort="10")
        ET.SubElement(j, "mimic", joint="drive_joint", multiplier=mult, offset="0")
    j = ET.SubElement(r, "joint", name="tcp_fix", type="fixed")
    ET.SubElement(j, "parent", link="xarm_gripper_base_link"); ET.SubElement(j, "child", link="link_tcp")
    ET.ElementTree(r).write(path)


def test_build():
    b.expand_base = fake_expand
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fake_arm(d / "arm.urdf")
        (d / "xd").mkdir()
        b.main(["--arm-urdf", str(d / "arm.urdf"), "--out-dir", str(d / "out"), "--xarm-description-dir", str(d / "xd")])
        urdf = ET.parse(d / "out" / "urdf" / "librarian_robot.urdf").getroot()
        links = {l.get("name") for l in urdf.findall("link")}
        joints = {j.get("name"): j for j in urdf.findall("joint")}
        assert "world" in links and "wrist_camera_optical" in links and "link_tcp" in links
        assert joints["arm_mount_to_arm"].find("parent").get("link") == "arm_mount"
        # mesh -> file://
        assert all(m.get("filename").startswith("file://") for m in urdf.iter("mesh"))
        # ros2_control: команда лише в рушійних суглобів
        rc = urdf.find("ros2_control")
        cmd = {j.get("name") for j in rc.findall("joint") if j.find("command_interface") is not None}
        assert cmd == {f"joint{i}" for i in range(1, 8)} | {"drive_joint"}, cmd
        mim = {j.get("name") for j in rc.findall("joint") if j.find("command_interface") is None}
        assert mim == {"left_finger_joint", "right_outer_knuckle_joint", "right_finger_joint"}, mim
        ctrl = yaml.safe_load((d / "out" / "config" / "controllers.yaml").read_text())
        assert ctrl["gripper_controller"]["ros__parameters"]["joints"] == ["drive_joint"]
        info = yaml.safe_load((d / "out" / "config" / "robot_info.yaml").read_text())
        assert info["pick_node"]["ros__parameters"]["tcp_link"] == "link_tcp"
        srdf = ET.parse(d / "out" / "config" / "librarian.srdf").getroot()
        assert srdf.find("group[@name='arm']/chain").get("tip_link") == "link_tcp"
        for f in ("joint_limits", "kinematics", "moveit_controllers"):
            yaml.safe_load((d / "out" / "config" / f"{f}.yaml").read_text())
        print("test_build OK")


if __name__ == "__main__":
    test_build()
