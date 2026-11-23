from glob import glob
from setuptools import find_packages, setup
 
package_name = "librarian_ros"
 
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/urdf", glob("urdf/*")),
        ("share/" + package_name + "/config", glob("config/*")),
        ("share/" + package_name + "/worlds", glob("worlds/*")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="you",
    maintainer_email="you@example.com",
    description="Робо-бібліотекар: xArm7 на мобільній базі",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "build_robot = librarian_ros.build_robot:main",
            "perception_node = librarian_ros.perception_node:main",
            "pick_node = librarian_ros.pick_node:main",
            "arm_cli = librarian_ros.arm_cli:main",
        ],
    },
)
 
