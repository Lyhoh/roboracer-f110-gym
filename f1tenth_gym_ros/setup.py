from setuptools import setup
import os
from glob import glob

package_name = 'f1tenth_gym_ros'

data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.xacro')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.rviz')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ]

tracks_root = os.path.join('maps', 'f1tenth_racetracks')

if os.path.isdir(tracks_root):
    for root, _, files in os.walk(tracks_root):
        if not files:
            continue
        rel_root = os.path.relpath(root, 'maps')   # -> 'f1tenth_racetracks/Austin'
        dest = os.path.join('share', package_name, 'maps', rel_root)
        src_files = [os.path.join(root, f) for f in files]
        data_files.append((dest, src_files))

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Billy Zheng',
    maintainer_email='billyzheng.bz@gmail.com',
    description='Bridge for using f1tenth_gym in ROS2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gym_bridge = f1tenth_gym_ros.gym_bridge:main'
        ],
    },
)
