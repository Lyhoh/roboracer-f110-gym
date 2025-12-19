from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lyh',
    maintainer_email='93693983+Lyhoh@users.noreply.github.com',
    description='Implements opponent vehicle detection and tracking based on LiDAR sensor data, providing estimated opponent states such as position and velocity for downstream planning modules.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detect = perception.detect:main',
            'opponent_tracker = perception.opponent_tracker:main',
            'tf_bridge_map_odom = perception.tf_bridge_map_odom:main',
            'ekf_vs_gt_monitor = perception.ekf_vs_gt_monitor:main',
        ],
    },
)
