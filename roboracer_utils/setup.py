from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'roboracer_utils'

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
    description='Provides shared utility functions, including coordinate transformations and common mathematical tools, used across multiple Roboracer packages.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'tf_bridge_map_odom = roboracer_utils.tf_bridge_map_odom:main',
            'frenet_odom_republisher = roboracer_utils.frenet_odom_republisher:main',
        ],
    },
)
