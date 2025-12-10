from setuptools import find_packages, setup
from glob import glob

package_name = 'controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', ['config/pure_pursuit.yaml'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lyh',
    maintainer_email='93693983+Lyhoh@users.noreply.github.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pure_pursuit = controller.pure_pursuit_node:main',
            'pp_overtake = controller.pp_overtake:main',
        ],
    },
)
