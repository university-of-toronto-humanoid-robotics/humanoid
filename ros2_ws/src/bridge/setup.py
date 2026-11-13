from setuptools import find_packages, setup

package_name = 'bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'mujoco_mid360_sim = bridge.mujoco_mid360_sim:main',
            'mujoco_g1_walk_sim = bridge.mujoco_g1_walk_sim:main',
            'body_to_pelvis = bridge.body_to_pelvis:main',
            'gravity_align = bridge.gravity_align:main',
            'self_filter = bridge.self_filter:main',
        ],
    },
)
