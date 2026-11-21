from glob import glob

from setuptools import find_packages, setup

setup(
    name='navigation',
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/navigation']),
        ('share/navigation', ['package.xml']),
        ('share/navigation/routes', glob('routes/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={'console_scripts': ['navigator = navigation.navigator_node:main',
                                      'cmd_vel_mux = navigation.cmd_vel_mux:main']},
)
