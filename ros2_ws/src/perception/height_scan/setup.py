from setuptools import find_packages, setup

setup(
    name='height_scan',
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/height_scan']),
        ('share/height_scan', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={'console_scripts': ['height_scan_node = height_scan.height_scan_node:main']},
)
