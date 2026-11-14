from setuptools import find_packages, setup

setup(
    name='policy',
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/policy']),
        ('share/policy', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={'console_scripts': ['walk_controller = policy.walk_controller:main']},
)
