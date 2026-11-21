from setuptools import find_packages, setup

setup(
    name='localization',
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/localization']),
        ('share/localization', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={'console_scripts': ['localizer = localization.localizer_node:main']},
)
