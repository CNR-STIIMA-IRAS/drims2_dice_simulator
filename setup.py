from setuptools import setup
import os
import glob

package_name = 'drims2_dice_simulator'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    py_modules=[],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*', recursive=True)),
        ('share/' + package_name + '/urdf', glob.glob('urdf/*', recursive=True)),
        ('share/' + package_name + '/srv', glob.glob('srv/*.srv', recursive=True)),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Cesare Tonola',
    maintainer_email='cesare.tonola@cnr.it',
    description='DRIMS2 Dice Simulator Package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dice_spawner = drims2_dice_simulator.dice_spawner:main'
        ],
    },
)
