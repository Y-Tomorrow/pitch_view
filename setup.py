from setuptools import find_packages, setup

package_name = 'pitch_view'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            ['launch/court_detector.launch.py']),
        ('share/' + package_name + '/config',
            ['config/court_detector.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yy',
    maintainer_email='you@example.com',
    description='Basketball court recognition and top-down 2D map projection.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'court_detector = pitch_view.court_detector_node:main',
        ],
    },
)
