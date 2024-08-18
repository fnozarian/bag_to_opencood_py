from setuptools import find_packages, setup

package_name = 'bag_to_opencood_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Farzad Nozarian',
    maintainer_email='fnozarian@gmail.com',
    description='A package to synchronize and save pose and lidar data to YAML and PCD files in OpenCood format.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sync_and_write_node = bag_to_opencood_py.bag_to_opencood:main',
        ],
    },
)
