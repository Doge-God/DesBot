from setuptools import find_packages, setup

package_name = 'des_bot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='fz',
    maintainer_email='futianzhou@hotmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_pub = des_bot.vision_pub:main',
            'naive_track = des_bot.naive_vision_track:main',
            'naive_face = des_bot.face_controller:main',
            'stt = des_bot.stt:main',
            'conversation_manager = des_bot.conversation_manager:main',

            'test_sub = des_bot.test_sub:main'
        ],
    },
)
