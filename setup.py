from setuptools import find_packages, setup


setup(
    name="laptq_yolov3",
    packages=find_packages(
        include=[
            "laptq_yolov3",
        ]
    ),
    version="0.1.0",
    description="YOLOv3",
    author="Tran Quoc Lap",

    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)
