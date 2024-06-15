from setuptools import setup, find_packages

setup(
    name='arrow_detection',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
    ],
    entry_points={
        'console_scripts': [
            'arrow_detection=src.arrow_recognition:main',
        ],
    },
)
