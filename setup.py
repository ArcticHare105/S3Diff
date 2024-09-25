from setuptools import setup, find_packages

setup(
    name='SRTurbo',
    version='0.0.1',
    description='',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)