from setuptools import setup, find_packages

setup(
    name='kristools',
    version='0.1',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'tensorflow',
        'gym',
        'scipy',
        'wesutils>=0.0.7',
        'matplotlib'
    ]
)
