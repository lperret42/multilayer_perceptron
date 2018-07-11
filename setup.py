from setuptools import setup, find_packages, Extension

setup(name="multilayer-perceptron",
      version="0.0.1",
      author="Lucas Perret",
      description="python3 implementation of a multilayer perceptron for\
                   multi classification",
      install_requires=[
          'numpy >=1.14',
      ],
      packages=find_packages(exclude=['examples']),
     )
