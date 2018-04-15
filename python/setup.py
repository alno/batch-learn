from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name='batch-learn',
    version='0.1.0',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    ext_modules=cythonize(
         "batch_learn/writer.pyx",
         #sources=["Rectangle.cpp"],  # additional source file(s)
         language="c++",
    )
)
