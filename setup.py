
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='scipion-chem-gnina',
    version='0.1.0',
    description='Scipion plugin for GNINA molecular docking with CNN scoring',
    long_description=long_description,
    url='https://github.com/scipion-chem/scipion-chem-gnina',
    author='Joaquin Algorta',
    author_email='scipion@cnb.csic.es',
    keywords=['scipion', 'docking', 'gnina', 'cnn', 'deep-learning', 'scipion-3'],
    packages=find_packages(),
    install_requires=[requirements],
    include_package_data=True,
    package_data={
        'gnina': ['icon.png', 'protocols.conf'],
    },
    entry_points={
        'pyworkflow.plugin': 'gnina = gnina'
    },
)