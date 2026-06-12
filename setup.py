
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
    author='Joaquin Algorta Bove',
    author_email='your@email.com',
    keywords='scipion docking gnina cnn deep-learning',
    packages=find_packages(),
    install_requires=[requirements],
    include_package_data=True,
    package_data={
        'gnina': ['gnina_logo.png'],
    },
    entry_points={
        'pyworkflow.plugin': 'gnina = gnina'
    },
)