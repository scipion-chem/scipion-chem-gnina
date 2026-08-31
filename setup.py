
import re
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(here, 'requirements.txt')) as f:
    requirements = f.read().splitlines()

# Single source of truth for the version: gnina/constants.py. It is read
# instead of imported because the plugin dependencies (pwchem, pwem...) are not
# necessarily installed yet when setup.py runs.
with open(path.join(here, 'gnina', 'constants.py'), encoding='utf-8') as f:
    version = re.search(r"^ALPHA_VERSION\s*=\s*['\"]([^'\"]+)['\"]", f.read(), re.M).group(1)

setup(
    name='scipion-chem-gnina',
    version=version,
    description='Scipion plugin for GNINA molecular docking with CNN scoring',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/scipion-chem/scipion-chem-gnina',
    author='Joaquin Algorta',
    author_email='scipion@cnb.csic.es',
    keywords=['scipion', 'docking', 'gnina', 'cnn', 'deep-learning', 'scipion-3'],
    packages=find_packages(),
    install_requires=[requirements],
    include_package_data=True,
    package_data={
        'gnina': ['icon.png', 'protocols.conf', 'testData.json'],
    },
    entry_points={
        'pyworkflow.plugin': 'gnina = gnina'
    },
)
