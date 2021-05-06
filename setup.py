from setuptools import find_packages
from distutils.core import setup
setup(
    name = 'vls',
    version = '1.2.0',
    url = 'git@github.com:orm011/vls.git',
    author = 'Oscar Moll',
    author_email = 'orm@csail.mit.edu',
    description = 'implementation of the vls research project',
    packages = find_packages(),
    install_requires = [],
)
