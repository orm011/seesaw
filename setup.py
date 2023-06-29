from setuptools import setup, find_packages

setup(
    name="seesaw",
    version="1.3.0",
    url="git@github.com:orm011/seesaw.git",
    author="Oscar Moll",
    author_email="orm@csail.mit.edu",
    description="implementation of the seesaw system for interactive image database search",
    packages=find_packages(where="seesaw"),
    # install_requires=[],  # installed from pyproject.toml
    python_requires=">=3.10",
)
