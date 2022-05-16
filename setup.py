# from setuptools import find_packages
from setuptools import setup, find_packages

setup(
    name="seesaw",
    version="1.3.0",
    url="git@github.com:orm011/seesaw.git",
    author="Oscar Moll",
    author_email="orm@csail.mit.edu",
    description="implementation of the seesaw system for interactive image database search",
    packages=find_packages(where="seesaw"),
    # install_requires=[],  # install from pyproject.toml using pip -r pyproject.toml
    # extras_require={
    #     "dev": [
    #         "plotnine",
    #         "pydantic-to-typescript",
    #         "bokeh",
    #         "py-spy",
    #         "pytest",
    #         "line-profiler",
    #     ]
    # },
    python_requires=">=3.8",
)
