from setuptools import find_packages
from distutils.core import setup
import pkg_resources
import os

setup(
    name = 'seesaw',
    version = '1.3.0',
    url = 'git@github.com:orm011/seesaw.git',
    author = 'Oscar Moll',
    author_email = 'orm@csail.mit.edu',
    description = 'implementation of the seesaw system for interactive image database search',
    packages = find_packages(include=['seesaw']),
    install_requires = [
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    extras_require = {
            'dev': ['plotnine',
                    'pydantic-to-typescript',
                    'bokeh',
                    'py-spy',
                    'pytest',
                    'line-profiler'
                    ]
    },
    python_requires='>=3.8',
)
