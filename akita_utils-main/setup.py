"""Python setup.py for akita_utils package"""
import io
import os
from setuptools import find_packages, setup


PKG_NAME = "akita_utils"
README_PATH = "README.md"
INSTALL_DEPS_PATH = "requirements.txt"
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
]


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("akita_utils", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]
 

setup(
    name="akita_utils",
    version=read("akita_utils", "VERSION"),
    description="Awesome akita_utils created by Fudenberg-Research-Group",
    url="https://github.com/Fudenberg-Research-Group/akita_utils/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Fudenberg-Research-Group",
    packages=find_packages(exclude=["tests", ".github"]),
    entry_points={
        "console_scripts": ["akita_utils = akita_utils.__main__:main"]
    },
    classifiers=CLASSIFIERS,
    python_requires=">=3.7",
    install_requires=read_requirements(INSTALL_DEPS_PATH),
    extras_require={"test": read_requirements("requirements-test.txt")},
)
