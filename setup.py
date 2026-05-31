"""Setup script for the tactical_topology package."""

from setuptools import find_packages, setup

setup(
    name="tactical_topology",
    version="0.1.0",
    description="Football as a Communication Network: tactical topology analysis.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Nikos Raptis",
    url="https://github.com/nraptisss/Football-as-a-Communication-Network",
    license="Apache-2.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
