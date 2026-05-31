"""Setup script for the tactical_topology package."""

from setuptools import find_packages, setup

setup(
    name="tactical_topology",
    version="0.0.1",
    description="Football as a Communication Network: tactical topology analysis.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
