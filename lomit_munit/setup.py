# !/usr/bin/env python  # NOQA

from setuptools import setup, find_packages

setup(
    name="replmunit",
    version="1.2.9",
    description="Ripples MUNIT framework",
    url="https://github.com/ElementAI/repl_munit",
    author="Ripples Team",
    author_email="ripples@elementai.com",
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_data={
        "": ["*.jpg"],
        # And include any *.yaml files found in the 'configs' subdirectory, too:
        "replmunit": ["configs/*.yaml", "datasets/*/*/*", "datasets/*/*"],
    },
    install_requires=[
    ]
)