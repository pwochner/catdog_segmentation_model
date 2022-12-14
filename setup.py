#!/usr/bin/env python
from setuptools import find_packages, setup

requirements = []
with open("requirements.txt") as f:
    for line in f:
        stripped = line.split("#")[0].strip()
        if len(stripped) > 0:
            requirements.append(stripped)

setup(
    name="catdog_segmentation_model",
    version="0.0.1",
    description="scivision plugin, using a UNet to segment cat/dog images",
    url="TODO: INSERT URL OF THE MODEL GITHUB REPO HERE",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.7",
)
