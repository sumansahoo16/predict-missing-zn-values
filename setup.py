"""Unearthed Submission build."""
from setuptools import find_packages, setup

setup(
    name="pressure-predictor",
    py_modules=[
        "preprocess",
        "train",
        # note predict and score modules are not required to be submitted
        # add any additional modules you want included in your submission here
        "ensemble_model"
    ],
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    version="1.0.0",
    description="Pressure Predictor Challenge Template",
    author="Unearthed Solutions",
    author_email="info@unearthed.solutions",
)
