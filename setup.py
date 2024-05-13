from setuptools import setup, find_packages

description = """Visibility based Tapered Gridded Estimator implementation to estimate the power spectrum with uncertainties."""

setup(
    name="TGEpy",
    version="0.0.1",
    description=description,
    url="http://github.com/chrisfinlay/TGEpy",
    author="Chris Finlay",
    author_email="christopher.finlay@unige.ch",
    license="MIT",
    packages=find_packages(),
    install_requires=["numba", "numpy", "scipy"],
    zip_safe=False,
)