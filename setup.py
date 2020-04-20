from setuptools import setup, find_packages

setup(
    name="Coupled cluster",
    version="0.2.2",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "quantum-systems @ git+https://github.com/Schoyen/quantum-systems",
    ],
)
