from setuptools import setup, find_packages

setup(
    name="Coupled cluster",
    version="0.2.7",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "opt_einsum",
        "quantum-systems @ git+https://github.com/Schoyen/quantum-systems",
    ],
)
