from setuptools import setup, find_packages

setup(
    name="coupled-cluster",
    version="0.2.7",
    packages=find_packages(),
    install_requires=[
        "scipy",
        "opt_einsum",
        "quantum-systems @ git+https://github.com/HyQD/quantum-systems",
    ],
)
