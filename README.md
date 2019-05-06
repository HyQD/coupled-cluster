# Time-dependent coupled cluster

[![Build Status](https://travis-ci.com/Schoyen/coupled-cluster.svg?token=MvgH7xLNL8iVfczJpp8Q&branch=master)](https://travis-ci.com/Schoyen/coupled-cluster)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


This repository provides coupled cluster solvers for single reference many-body quantum mechanical problems. An emphasis has been put on time-dependent systems, but as a consequence we also provide coupled cluster solvers for groundstate computations.

## Implemented solvers
Below follows a list of solvers this repository provides.

### Ground state solvers
1. [Coupled cluster doubles](https://github.com/Schoyen/coupled-cluster/blob/master/coupled_cluster/ccd/ccd.py) method with static orbitals.
2. [Coupled cluster singles-and-doubles](https://github.com/Schoyen/coupled-cluster/blob/master/coupled_cluster/ccsd/ccsd.py) method with static orbitals.
3. [Orbital-adaptive coupled cluster doubles](https://github.com/Schoyen/coupled-cluster/blob/master/coupled_cluster/ccd/oaccd.py) method, also known as the non-orthogonal coupled cluster doubles method. Note that this method requires orthonormal basis functions, e.g., the Hartree-Fock basis.

### Time-dependent solvers
These solvers use the ground state calculations from their corresponding ground state classes as starting point of the time-evolution.
1. [Time-dependent coupled cluster doubles](https://github.com/Schoyen/coupled-cluster/blob/master/coupled_cluster/ccd/tdccd.py) method with static orbitals.
2. [Time-dependent coupled cluster singles-and-doubles](https://github.com/Schoyen/coupled-cluster/blob/master/coupled_cluster/ccsd/tdccsd.py) method with static orbitals.
3. [Orbital-adaptive time-dependent coupled cluster doubles](https://github.com/Schoyen/coupled-cluster/blob/master/coupled_cluster/ccd/oatdccd.py) method, also know as the non-orthogonal time-dependent coupled cluster doubles method as we by default avoid truncating the basis when computing the Q-space equations. This method also requires an orthonormal basis of single-particle functions as it uses the orbital-adaptive coupled cluster doubles method as an initial starting point.

## Installation
This project can be installed by running:

```bash
pip install git+https://github.com/Schoyen/coupled-cluster.git
```

During development it is a good idea to create a _conda environment_ such that all dependencies gets installed correctly. This is easiest done by executing:

```bash
conda env create -f environment.yml
source activate cc
```

Once you are done, you can deactivate the environment by:

```bash
source deactivate
```

If the environment changes you can update the new changes by:

```bash
conda env update -f environment.yml
```
