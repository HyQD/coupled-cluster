# Time-dependent coupled cluster

[![Build Status](https://travis-ci.com/Schoyen/coupled-cluster.svg?token=MvgH7xLNL8iVfczJpp8Q&branch=master)](https://travis-ci.com/Schoyen/coupled-cluster)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


This repository provides coupled cluster solvers for single reference many-body quantum mechanical problems. An emphasis has been put on time-dependent systems, but as a consequence we also provide coupled cluster solvers for groundstate computations. See the [documentation](https://schoyen.github.io/coupled-cluster/).

## Implemented solvers
Below follows a list of solvers this repository provides.

### Ground state solvers
1. [Coupled cluster doubles](https://github.com/Schoyen/coupled-cluster/blob/master/coupled_cluster/ccd/ccd.py) method with static orbitals.
2. [Coupled cluster singles-and-doubles](https://github.com/Schoyen/coupled-cluster/blob/master/coupled_cluster/ccsd/ccsd.py) method with static orbitals.
3. [Orbital-adaptive coupled cluster doubles](https://github.com/Schoyen/coupled-cluster/blob/master/coupled_cluster/ccd/oaccd.py) method, also known as the non-orthogonal coupled cluster doubles method.

### Time-dependent solvers
These solvers use the ground state calculations from their corresponding ground state classes as starting point of the time-evolution.
1. [Time-dependent coupled cluster doubles](https://github.com/Schoyen/coupled-cluster/blob/master/coupled_cluster/ccd/tdccd.py) method with static orbitals.
2. [Time-dependent coupled cluster singles-and-doubles](https://github.com/Schoyen/coupled-cluster/blob/master/coupled_cluster/ccsd/tdccsd.py) method with static orbitals.
3. [Orbital-adaptive time-dependent coupled cluster doubles](https://github.com/Schoyen/coupled-cluster/blob/master/coupled_cluster/ccd/oatdccd.py) method, also know as the non-orthogonal time-dependent coupled cluster doubles method as we by default avoid truncating the basis when computing the Q-space equations.

## Installation
This project can be installed via `pip` by running:
```bash
pip install git+https://github.com/Schoyen/coupled-cluster.git
```
Optionally add a `-U` or `--upgrade` to ensure that previously installed versions gets upgraded.
If you have set up ssh-key authentication to Github you can run:
```bash
pip install git+ssh://git@github.com/Schoyen/coupled-cluster.git
```
In a project both options can be added to a `requirements.txt`-file (sans the `pip install`-part).

### Installation via Pipenv
If you choose to use Pipenv the project can be included in the `Pipfile` by running:
```bash
pipenv install -e git+ssh://git@github.com/Schoyen/coupled-cluster.git#egg=coupled-cluster
```
for installation via ssh, and by running:
```bash
pipenv install -e git+https://github.com/Schoyen/coupled-cluster.git#egg=coupled-cluster
```
for installation via https.

## Development
During development it is a good idea to create an environment such that all dependencies gets installed correctly.
In the following examples we assume that you have cloned the repository and that you are standing in the top directory.
The recommended way to do development is then to use a virtual environment alongside pip.
This is achieved by running:
```bash
python -m venv venv
```
which sets up a virtual environment called `venv`.
To activate this environment run:
```bash
source activate venv/bin/activate
```
It is recommended to upgrade pip the first time the environment has been activated:
```bash
pip install -U pip
```
The development dependencies for this project can now be installed by:
```bash
pip install -r requirements.txt
```
and the project itself by:
```bash
pip install .
```
Here as well, you can add a `-U` or `--upgrade` to ensure that the project is updated to the latest version.

### Pipenv
If you wish to use Pipenv the dependencies can be installed via:
```bash
pipenv install
```
and the environment can be activated by:
```bash
pipenv shell
```
Once the environment is activated install the project by:
```bash
pip install .
```

### Anaconda
If you wish to use a _conda environment_ this is easiest done by executing:

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
