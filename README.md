# Time-dependent coupled cluster

[![Build Status](https://travis-ci.com/Schoyen/coupled-cluster.svg?token=MvgH7xLNL8iVfczJpp8Q&branch=master)](https://travis-ci.com/Schoyen/coupled-cluster)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


This repository provides coupled cluster solvers for single reference many-body quantum mechanical problems. An emphasize has been put on time-dependent systems, but as a consequence we also provide coupled cluster solvers for groundstate computations.

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
