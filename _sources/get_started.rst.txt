Getting started
===============

Installation
------------

The module can be easily installed with ``pip``::

    pip install git+https://github.com/Schoyen/coupled-cluster.git

For development, or if it is your preference, we have provideded
a specification to automatically create a ``conda`` environment
with all dependencies. It can easily be created like this,::

    conda env create -f environment.yml
    conda activate cc

This project has several dependencies, see
`requirements.txt <https://github.com/Schoyen/coupled-cluster/blob/master/requirements.txt>`_.
Of the dependecies, one is a custom module called
`quantum-systems <https://github.com/Schoyen/quantum-systems>`_.
