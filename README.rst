deepometry
==========

Image classification for imaging flow cytometry.

Installation
------------

Get started with ``deepometry`` by downloading the source code and installing its dependencies. ``deepometry``
supports Python 2.7 and 3.5+. However, parsing data from ``CIF`` files to a data format supported for training,
evaluation, and prediction is only supported in Python 2.7.

::

    git clone https://github.com/broadinstitute/deepometry.git
    cd deepometry
    pip install --upgrade pip setuptools wheel
    pip install --upgrade numpy
    pip install --upgrade .

If you want to install ``deepometry`` in development mode, run:

::

    pip install --upgrade --editable .[development]

Use
---

Display a list of available subcommands:

::

    deepometry --help

To display subcommand use and options:

::

    deepometry SUBCOMMAND --help

Examples
--------

The ``examples`` directory contains Jupyter notebooks illustrating the Python interface. To get started with Jupyter:

::

    pip install --upgrade jupyter
    cd examples
    jupyter notebook