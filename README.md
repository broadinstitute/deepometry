<h1>Deepometry</h1>

Deep learning-based image classification and featurization for imaging (flow) cytometry.

This workflow was originally built for imaging flow cytometry data but can be readily adapted for microscopic images of isolated single objects. The modified implementation of ResNet50 allows researchers to use any image frame size and any number of color channels.

<h2>Installation</h2>

A full installation guide can be found [**here**](https://www.evernote.com/shard/s730/sh/f60a69be-cb67-45f7-8054-c71035478b5e/5d7ca2a094dd33a599ef57715403cead). Briefly, the following dependencies are needed:
- Python 3.6
- Tensorflow-gpu 1.9.0
- Keras 2.1.5
- Numpy 1.18.1
- Scipy 1.4.1
- Keras-resnet 0.0.7
- Java JDK 8.0 or 11.0
- Python-bioformats 1.5.2

Once the above dependencies are installed, clone this ``Deepometry`` repository by :

    git clone https://github.com/broadinstitute/deepometry.git
    cd deepometry
    pip install .

If you want to install ``deepometry`` in development mode, run:

    pip install --editable .[development]


<h2>Use</h2>
---
Execute ``Deepometry`` functions through any of the following interfaces:

<h3>CLI</h3>
============

Switch to [CLI](https://github.com/broadinstitute/deepometry/tree/CLI) branch:

    git checkout CLI

Display a list of available subcommands:

    deepometry --help

To display subcommand use and options:

    deepometry SUBCOMMAND --help


<h3>IPYNB</h3>
============

- Use these [Jupyter notebooks](https://github.com/broadinstitute/deepometry/tree/master/examples)

<h3>GUI (recommended)</h3>
============
Switch to [GUI](https://github.com/broadinstitute/deepometry/tree/GUI) branch:
    
    git checkout GUI

    python Deepometry_GUI.py


Open a web-browser, navigate to **http://localhost:5000/**

![Full view GUI](https://github.com/broadinstitute/deepometry/raw/GUI/assets/full_GUI.png)