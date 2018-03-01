# coding: utf-8

import os

import click


@click.command(
    "fit",
    help="""
    Train a model.

    INPUT should be a directory or list of directories. Subdirectories of INPUT directories are class labels and
    subdirectory contents are image data as NPY arrays.
    """
)
@click.argument(
    "input",
    nargs=-1,
    required=True,
    type=click.Path(exists=True)
)
@click.option(
    "--batch-size",
    default=32,
    help="Number of samples per gradient update.",
    type=click.INT
)
@click.option(
    "--directory",
    default=None,
    help="Output directory for model checkpoints, metrics, and metadata.",
    type=click.Path(exists=True)
)
@click.option(
    "--epochs",
    default=128,
    help="Number of iterations over training data.",
    type=click.INT
)
@click.option(
    "--name",
    default=None,
    help="A unique identifier for referencing this model.",
    type=click.STRING
)
@click.option(
    "--validation-split",
    default=0.2,
    help="Fraction of training data withheld for validation.",
    type=click.FLOAT
)
@click.option(
    "--verbose",
    is_flag=True
)
def command(input, batch_size, directory, epochs, name, validation_split, verbose):
    import deepometry.utils

    directories = [os.path.realpath(directory) for directory in input]

    x, y, units = deepometry.utils.load(directories, sample=True)

    _fit(x, y, units, batch_size, directory, epochs, name, validation_split, verbose)


def _fit(x, y, units, batch_size, directory, epochs, name, validation_split, verbose):
    import deepometry.model

    model = deepometry.model.Model(
        directory=directory,
        name=name,
        shape=x.shape[1:],
        units=units
    )

    model.compile()

    model.fit(
        x,
        y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        verbose=1 if verbose else 0
    )
