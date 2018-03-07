import glob
import os

import click
import numpy


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
    "--exclude",
    default=None,
    help="A comma-separated list of prefixes of files to withhold from the training dataset."
         " E.g., \"patient_A, patient_X\". All files will be collected for fitting if this flag is omitted."
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
def command(input, batch_size, directory, epochs, exclude, name, validation_split, verbose):
    import deepometry.model

    directories = [os.path.realpath(directory) for directory in input]

    pathnames = _sample(directories)

    labels = set([os.path.split(os.path.dirname(pathname))[-1] for pathname in pathnames])

    x, y = _load(pathnames, labels, exclude=exclude)

    model = deepometry.model.Model(
        directory=directory,
        name=name,
        shape=x.shape[1:],
        units=len(labels)
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


def _load(pathnames, labels, exclude=None):
    if exclude:
        pathnames = [x for x in pathnames if exclude not in x]

    x = numpy.empty((len(pathnames),) + _shape(pathnames[0]), dtype=numpy.uint8)

    y = numpy.empty((len(pathnames),), dtype=numpy.uint8)

    label_to_index = {label: index for index, label in enumerate(sorted(labels))}

    for index, pathname in enumerate(pathnames):
        if os.path.isfile(pathname):  # in case there is a mixture of directories and files
            label = os.path.split(os.path.dirname(pathname))[-1]

            x[index] = numpy.load(pathname)

            y[index] = label_to_index[label]

    return x, y


def _sample(directories):
    sampled_pathnames = []

    for directory in directories:
        subdirectories = sorted(glob.glob(os.path.join(directory, "*")))

        subdirectory_pathnames = [glob.glob(os.path.join(subdirectory, "*")) for subdirectory in subdirectories]

        nsamples = int(numpy.median([len(pathnames) for pathnames in subdirectory_pathnames]))

        sampled_pathnames += [
            list(numpy.random.permutation(pathnames)[:nsamples]) for pathnames in subdirectory_pathnames
        ]

    return sum(sampled_pathnames, [])


def _shape(pathname):
    return numpy.load(pathname).shape
