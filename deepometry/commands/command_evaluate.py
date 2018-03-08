import glob
import os

import click
import numpy


@click.command(
    "evaluate",
    help="""
    Compute loss & accuracy values.

    INPUT should be a directory or list of directories. Subdirectories of INPUT directories are class labels and
    subdirectory contents are image data as NPY arrays.

    Computation is done in batches."""
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
    help="Number of samples evaluated per batch.",
    type=click.INT
)
@click.option(
    "--directory",
    default=None,
    help="Output directory for model checkpoints, metrics, and metadata.",
    type=click.Path(exists=True)
)
@click.option(
    "--name",
    default=None,
    help="A unique identifier for referencing this model.",
    type=click.STRING
)
@click.option(
    "--samples",
    default=None,
    help="Number of objects to be collected per class label to pool into testing data set."
         " This setting is useful to limit the number of data points displayed in unsupervised PCA/t-SNE plots."
         " All data will be collected for testing if this flag is omitted.",
    type=click.INT
)
@click.option(
    "--verbose",
    is_flag=True
)
def command(input, batch_size, directory, name, samples, verbose):
    directories = [os.path.realpath(directory) for directory in input]

    pathnames = _sample(directories, samples)

    labels = set([os.path.split(os.path.dirname(pathname))[-1] for pathname in pathnames])

    x, y = _load(pathnames, labels)

    metrics_names, metrics = _evaluate(x, y, batch_size, directory, name, 1 if verbose else 0)

    for metric_name, metric in zip(metrics_names, metrics):
        click.echo("{metric_name}: {metric}".format(**{
            "metric_name": metric_name,
            "metric": metric
        }))


def _evaluate(x, y, batch_size, directory, name, verbose):
    import deepometry.model

    model = deepometry.model.Model(
        directory=directory,
        name=name,
        shape=x.shape[1:],
        units=len(numpy.unique(y))
    )

    model.compile()

    metrics = model.evaluate(x, y, batch_size=batch_size, verbose=verbose)

    return model.model.metrics_names, metrics


def _filter(paths):
    return [path for path in paths if os.path.splitext(path)[-1].lower() == ".npy"]


def _load(pathnames, labels):
    x = numpy.empty((len(pathnames),) + _shape(pathnames[0]), dtype=numpy.uint8)

    y = numpy.empty((len(pathnames),), dtype=numpy.uint8)

    label_to_index = {label: index for index, label in enumerate(sorted(labels))}

    for index, pathname in enumerate(pathnames):
        label = os.path.split(os.path.dirname(pathname))[-1]

        x[index] = numpy.load(pathname)

        y[index] = label_to_index[label]

    return x, y


def _sample(directories, nsamples):
    samples = []

    for directory in directories:
        # List subdirectories, filtering non-directory files
        subdirectories = sorted([
            directory for directory in glob.glob(os.path.join(directory, "*")) if os.path.isdir(directory)
        ])

        # Remove files that aren't NPYs
        subdirectory_paths = [_filter(glob.glob(os.path.join(subdirectory, "*"))) for subdirectory in subdirectories]

        samples += [
            list(numpy.random.permutation(pathnames)[:nsamples]) for pathnames in subdirectory_paths
        ]

    return sum(samples, [])


def _shape(pathname):
    return numpy.load(pathname).shape
