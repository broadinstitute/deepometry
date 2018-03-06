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

    Computation is done in batches.
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
    "--verbose",
    is_flag=True
)
@click.option(
    "--exclude",
    default=None,
    help="A comma-separated list of prefixes (string) specifying the files that needs to be held off the testing dataset."
         " E.g., \"'patient_A', 'patient_X'\". All files will be collected for testing if this flag is omitted."
)
@click.option(
    "--samples",
    default=None,
    help="Number of objects to be collected per class label to pool into testing dataset."
         "This setting is useful to limit certain amount of datapoint to be displayed in unsupervised PCA/t-SNE plots."
         "All numpy arrays will be collected for testing if this flag is omitted."
)
def command(input, exclusion, nsamples, batch_size, directory, name, verbose):
    directories = [os.path.realpath(directory) for directory in input]

    pathnames = _sample(directories, nsamples)

    labels = set([os.path.split(os.path.dirname(pathname))[-1] for pathname in pathnames])

    x, y = _load(pathnames, labels, exclusion)

    metrics_names, metrics = _evaluate(x, y, batch_size, directory, name, 1 if verbose else 0)

    for metric_name, metric in zip(metrics_names, metrics):
        click.echo("{metric_name}: {metric}".format(**{
            "metric_name": metric_name,
            "metric": metric
        }))


def _sample(directories, nsamples):
    pathnames = []

    for directory in directories:
        subdirectories = sorted(glob.glob(os.path.join(directory, "*")))

        # transform the files of the same label into directory
        subdirectory_pathnames = [glob.glob(os.path.join(subdirectory, "*.npy")) for subdirectory in subdirectories]

        pathnames += [list(numpy.random.permutation(pathnames)[:nsamples]) for pathnames in subdirectory_pathnames]

    return sum(pathnames, [])


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


def _load(pathnames, labels, exclusion):

    pathnames = [x for x in pathnames if numpy.all([not z in x for z in exclusion])]

    x = numpy.empty((len(pathnames),) + _shape(pathnames[0]), dtype=numpy.uint8)

    y = numpy.empty((len(pathnames),), dtype=numpy.uint8)

    label_to_index = {label: index for index, label in enumerate(sorted(labels))}

    for index, pathname in enumerate(pathnames):
        if os.path.isfile(pathname): # in case there is a mixture of directories and files

            label = os.path.split(os.path.dirname(pathname))[-1]

            x[index] = numpy.load(pathname)

            y[index] = label_to_index[label]

    return x, y


def _shape(pathname):
    return numpy.load(pathname).shape
