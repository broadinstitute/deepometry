import glob
import os

import click
import numpy
import pandas
import pkg_resources


@click.command(
    "extract",
    help="""
    Extract features from a trained model.
    
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
    help="Number of samples evaluated per batch.",
    type=click.INT
)
@click.option(
    "--directory",
    default=None,
    help="Directory containing model checkpoints, metrics, and metadata.",
    type=click.Path(exists=True)
)
@click.option(
    "--name",
    default=None,
    help="A unique identifier for referencing this model.",
    type=click.STRING
)
@click.option(
    "--standardize",
    help="Center to the mean and component wise scale to unit variance.",
    is_flag=True
)
@click.option(
    "--verbose",
    is_flag=True
)
def command(input, batch_size, directory, name, standardize, verbose):
    directories = [os.path.realpath(directory) for directory in input]

    pathnames = _collect_pathnames(directories)

    labels = sorted(set([os.path.split(os.path.dirname(pathname))[-1] for pathname in pathnames]))

    x, y = _load(pathnames, labels)

    features = _extract(x, y, batch_size, directory, name, standardize, 1 if verbose else 0)

    metadata = [labels[yi] for yi in y]

    _export(features, metadata, directory, name)


def _collect_pathnames(directories):
    pathnames = []

    for directory in directories:
        subdirectories = glob.glob(os.path.join(directory, "*"))

        pathnames += [glob.glob(os.path.join(subdirectory, "*")) for subdirectory in subdirectories]

    return sum(pathnames, [])


def _export(features, metadata, directory, name):
    # Export the features, as tsv.
    resource_filename = _resource("features.tsv", directory=directory, name=name)
    df = pandas.DataFrame(data=features)
    df.to_csv(resource_filename, index=False, sep="\t")
    click.echo("Features TSV: {:s}".format(resource_filename))

    # Export label metadata, as tsv.
    resource_filename = _resource("metadata.tsv", directory=directory, name=name)
    df = pandas.DataFrame(data=metadata, columns=["Label"])
    df.to_csv(resource_filename, index=False, sep="\t")
    click.echo("Metadata TSV: {:s}".format(resource_filename))


def _extract(x, y, batch_size, directory, name, standardize, verbose):
    import deepometry.model

    model = deepometry.model.Model(
        directory=directory,
        name=name,
        shape=x.shape[1:],
        units=len(numpy.unique(y))
    )

    model.compile()

    return model.extract(x, batch_size=batch_size, standardize=standardize, verbose=verbose)


def _load(pathnames, labels):
    x = numpy.empty((len(pathnames),) + _shape(pathnames[0]), dtype=numpy.uint8)

    y = numpy.empty((len(pathnames),), dtype=numpy.uint8)

    label_to_index = {label: index for index, label in enumerate(sorted(labels))}

    for index, pathname in enumerate(pathnames):
        label = os.path.split(os.path.dirname(pathname))[-1]

        x[index] = numpy.load(pathname)

        y[index] = label_to_index[label]

    return x, y


def _resource(filename, directory=None, name=None):
    if name is None:
        resource_filename = filename
    else:
        resource_filename = "{:s}_{:s}".format(name, filename)

    if directory is None:
        return pkg_resources.resource_filename(
            "deepometry",
            os.path.join("data", resource_filename)
        )

    return os.path.join(directory, resource_filename)


def _shape(pathname):
    return numpy.load(pathname).shape