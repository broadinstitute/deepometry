import glob
import os

import click
import numpy
import pandas
import pkg_resources
import skimage.io


@click.command(
    "extract",
    help="""\
Extract features from a trained model.

INPUT should be a directory or list of directories. Subdirectories of INPUT directories are class labels and \
subdirectory contents are image data as NPY arrays.\
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
    "--model-directory",
    default=None,
    help="Directory containing model checkpoints, metrics, and metadata.",
    type=click.Path(exists=True)
)
@click.option(
    "--model-name",
    default=None,
    help="A unique identifier for referencing this model.",
    type=click.STRING
)
@click.option(
    "--output-directory",
    default=pkg_resources.resource_filename("deepometry", "data"),
    help="Output directory for extracted data.",
    type=click.Path()
)
@click.option(
    "--sprites",
    help="Export sprites. Images must be grayscale or RGB.",
    is_flag=True
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
def command(input, batch_size, model_directory, model_name, output_directory, sprites, standardize, verbose):
    directories = [os.path.realpath(directory) for directory in input]

    pathnames = _collect_pathnames(directories)

    labels = sorted(set([os.path.split(os.path.dirname(pathname))[-1] for pathname in pathnames]))

    x, y = _load(pathnames, labels)

    features = _extract(x, y, batch_size, model_directory, model_name, standardize, 1 if verbose else 0)

    metadata = [labels[yi] for yi in y]

    sprite_img = None
    if sprites:
        sprite_img = _images_to_sprite(x)

    _export(features, metadata, sprite_img, output_directory, model_name)


def _collect_pathnames(directories):
    pathnames = []

    for directory in directories:
        subdirectories = glob.glob(os.path.join(directory, "*"))

        pathnames += [glob.glob(os.path.join(subdirectory, "*")) for subdirectory in subdirectories]

    return sum(pathnames, [])


def _export(features, metadata, sprites, directory, name):
    # Export the features, as tsv.
    resource_filename = _resource("features.tsv", directory=directory, prefix=name)
    df = pandas.DataFrame(data=features)
    df.to_csv(resource_filename, header=False, index=False, sep="\t")
    click.echo("Features TSV: {:s}".format(resource_filename))

    # Export label metadata, as tsv.
    resource_filename = _resource("metadata.tsv", directory=directory, prefix=name)
    df = pandas.DataFrame(data=metadata)
    df.to_csv(resource_filename, header=False, index=False, sep="\t")
    click.echo("Metadata TSV: {:s}".format(resource_filename))

    resource_filename = _resource("sprites.png", directory=directory, prefix=name)
    skimage.io.imsave(resource_filename, sprites)
    click.echo("Sprites PNG: {:s}".format(resource_filename))


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


def _images_to_sprite(x):
    """Creates the sprite image along with any necessary padding
    Args:
      x: NxHxW[x3] tensor containing the images.
    Returns:
      x: Properly shaped HxWx3 image with any necessary padding.
    """
    if x.ndim == 3:
        x = numpy.tile(x[..., numpy.newaxis], (1, 1, 1, 3))

    x = x.astype(numpy.float32)

    x_min = numpy.min(x.reshape((x.shape[0], -1)), axis=1)
    x = (x.transpose((1, 2, 3, 0)) - x_min).transpose((3, 0, 1, 2))

    x_max = numpy.max(x.reshape((x.shape[0], -1)), axis=1)
    x = (x.transpose((1, 2, 3, 0)) / x_max).transpose((3, 0, 1, 2))

    # Tile the individual thumbnails into an image.
    n = int(numpy.ceil(numpy.sqrt(x.shape[0])))
    padding = ((0, n ** 2 - x.shape[0]), (0, 0), (0, 0)) + ((0, 0),) * (x.ndim - 3)
    x = numpy.pad(x, padding, mode="constant", constant_values=0)
    x = x.reshape((n, n) + x.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, x.ndim + 1)))
    x = x.reshape((n * x.shape[1], n * x.shape[3]) + x.shape[4:])
    x = (x * 255).astype(numpy.uint8)

    return x


def _load(pathnames, labels):
    x = numpy.empty((len(pathnames),) + _shape(pathnames[0]), dtype=numpy.uint8)

    y = numpy.empty((len(pathnames),), dtype=numpy.uint8)

    label_to_index = {label: index for index, label in enumerate(sorted(labels))}

    for index, pathname in enumerate(pathnames):
        label = os.path.split(os.path.dirname(pathname))[-1]

        x[index] = numpy.load(pathname)

        y[index] = label_to_index[label]

    return x, y


def _resource(filename, directory, prefix=None):
    if prefix is None:
        resource_filename = filename
    else:
        resource_filename = "{:s}_{:s}".format(prefix, filename)

    return os.path.join(directory, resource_filename)


def _shape(pathname):
    return numpy.load(pathname).shape