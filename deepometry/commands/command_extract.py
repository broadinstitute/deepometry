# coding: utf-8

import os

import click
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
    import deepometry.utils

    directories = [os.path.realpath(directory) for directory in input]

    x, labels, units = deepometry.utils.load(directories, convert=False)

    features = _extract(x, units, batch_size, model_directory, model_name, standardize, 1 if verbose else 0)

    sprites_img = _sprites(x) if sprites else None

    _export(features, labels, sprites_img, output_directory, model_name)


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

    if sprites:
        resource_filename = _resource("sprites.png", directory=directory, prefix=name)
        skimage.io.imsave(resource_filename, sprites)
        click.echo("Sprites PNG: {:s}".format(resource_filename))


def _extract(x, units, batch_size, directory, name, standardize, verbose):
    import deepometry.model

    model = deepometry.model.Model(
        directory=directory,
        name=name,
        shape=x.shape[1:],
        units=units
    )

    model.compile()

    return model.extract(x, batch_size=batch_size, standardize=standardize, verbose=verbose)


def _resource(filename, directory, prefix=None):
    if prefix is None:
        resource_filename = filename
    else:
        resource_filename = "{:s}_{:s}".format(prefix, filename)

    return os.path.join(directory, resource_filename)


def _sprites(x):
    import deepometry.visualize

    return deepometry.visualize.images_to_sprite(x)
