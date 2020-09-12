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

    MODEL_FILE absolute path to the .h5 model file that was saved after model training session.
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
    "--layer",
    default="pool5",
    help="Name of the feature extraction layer, e.g. res4a_relu / res5a_relu / pool5",
    type=click.STRING
)
@click.option(
    "--checkpoint-directory",
    default=None,
    help="Directory containing model checkpoints, metrics, and metadata.",
    type=click.Path(exists=True)
)
@click.option(
    "--model-file",
    default=None,
    help="Location of the saved .h5 model file",
    type=click.STRING
)
@click.option(
    "--output-directory",
    default=None,
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
def command(input, layer, model_file, batch_size, checkpoint_directory, output_directory, sprites, standardize, verbose):
    import deepometry.utils

    directories = [os.path.realpath(directory) for directory in input]

    x, labels, units = deepometry.utils.quickload(directories, convert=False)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)  

    features = _extract(x, units=units, layer=layer, model_file=model_file, batch_size=batch_size, checkpoint_directory=checkpoint_directory, standardize=standardize, name=None, verbose=1 if verbose else 0)

    sprites_img = _sprites(x) if sprites else None

    _export(features, labels, sprites_img, output_directory)


def _export(features, metadata, sprites, output_directory, name=None):
    # Export the features, as tsv.
    resource_filename = _resource("features.tsv", output_directory=output_directory, prefix=name)
    df = pandas.DataFrame(data=features)
    df.to_csv(resource_filename, header=False, index=False, sep="\t")
    click.echo("Features TSV: {:s}".format(resource_filename))

    # Export label metadata, as tsv.
    resource_filename = _resource("metadata.tsv", output_directory=output_directory, prefix=name)
    df = pandas.DataFrame(data=metadata)
    df.to_csv(resource_filename, header=False, index=False, sep="\t")
    click.echo("Metadata TSV: {:s}".format(resource_filename))

    if sprites:
        resource_filename = _resource("sprites.png", output_directory=output_directory, prefix=name)
        skimage.io.imsave(resource_filename, sprites)
        click.echo("Sprites PNG: {:s}".format(resource_filename))


def _extract(x, units, layer, model_file, batch_size, checkpoint_directory, name, standardize, verbose):
    import deepometry.model

    model = deepometry.model.Model(
        directory=checkpoint_directory,
        name=name,
        shape=x.shape[1:],
        units=units
    )

    model.compile()

    return model.extract(x, selected_layer=layer, saved_model_location=model_file, batch_size=batch_size, standardize=standardize, verbose=verbose)


def _resource(filename, output_directory, prefix=None):
    if prefix is None:
        resource_filename = filename
    else:
        resource_filename = "{:s}_{:s}".format(prefix, filename)

    return os.path.join(output_directory, resource_filename)


def _sprites(x):
    import deepometry.visualize

    return deepometry.visualize.images_to_sprite(x)
