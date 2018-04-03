# coding: utf-8

import os.path
import subprocess

import click
import pkg_resources


@click.command(
    "visualize",
    help="""\
Visualize extracted feature embedding with TensorBoard.

FEATURES should be a TSV containing the embedding.
"""
)
@click.argument(
    "features",
    required=True,
    type=click.Path(exists=True)
)
@click.option(
    "--metadata",
    help="Additional fields (e.g., labels) corresponding to data, as a TSV."
         " Single-column metadata should exclude column label. Multi-column"
         " metadata must include column labels.",
    type=click.Path(exists=True)
)
@click.option(
    "--output-directory",
    default=pkg_resources.resource_filename("deepometry", "data"),
    help="Output directory for visualized data.",
    type=click.Path()
)
@click.option(
    "--sprites",
    help="Sprites image.",
    type=click.Path(exists=True)
)
@click.option(
    "--sprites-dim",
    help="Dimension of a sprite.",
    type=click.INT
)
def command(features, metadata, output_directory, sprites, sprites_dim):
    import deepometry.visualize

    log_directory = deepometry.visualize.make_projection(
        features,
        metadata=metadata,
        log_directory=output_directory,
        sprites=sprites,
        sprites_dim=sprites_dim
    )

    click.echo(
        "Starting TensorBoard...\n"
        "Please open your web browser and navigate to the address provided:"
    )

    subprocess.call(["tensorboard", "--logdir", log_directory])


def _validate_tsv(path):
    ext = os.path.splitext(path)[-1]

    if ext != ".tsv":
        raise ValueError(
            "Unsupported extension '{:s}'. Expected '.tsv', got '{:s}'".format(ext, path)
        )
