# coding: utf-8

import os

import click


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
    import deepometry.utils

    directories = [os.path.realpath(directory) for directory in input]

    x, y, units = deepometry.utils.load(directories, sample=samples)

    metrics_names, metrics = _evaluate(x, y, units, batch_size, directory, name, 1 if verbose else 0)

    for metric_name, metric in zip(metrics_names, metrics):
        click.echo("{metric_name}: {metric}".format(**{
            "metric_name": metric_name,
            "metric": metric
        }))


def _evaluate(x, y, units, batch_size, directory, name, verbose):
    import deepometry.model

    model = deepometry.model.Model(
        directory=directory,
        name=name,
        shape=x.shape[1:],
        units=units
    )

    model.compile()

    metrics = model.evaluate(x, y, batch_size=batch_size, verbose=verbose)

    return model.model.metrics_names, metrics
