# coding: utf-8

import logging
import os.path

import numpy
import pandas
import pkg_resources


logger = logging.getLogger(__name__)


def make_projection(features, metadata=None, log_directory=None, sprites=None, sprites_dim=None):
    """

    :param features: Embedded features TSV, one row per sample.
    :param metadata: Additional fields (e.g., labels) corresponding to sample data,
        as a TSV. Single-column metadata should exclude column label. Multi-column
        metadata must include column labels.
    :param log_directory: Output directory for visualized data.
    :param sprites: Path to sprites image.
    :param sprites_dim: Dimension ``(sprites_dim, sprites_dim)`` of a sprite.
    :return: Output directory for visualized data.
    """
    try:
        import tensorflow
    except ImportError as error:
        logger.error(
            "Tensorflow is required for `deepometry.visualze.make_projection`."
            " To install tensorflow, run `pip install tensorflow`."
        )

        raise error

    _validate_tsv(features)
    features_df = pandas.read_csv(features, header=None, sep="\t")
    features_tf = tensorflow.Variable(features_df.values, name="features")

    if not log_directory:
        log_directory = pkg_resources.resource_filename("deepometry", "data")

    with tensorflow.Session() as session:
        saver = tensorflow.train.Saver([features_tf])
        session.run(features_tf.initializer)
        saver.save(session, os.path.join(log_directory, "embedding.ckpt"))

    config = tensorflow.contrib.tensorboard.plugins.projector.ProjectorConfig()

    embedding = config.embeddings.add()
    embedding.tensor_name = features_tf.name

    if metadata:
        _validate_tsv(metadata)
        embedding.metadata_path = os.path.realpath(metadata)

    if sprites:
        if not sprites_dim:
            raise ValueError(
                "Missing required parameter: `sprites_dim`."
                " Please supply a valid integer for `sprites_dim`."
            )

        embedding.sprite.image_path = os.path.realpath(sprites)
        embedding.sprite.single_image_dim.extend([sprites_dim, sprites_dim])

    tensorflow.contrib.tensorboard.plugins.projector.visualize_embeddings(
        tensorflow.summary.FileWriter(log_directory),
        config
    )

    return log_directory


def images_to_sprite(x):
    """
    Creates the sprite image along with any necessary padding.

    :param x: NxHxW[x3] tensor containing the images.
    :return: Properly shaped HxWx3 image with any necessary padding.
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


def _validate_tsv(path):
    ext = os.path.splitext(path)[-1]

    if ext != ".tsv":
        raise ValueError(
            "Unsupported extension '{:s}'. Expected '.tsv', got '{:s}'".format(ext, path)
        )
