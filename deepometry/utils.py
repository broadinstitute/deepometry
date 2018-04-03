# coding: utf-8

import glob
import os.path

import numpy


def load(directories, convert=True, sample=None):
    """
    Load image and label data.

    :param directories: List of directories. Subdirectories of `directories` directories are
        class labels and subdirectory contents are image data as NPY arrays.
    :param convert: Convert label strings to integers (default: `True`).
    :param sample: Undersample image data per subdirectory (default: `None`).
    :return: `(x, y, units)` where `x` is concatenated image data of shape
        `(N samples, row, col, channels)`, `y` is a list of labels of length `N samples`,
        and `units` is the number of unique labels.
    """
    paths, labels = _collect(directories, sample)
    label_to_index = {label: index for index, label in enumerate(labels)}

    x = numpy.empty((len(paths),) + _shape(paths[0]), dtype=numpy.uint8)

    if convert:
        y = numpy.empty((len(paths),), dtype=numpy.uint8)
    else:
        y = [None] * len(paths)

    for index, pathname in enumerate(paths):
        x[index] = numpy.load(pathname)

        label = os.path.split(os.path.dirname(pathname))[-1]

        if convert:
            y[index] = label_to_index[label]
        else:
            y[index] = label

    return x, y, len(labels)


def _collect(directories, sample=None):
    paths = []

    for directory in directories:
        subdirectories = glob.glob(os.path.join(directory, "*"))
        subdirectory_paths = [_filter(glob.glob(os.path.join(subdirectory, "*"))) for subdirectory in subdirectories]

        if sample:
            if isinstance(sample, bool):
                n_samples = int(numpy.median([len(subdir_paths) for subdir_paths in subdirectory_paths]))
            else:
                n_samples = sample

            subdirectory_paths = [
                list(numpy.random.permutation(subdir_paths)[:n_samples]) for subdir_paths in subdirectory_paths
            ]

        paths += subdirectory_paths

    paths = sum(paths, [])
    labels = sorted(set([os.path.split(os.path.dirname(pathname))[-1] for pathname in paths]))

    return paths, labels


def _filter(paths):
    return [path for path in paths if os.path.splitext(path)[-1].lower() == ".npy"]


def _shape(pathname):
    return numpy.load(pathname).shape
