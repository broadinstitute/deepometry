import collections
import hashlib
import os.path
import re
import time

import bioformats.formatreader
import javabridge
import numpy
import scipy.stats
import skimage.exposure
import skimage.io
import skimage.util

SUPPORTED_FORMATS = [".cif", ".tif", ".tiff"]


def parse(paths, output_directory, size, channels=None):
    """
    Prepare a collection of images for training, evaluation, and prediction.

    .TIF/.TIFF data is expected to be single-channel data with channel information encoded in the filename: e.g.,
    ``/.*_Ch[0-9]+.TIF/``.

    :param paths: List of pathnames to parse.
    :param output_directory: Location to exported parsed image data.
    :param size: Final dimension ``(size, size)`` of parsed image data.
    :param channels: List of channels to export. Use ``None`` to export all available channels.
    """
    ext = os.path.splitext(paths[0])[-1].lower()

    if ext == ".cif":
        for path in paths:
            _parse_cif(path, output_directory, size, channels)

    if ext in (".tif", ".tiff"):
        _parse_tif(paths, output_directory, size, channels)


def _group(paths, channels):
    pattern = "(.*)Ch"

    if channels:
        pattern += "(?:{:s})\.".format("|".join([str(channel) for channel in channels]))

    groups = collections.defaultdict(list)

    for path in paths:
        md = re.match(pattern, path)

        if md:
            group = os.path.splitext(os.path.basename(md.group(1)))[0]

            groups[group].append(path)

    # E.g., {"foo": ["foo_Ch0.tif", "foo_Ch3.tif"]}
    return groups


def _parse_cif(path, output_directory, size, channels):
    reader = bioformats.formatreader.get_image_reader("tmp", path=path)

    image_count = javabridge.call(reader.metadata, "getImageCount", "()I")

    # TODO: An overflow error occurs when reading image 190663 and above. :(
    image_count = numpy.min((image_count, 190662))

    if channels is None:
        channel_count = javabridge.call(reader.metadata, "getChannelCount", "(I)I", 0)

        channels = range(channel_count)

    for image_index in range(0, image_count, 2):
        image = reader.read(series=image_index)

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        parsed_image = numpy.empty((size, size, len(channels)), dtype=numpy.uint8)

        for (channel_index, channel) in enumerate(channels):
            parsed_image[:, :, channel_index] = _resize(image[:, :, channel], size)

        output_pathname = os.path.join(
            output_directory,
            "{:s}__{:s}.npy".format(
                os.path.basename(path).replace(".cif", ""),
                hashlib.md5(str(time.time()).encode("utf8")).hexdigest())
        )

        numpy.save(output_pathname, parsed_image)

    return True


def _parse_tif(paths, output_directory, size, channels):
    groups = _group(paths, channels)

    for group, group_paths in groups.items():
        parsed_image = numpy.empty((size, size, len(group_paths)), dtype=numpy.uint8)

        for index, path in enumerate(group_paths):
            data = skimage.io.imread(path)

            parsed_image[:, :, index] = _resize(data, size)

        output_pathname = os.path.join(
            output_directory,
            "{:s}__{:s}.npy".format(
                group,
                hashlib.md5(str(time.time()).encode("utf8")).hexdigest()
            )
        )

        numpy.save(output_pathname, parsed_image)


def _resize(x, size):
    column_adjust = size - x.shape[0]
    column_adjust_start = int(numpy.floor(column_adjust / 2.0))
    column_adjust_end = int(numpy.ceil(column_adjust / 2.0))

    row_adjust = size - x.shape[1]
    row_adjust_start = int(numpy.floor(row_adjust / 2.0))
    row_adjust_end = int(numpy.ceil(row_adjust / 2.0))

    # Crop
    cropped = _crop(
        x,
        crop_width=(
            numpy.abs(numpy.minimum((column_adjust_start, column_adjust_end), (0, 0))),
            numpy.abs(numpy.minimum((row_adjust_start, row_adjust_end), (0, 0)))
        )
    )

    # Clip and rescale
    clipped = _clip(cropped)

    # Pad
    padded = _pad(
        clipped,
        pad_width=(
            numpy.maximum((column_adjust_start, column_adjust_end), (0, 0)),
            numpy.maximum((row_adjust_start, row_adjust_end), (0, 0))
        )
    )

    # Convert to uint-8
    return _convert(padded)


def _crop(x, crop_width):
    return skimage.util.crop(x, crop_width)


def _clip(x):
    vmin, vmax = scipy.stats.scoreatpercentile(x, (0.5, 99.5))

    return skimage.exposure.rescale_intensity(x, in_range=(vmin, vmax))


def _pad(x, pad_width):
    corners = numpy.asarray((
        x[:10, :10].flatten(),
        x[:10, -10:].flatten(),
        x[-10:, :10].flatten(),
        x[-10:, -10:].flatten()
    ))

    means = numpy.mean(corners, axis=1)
    stds = numpy.std(corners, axis=1)

    # Choose the corner with the lowest standard deviation.
    # This is most likely to be background, in the majority
    # of observed cases.
    std = numpy.min(stds)

    idx = numpy.where(stds == std)[0][0]
    mean = means[idx]

    return numpy.pad(x, pad_width, _pad_normal, mean=mean, std=std)


def _convert(x):
    if x.dtype == numpy.uint8:
        return x

    return skimage.img_as_ubyte(x)


def _pad_normal(vector, pad_width, iaxis, kwargs):
    if pad_width[0] > 0:
        vector[:pad_width[0]] = numpy.random.normal(
            kwargs["mean"],
            kwargs["std"],
            vector[:pad_width[0]].shape
        )

    if pad_width[1] > 0:
        vector[-pad_width[1]:] = numpy.random.normal(
            kwargs["mean"],
            kwargs["std"],
            vector[-pad_width[1]:].shape
        )

    return vector
