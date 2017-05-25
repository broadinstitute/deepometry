import numpy
import os.path

import bioformats.formatreader
import javabridge
import skimage.util


def parse(pathname, size, channels):
    ext = os.path.splitext(pathname)[-1].lower()

    if ext == ".cif":
        return _parse_cif(pathname, size, channels)

    if ext in [".tif", ".tiff"]:
        return _parse_tiff(pathname, size, channels)

    raise NotImplementedError("Unsupported file format: {}".format(ext))


def _parse_cif(pathname, size, channels):
    reader = bioformats.formatreader.get_image_reader("tmp", path=pathname)

    image_count = javabridge.call(reader.metadata, "getImageCount", "()I")

    tensor = ()

    for channel in channels:
        images = numpy.asarray([
            _resize(
                reader.read(c=channel, series=index),
                size
            ) for index in range(image_count)[::2]
        ])

        tensor += (images,)

    return numpy.stack(tensor, axis=3)


def _parse_tiff(pathname, size, channels):
    pass


def _resize(image, size):
    column_adjust = size - image.shape[0]
    column_adjust_start = int(numpy.floor(column_adjust / 2.0))
    column_adjust_end = int(numpy.ceil(column_adjust / 2.0))

    row_adjust = size - image.shape[1]
    row_adjust_start = int(numpy.floor(row_adjust / 2.0))
    row_adjust_end = int(numpy.ceil(row_adjust / 2.0))

    if column_adjust <= 0:
        resized = skimage.util.crop(image, ((numpy.abs(column_adjust_start), numpy.abs(column_adjust_end)), (0, 0)))
    else:
        resized = _pad(image, ((column_adjust_start, column_adjust_end), (0, 0)))

    if row_adjust <= 0:
        resized = skimage.util.crop(resized, ((0, 0), (numpy.abs(row_adjust_start), numpy.abs(row_adjust_end))))
    else:
        resized = _pad(resized, ((0, 0), (row_adjust_start, row_adjust_end)))

    return resized


def _pad(image, pad_width):
    # Pad with random noise sampled from the image background.
    sample = image[:10, :10]

    return numpy.pad(image, pad_width, _pad_normal, mean=numpy.mean(sample), std=numpy.std(sample))


def _pad_normal(vector, pad_width, iaxis, kwargs):
    if pad_width[0] > 0:
        vector[:pad_width[0]] = numpy.random.normal(kwargs["mean"], kwargs["std"], vector[:pad_width[0]].shape)

    if pad_width[1] > 0:
        vector[-pad_width[1]:] = numpy.random.normal(kwargs["mean"], kwargs["std"], vector[-pad_width[1]:].shape)

    return vector
