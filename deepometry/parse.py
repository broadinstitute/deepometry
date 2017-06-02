import glob
import numpy
import os.path

import bioformats.formatreader
import javabridge
import joblib
import skimage.exposure
import skimage.util


def parse_directory(directory, size, channels):
    tensor = [parse(pathname, size, channels) for pathname in glob.glob(os.path.join(directory, "*"))]

    return numpy.concatenate(tensor)


def parse(pathname, size, channels):
    """
    Convert an image file to a NumPy array.

    For microscopic image formats with OME support (e.g., .CIF), the javabridge JVM is required:

        import bioformats
        import javabridge

        javabridge.start_vm(class_path=bioformats.JARS)

    :param pathname: Image pathname.
    :param size: Final image dimensions (size, size, channels).
    :param channels: Image channels to extract.
    :return: NumPy array of image data. If the file contains a single image, returns (size, size, channels). If the
    file contains multiple image (e.g., is a .CIF file), returns (N_images, size, size, channels).
    """
    ext = os.path.splitext(pathname)[-1].lower()

    if ext == ".cif":
        return _parse_cif(pathname, size, channels)

    # if ext in [".tif", ".tiff"]:
    #     return _parse_tiff(pathname, size, channels)

    raise NotImplementedError("Unsupported file format: {}".format(ext))


def _parse_cif(pathname, size, channels):
    reader = bioformats.formatreader.get_image_reader("tmp", path=pathname)

    image_count = javabridge.call(reader.metadata, "getImageCount", "()I")

    images = numpy.zeros((image_count // 2, size, size, len(channels)))

    for image_index in range(0, image_count, 2):
        image = reader.read(series=image_index)

        for (channel_index, channel) in enumerate(channels):
            images[image_index // 2, :, :, channel_index] = _resize(image[:, :, channel_index], size)

    return images


def _parse_tiff(pathname, size, channels):
    pass


def _resize(image, size):
    column_adjust = size - image.shape[0]
    column_adjust_start = int(numpy.floor(column_adjust / 2.0))
    column_adjust_end = int(numpy.ceil(column_adjust / 2.0))

    row_adjust = size - image.shape[1]
    row_adjust_start = int(numpy.floor(row_adjust / 2.0))
    row_adjust_end = int(numpy.ceil(row_adjust / 2.0))

    resized = skimage.util.crop(
        image,
        (
            numpy.abs(numpy.minimum((column_adjust_start, column_adjust_end), (0, 0))),
            numpy.abs(numpy.minimum((row_adjust_start, row_adjust_end), (0, 0)))
        )
    )

    return skimage.exposure.rescale_intensity(
        _pad(
            resized,
            (
                numpy.maximum((column_adjust_start, column_adjust_end), (0, 0)),
                numpy.maximum((row_adjust_start, row_adjust_end), (0, 0))
            )
        )
    )


def _pad(image, pad_width):
    sample = image[:10, :10]

    return numpy.pad(image, pad_width, _pad_normal, mean=numpy.mean(sample), std=numpy.std(sample))


def _pad_normal(vector, pad_width, iaxis, kwargs):
    if pad_width[0] > 0:
        vector[:pad_width[0]] = numpy.random.normal(kwargs["mean"], kwargs["std"], vector[:pad_width[0]].shape)

    if pad_width[1] > 0:
        vector[-pad_width[1]:] = numpy.random.normal(kwargs["mean"], kwargs["std"], vector[-pad_width[1]:].shape)

    return vector
