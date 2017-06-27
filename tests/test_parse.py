import bioformats.formatreader
import javabridge
import numpy
import numpy.random
import numpy.testing
import scipy.stats
import skimage.exposure

import deepometry.parse


def test_parse_larger_image(mocker):
    mocker.patch("bioformats.formatreader.get_image_reader")

    reader = mocker.patch("bioformats.formatreader.ImageReader")

    bioformats.formatreader.get_image_reader.return_value = reader

    mocker.patch("javabridge.call")

    javabridge.call.return_value = 2

    reader.read.return_value = numpy.random.rand(51, 42, 5)

    images = deepometry.parse.parse("cells.cif", 32, [0])

    bioformats.formatreader.get_image_reader.assert_called_with("tmp", path="cells.cif")

    assert images.shape == (1, 32, 32, 1)

    assert images.dtype == numpy.uint8


def test_parse_smaller_image(mocker):
    mocker.patch("bioformats.formatreader.get_image_reader")

    reader = mocker.patch("bioformats.formatreader.ImageReader")

    bioformats.formatreader.get_image_reader.return_value = reader

    mocker.patch("javabridge.call")

    javabridge.call.return_value = 2

    reader.read.return_value = numpy.random.rand(28, 31, 5)

    images = deepometry.parse.parse("cells.cif", 32, [0])

    bioformats.formatreader.get_image_reader.assert_called_with("tmp", path="cells.cif")

    assert images.shape == (1, 32, 32, 1)

    assert images.dtype == numpy.uint8


def test_parse_larger_smaller_image(mocker):
    mocker.patch("bioformats.formatreader.get_image_reader")

    reader = mocker.patch("bioformats.formatreader.ImageReader")

    bioformats.formatreader.get_image_reader.return_value = reader

    mocker.patch("javabridge.call")

    javabridge.call.return_value = 2

    reader.read.return_value = numpy.random.rand(28, 43, 5)

    images = deepometry.parse.parse("cells.cif", 32, [0])

    bioformats.formatreader.get_image_reader.assert_called_with("tmp", path="cells.cif")

    assert images.shape == (1, 32, 32, 1)

    assert images.dtype == numpy.uint8


def test_parse_channels(mocker):
    mocker.patch("bioformats.formatreader.get_image_reader")

    reader = mocker.patch("bioformats.formatreader.ImageReader")

    bioformats.formatreader.get_image_reader.return_value = reader

    mocker.patch("javabridge.call")

    javabridge.call.return_value = 2

    image = numpy.ones((48, 48, 5), dtype=numpy.uint8)

    reader.read.return_value = image

    images = deepometry.parse.parse("cells.cif", 48, [0, 3, 4])

    bioformats.formatreader.get_image_reader.assert_called_with("tmp", path="cells.cif")

    assert images.shape == (1, 48, 48, 3)

    assert images.dtype == numpy.uint8

    expected = numpy.zeros((1, 48, 48, 3), dtype=numpy.uint8)

    for index, channel in enumerate([0, 3, 4]):
        vmin, vmax = scipy.stats.scoreatpercentile(image[:, :, channel], (0.5, 99.5))

        expected[0, :, :, index] = skimage.exposure.rescale_intensity(
            image[:, :, channel],
            in_range=(vmin, vmax),
            out_range=numpy.uint8
        ).astype(numpy.uint8)

    numpy.testing.assert_array_equal(images, expected)
