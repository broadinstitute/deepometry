import bioformats.formatreader
import javabridge
import numpy
import numpy.random
import numpy.testing

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
