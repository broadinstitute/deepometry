import glob
import os.path

import bioformats.formatreader
import javabridge
import numpy
import numpy.random
import numpy.testing
import scipy.stats
import skimage.exposure

import deepometry.parse


def test_parse_larger_image(mocker, tmpdir):
    mocker.patch("bioformats.formatreader.get_image_reader")

    reader = mocker.patch("bioformats.formatreader.ImageReader")

    bioformats.formatreader.get_image_reader.return_value = reader

    mocker.patch("javabridge.call")

    javabridge.call.return_value = 2

    reader.read.return_value = numpy.random.rand(51, 42, 5)

    output_directory = tmpdir.mkdir("output")

    deepometry.parse.parse(["cells.cif"], str(output_directory), 32, channels=[0])

    bioformats.formatreader.get_image_reader.assert_called_with("tmp", path="cells.cif")

    parsed_pathnames = glob.glob(os.path.join(str(output_directory), "*.npy"))

    assert len(parsed_pathnames) == 1

    image = numpy.load(parsed_pathnames[0])

    assert image.shape == (32, 32, 1)

    assert image.dtype == numpy.uint8


def test_parse_smaller_image(mocker, tmpdir):
    mocker.patch("bioformats.formatreader.get_image_reader")

    reader = mocker.patch("bioformats.formatreader.ImageReader")

    bioformats.formatreader.get_image_reader.return_value = reader

    mocker.patch("javabridge.call")

    javabridge.call.return_value = 2

    reader.read.return_value = numpy.random.rand(28, 31, 5)

    output_directory = tmpdir.mkdir("output")

    deepometry.parse.parse(["cells.cif"], str(output_directory), 32, channels=[0])

    bioformats.formatreader.get_image_reader.assert_called_with("tmp", path="cells.cif")

    parsed_pathnames = glob.glob(os.path.join(str(output_directory), "*.npy"))

    assert len(parsed_pathnames) == 1

    image = numpy.load(parsed_pathnames[0])

    assert image.shape == (32, 32, 1)

    assert image.dtype == numpy.uint8


def test_parse_larger_smaller_image(mocker, tmpdir):
    mocker.patch("bioformats.formatreader.get_image_reader")

    reader = mocker.patch("bioformats.formatreader.ImageReader")

    bioformats.formatreader.get_image_reader.return_value = reader

    mocker.patch("javabridge.call")

    javabridge.call.return_value = 2

    reader.read.return_value = numpy.random.rand(28, 43, 5)

    output_directory = tmpdir.mkdir("output")

    deepometry.parse.parse(["cells.cif"], str(output_directory), 32, channels=[0])

    bioformats.formatreader.get_image_reader.assert_called_with("tmp", path="cells.cif")

    parsed_pathnames = glob.glob(os.path.join(str(output_directory), "*.npy"))

    assert len(parsed_pathnames) == 1

    image = numpy.load(parsed_pathnames[0])

    assert image.shape == (32, 32, 1)

    assert image.dtype == numpy.uint8


def test_parse_channels(mocker, tmpdir):
    mocker.patch("bioformats.formatreader.get_image_reader")

    reader = mocker.patch("bioformats.formatreader.ImageReader")

    bioformats.formatreader.get_image_reader.return_value = reader

    mocker.patch("javabridge.call")

    javabridge.call.return_value = 2

    cif_image = numpy.random.rand(48, 48, 5)

    reader.read.return_value = cif_image

    output_directory = tmpdir.mkdir("output")

    deepometry.parse.parse(["cells.cif"], str(output_directory), 48, channels=[0, 3, 4])

    bioformats.formatreader.get_image_reader.assert_called_with("tmp", path="cells.cif")

    parsed_pathnames = glob.glob(os.path.join(str(output_directory), "*.npy"))

    assert len(parsed_pathnames) == 1

    image = numpy.load(parsed_pathnames[0])

    assert image.shape == (48, 48, 3)

    assert image.dtype == numpy.uint8

    expected = numpy.zeros((48, 48, 3), dtype=numpy.uint8)

    for index, channel in enumerate([0, 3, 4]):
        vmin, vmax = scipy.stats.scoreatpercentile(cif_image[:, :, channel], (0.5, 99.5))

        expected[:, :, index] = skimage.exposure.rescale_intensity(
            cif_image[:, :, channel],
            in_range=(vmin, vmax),
            out_range=numpy.uint8
        ).astype(numpy.uint8)

    numpy.testing.assert_array_equal(image, expected)


def test_parse_tifs(mocker, tmpdir):
    paths = [
        "foo_Ch0.tif", "foo_Ch1.tif", "foo_Ch2.tif",
        "bar_Ch0.tif", "bar_Ch1.tif", "bar_Ch2.tif",
        "baz_Ch0.tif", "baz_Ch1.tif", "baz_Ch2.tif"
    ]

    reader = mocker.patch("skimage.io.imread")
    reader.return_value = numpy.random.rand(48, 48)

    output_directory = tmpdir.mkdir("output")

    deepometry.parse.parse(paths, str(output_directory), 48)

    parsed_pathnames = glob.glob(os.path.join(str(output_directory), "*.npy"))

    assert len(parsed_pathnames) == 3

    image = numpy.load(parsed_pathnames[0])

    assert image.shape == (48, 48, 3)

    assert image.dtype == numpy.uint8


def test_parse_tifs_channels(mocker, tmpdir):
    paths = [
        "foo_Ch0.tif", "foo_Ch1.tif", "foo_Ch2.tif",
        "bar_Ch0.tif", "bar_Ch1.tif", "bar_Ch2.tif",
        "baz_Ch0.tif", "baz_Ch1.tif", "baz_Ch2.tif"
    ]

    reader = mocker.patch("skimage.io.imread")
    reader.return_value = numpy.random.rand(48, 48)

    output_directory = tmpdir.mkdir("output")

    deepometry.parse.parse(paths, str(output_directory), 48, channels=[0, 2])

    parsed_pathnames = glob.glob(os.path.join(str(output_directory), "*.npy"))

    assert len(parsed_pathnames) == 3

    image = numpy.load(parsed_pathnames[0])

    assert image.shape == (48, 48, 2)

    assert image.dtype == numpy.uint8
