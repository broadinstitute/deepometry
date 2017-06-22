import glob
import os.path

import bioformats
import click
import javabridge
import numpy
import pkg_resources

import deepometry.parse


@click.command(
    "parse",
    help="""
    Parse a directory of .CIF files.

    Convert .CIFs to NumPy arrays, which can be used as training, validation, or test data for a classifier.
    Subdirectories of INPUT are class labels and subdirectory contents are .CIF files containing data corresponding to
    that label.

    The OUTPUT directory will be created if it does not already exist. Subdirectories of OUTPUT are class labels
    corresponding to the subdirectories of INPUT. The contents of the subdirectories are .NPY files containing parsed
    .CIF image data.
    """
)
@click.argument(
    "input",
    type=click.Path(exists="True")
)
@click.argument(
    "output",
    type=click.Path()
)
@click.option(
    "--channels",
    default=None,
    help="A comma-separated list of zero-indexed channels to parse. Use \"-\" to specify a range of channels. E.g.,"
         " \"0,5,6,7\" or \"0,5-7\". All channels will be parsed if this flag is omitted."
)
@click.option(
    "--image-size",
    default=48,
    help="Width and height dimension of the parsed images. The minimum suggested size is 48 pixels. Image dimensions"
         " larger than the specified size will be cropped toward the image center. Image dimensions smaller than the"
         " specified size will be padded with random noise following the distribution of the image background."
)
@click.option(
    "--verbose",
    is_flag=True
)
def command(input, output, channels, image_size, verbose):
    input_directory = os.path.realpath(input)

    output_directory = os.path.realpath(output)

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    label_directories = glob.glob(os.path.join(input_directory, "*"))

    parsed_channels = None if channels is None else _parse_channels(channels)

    try:
        log_config = pkg_resources.resource_filename("deepometry", "resources/logback.xml")

        javabridge.start_vm(
            args=[
                "-Dlogback.configurationFile={}".format(log_config),
                "-Dloglevel={}".format("DEBUG" if verbose else "OFF")
            ],
            class_path=bioformats.JARS,
            run_headless=True
        )

        for label_directory in label_directories:
            _, label = os.path.split(label_directory)

            output_label_directory = os.path.join(output_directory, label)

            if not os.path.exists(output_label_directory):
                os.mkdir(output_label_directory)

            _parse_directory(
                os.path.join(input_directory, label),
                output_label_directory,
                parsed_channels,
                image_size
            )
    finally:
        javabridge.kill_vm()


def _parse_channels(channel_str):
    groups = [group.split("-") for group in channel_str.split(",")]

    channels = [
        [int(group[0])] if len(group) == 1 else list(range(int(group[0]), int(group[1]) + 1)) for group in groups
    ]

    return sum(channels, [])


def _parse_directory(input, output, channels, image_size):
    pathnames = glob.glob(os.path.join(input, "*.cif"))

    for pathname in pathnames:
        filename = os.path.basename(pathname)

        name, _ = os.path.splitext(filename)

        images = deepometry.parse.parse(pathname, image_size, channels)

        numpy.save(os.path.join(output, "{}.npy".format(name)), images)
