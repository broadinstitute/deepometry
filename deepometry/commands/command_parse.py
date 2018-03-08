import glob
import os.path

import bioformats
import click
import pkg_resources

import deepometry.parse


@click.command(
    "parse",
    help="""Parse a directory of .CIF or .TIF files.

    Convert CIF or TIF files to NPY arrays, which can be used as training, validation, or test data for a classifier.
    Subdirectories of INPUT are class labels and subdirectory contents are .CIF or .TIF files containing data 
    corresponding to that label.

    The OUTPUT directory will be created if it does not already exist. Subdirectories of OUTPUT are class labels
    corresponding to the subdirectories of INPUT. The contents of the subdirectories are NPY files containing parsed
    .CIF or .TIF image data.
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

    # Collect subdirectories.
    subdirectories = [
        directory for directory in glob.glob(os.path.join(input_directory, "*")) if os.path.isdir(directory)
    ]

    # Collect channels to parse.
    if channels:
        channels = _parse_channels(channels)

    jvm_started = False

    for subdirectory in subdirectories:
        # Collect images in subdirectory, filtering out unsupported image formats.
        paths = _collect(subdirectory)
        ext = os.path.splitext(paths[0])[-1].lower()

        if ext == ".cif" and not jvm_started:
            import javabridge

            log_config = pkg_resources.resource_filename("deepometry", "resources/logback.xml")

            javabridge.start_vm(
                args=[
                    "-Dlogback.configurationFile={}".format(log_config),
                    "-Dloglevel={}".format("DEBUG" if verbose else "OFF")
                ],
                class_path=bioformats.JARS,
                max_heap_size="8G",
                run_headless=True
            )

            jvm_started = True

        label = os.path.split(subdirectory)[-1]
        label_directory = os.path.join(output_directory, label)

        try:
            deepometry.parse.parse(
                paths,
                output_directory=label_directory,
                size=image_size,
                channels=channels
            )
        except Exception as exception:
            if jvm_started:
                javabridge.kill_vm()

            raise exception

    if jvm_started:
        javabridge.kill_vm()


def _collect(directory):
    def is_image_file(path):
        ext = os.path.splitext(path)[-1].lower()
        return ext in deepometry.parse.SUPPORTED_FORMATS

    return [path for path in glob.glob(os.path.join(directory, "*")) if is_image_file(path)]


def _parse_channels(channel_str):
    groups = [group.split("-") for group in channel_str.split(",")]

    channels = [
        [int(group[0])] if len(group) == 1 else list(range(int(group[0]), int(group[1]) + 1)) for group in groups
    ]

    return sum(channels, [])
