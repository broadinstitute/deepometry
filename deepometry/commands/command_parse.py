import glob
import os.path
import javabridge
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

    The OUTPUT directory will be created if it does not already exist. Subdirectories of OUTPUT will mirror the
    folders and sub-folders structure of INPUT. The contents of the subdirectories are NPY files containing parsed
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
    "--image-size",
    default=48,
    help="Width and height dimension of the parsed images. The minimum suggested size is 48 pixels. Image dimensions"
         " larger than the specified size will be cropped toward the image center. Image dimensions smaller than the"
         " specified size will be padded with random noise following the distribution of the image background."
)
@click.option(
    "--channels",
    default=None,
    help="A comma-separated list of zero-indexed channels to parse. Use \"-\" to specify a range of channels. E.g.,"
         " \"0,5,6,7\" or \"0,5-7\". All channels will be parsed if this flag is omitted."
)
@click.option(
    "--montage-size",
    default=0,
    help="Use this option to generate per-channel tiled (stitched) montage, which is an NxN grid of single-cell images."
         " Leave default \"0\" for no stitching."
)
@click.option(
    "--verbose",
    is_flag=True
)


def command(input, output, image_size, channels, montage_size, verbose):

    input_dir = os.path.realpath(input)

    output_dir = os.path.realpath(output)
    
    # Collect channels to parse.
    if channels:
        channels = _parse_channels(channels)

    all_subdirs = [x[0] for x in os.walk(input_dir)]

    possible_labels = sorted(list(set([os.path.basename(i) for i in all_subdirs])))

    # Book-keepers for all metadata
    experiments = [i for i in possible_labels if 'experiment' in i.lower()]
    days = [i for i in possible_labels if 'day' in i.lower()]
    samples = [i for i in possible_labels if 'sample' in i.lower()]
    replicates = [i for i in possible_labels if 'replicate' in i.lower()]
    classes = [i for i in possible_labels if 'class' in i.lower()]
    
    print('Parsing... Please wait!')

    #TODO: this could be improved
    for exp in experiments:
        for day in days:
            for sample in samples:
                for rep in replicates:
                    for cl in classes:
                        folder_path = os.path.join(input_dir,exp,day,sample,rep,cl)
                        
                        if os.path.exists(folder_path):
                            pathnames_tif = glob.glob(os.path.join(folder_path, '*.tif'))
                            pathnames_tiff = glob.glob(os.path.join(folder_path, '*.tiff'))
                            pathnames_cif = glob.glob(os.path.join(folder_path, '*.cif'))
                            
                            for paths in [pathnames_tif, pathnames_tiff, pathnames_cif]:
                                if len(paths) > 0:
                                    dest_dir = os.path.join(output_dir,exp,day,sample,rep,cl)

                                    deepometry.parse.parse(
                                        paths=paths, 
                                        output_directory=dest_dir, 
                                        meta=exp + '_' + day + '_' + sample + '_' + rep + '_' + cl,                                        
                                        size=int(image_size),
                                        channels=channels,
                                        montage_size=int(montage_size)
                                    )

    print('Done')

    javabridge.kill_vm()

    return True


def _parse_channels(channel_str):
    groups = [group.split("-") for group in channel_str.split(",")]

    channels = [
        [int(group[0])] if len(group) == 1 else list(range(int(group[0]), int(group[1]) + 1)) for group in groups
    ]

    return sum(channels, [])
