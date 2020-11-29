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

#---for stitching---#
from skimage.util import montage
import os
import pandas
import bioformats
import math
import glob

SUPPORTED_FORMATS = [".cif", ".tif", ".tiff"]
                       

def parse(paths, output_directory, meta, size, channels=None, montage_size=0):
    """
    Prepare a collection of images for training, evaluation, and prediction.

    .TIF/.TIFF data is expected to be single-channel data with channel information encoded in the filename: e.g.,
    ``/.*_Ch[0-9]+.TIF/``.

    paths: List of pathnames to parse.
    output_directory: Location to exported parsed image data.
    size: Final dimension ``(size, size)`` of parsed image data.
    channels: List of channels to export. Use ``None`` to export all available channels.
    """

    ext = os.path.splitext(paths[0])[-1].lower()

    if ext == ".cif":
        # This line is critical for reading .CIF files
        # javabridge.start_vm(class_path=bioformats.JARS, max_heap_size="8G")     
        
        for path in paths:
            print(path)
            _parse_cif(path, output_directory, size, meta, channels, montage_size)

    if ext in (".tif", ".tiff"):
        _parse_tif(paths, output_directory, size, meta, channels, montage_size)


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


def _parse_cif(path, output_directory, size, meta, channels, montage_size):

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    reader = bioformats.formatreader.get_image_reader("tmp", path=path)

    image_count = javabridge.call(reader.metadata, "getImageCount", "()I")

    # TODO: An overflow error occurs when reading image 190663 and above. :(
    image_count = numpy.min((image_count, 190662))

    if channels is None:
        channel_count = javabridge.call(reader.metadata, "getChannelCount", "(I)I", 0)

        channels = range(channel_count)
        
    if montage_size > 0:
        print('Stitching')

        n_chunks = __compute_chunks(image_count/2, montage_size)
        chunk_size = montage_size**2

        for channel in channels:
            for chunk in range(n_chunks):
                try:
                    images = [
                        reader.read(c=channel, series=image) for image in range(image_count)[::2][chunk*chunk_size:(chunk+1)*chunk_size]
                    ]
                except javabridge.jutil.JavaException:
                    break

                images = [_resize(image, size) for image in images]

                stitched_image = montage(numpy.asarray(images), 0)

                if chunk == (n_chunks-1):
                    stitched_image = __pad_to_same_chunk_size(stitched_image, size, montage_size)

                skimage.io.imsave(os.path.join(output_directory, "{}_ch{:d}_{:d}.tif".format(meta, channel + 1, chunk + 1)), stitched_image) 
                
    else:

        for image_index in range(0, image_count, 2):
            image = reader.read(series=image_index)

            parsed_image = numpy.empty((size, size, len(channels)), dtype=numpy.uint16)

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


def _parse_tif(paths, output_directory, size, meta, channels, montage_size):
    groups = _group(sorted(paths), sorted(channels))

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if montage_size > 0:
        print('Stitching', meta)

        n_chunks = __compute_chunks(len(groups), montage_size)
        chunk_size = montage_size**2        
        
        for i,channel in enumerate(channels):
            for chunk in range(n_chunks):

                images = [
                    _resize(skimage.io.imread(group_paths[i]), size) for group_paths in list(groups.values())[chunk*chunk_size:(chunk+1)*chunk_size ]
                ]

                stitched_image = montage(numpy.asarray(images), 0)

                if chunk == (n_chunks-1):
                    stitched_image = __pad_to_same_chunk_size(stitched_image, size, montage_size)

                skimage.io.imsave(os.path.join(output_directory, "{}_ch{:d}_{:d}.tif".format(meta, channel + 1, chunk + 1)), stitched_image) 
                        
    else:
        
        for group, group_paths in groups.items():
            parsed_image = numpy.empty((size, size, len(group_paths)), dtype=numpy.uint16)

            for index, path in enumerate(group_paths):
                data = skimage.io.imread(path)

                parsed_image[:, :, index] = _resize(data, size)                          
            
            output_pathname = os.path.join(
                output_directory,
                "{:s}__{:s}.npy".format(
                    meta,
                    hashlib.md5(str(time.time()).encode("utf8")).hexdigest())
            )

            numpy.save(output_pathname, parsed_image)


def _resize(x, size):
    
    if len(x.shape) > 2:
        x = x[:,:,0]    
    
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
    #clipped = _clip(cropped)

    # Pad
    padded = _pad(
        #clipped,
        cropped,
        pad_width=(
            numpy.maximum((column_adjust_start, column_adjust_end), (0, 0)),
            numpy.maximum((row_adjust_start, row_adjust_end), (0, 0))
        )
    )

    # Convert to uint-8
    return _convert(padded)
    #return padded


def _crop(x, crop_width):
    return skimage.util.crop(x, crop_width)


def _clip(x):
    vmin, vmax = scipy.stats.scoreatpercentile(x, (0.5, 99.5))

    return numpy.clip(x, vmin, vmax)


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

    return skimage.exposure.rescale_intensity(
        x,
        out_range=numpy.uint8
    ).astype(numpy.uint8)


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


def __pad_to_same_chunk_size(small_montage, image_size, montage_size):
    pad_x = float(montage_size*image_size - small_montage.shape[0])

    pad_y = float(montage_size*image_size - small_montage.shape[1])

    npad = ((0,int(pad_y)), (0,int(pad_x)))

    return numpy.pad(small_montage, pad_width=npad, mode='constant', constant_values=0)


def __compute_chunks(n_images, montage_size):

    def remainder(images, groups):
        return (images - groups * (montage_size ** 2))

    n_groups = 1

    while remainder(n_images, n_groups) > 0:
        n_groups += 1

    return n_groups