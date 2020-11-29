# Native
import os, glob

# Libraries
import numpy


def collect_pathnames(input_dir, labels_of_interest, n_samples=None):
    """
    Gather path locations of parsed images (3D tensors/numpy arrays), grouped into categories of choice
    """
    
    all_npy = [j for i in [x[0] for x in os.walk(input_dir)] for j in glob.glob(os.path.join(i,'*')) if '.npy' in j]
    
    filelist = []
    for label in labels_of_interest:
        filelist.append([i for i in all_npy for j in split_all(i)[-6:-1] if label in j.lower()]) 
    
    # Collect similar number of samples per class
    pathnames = []
    if n_samples==None:
        pathnames += [list(numpy.random.permutation(pathnames)) for pathnames in filelist]
    else:
        pathnames += [list(numpy.random.permutation(pathnames))[:n_samples] for pathnames in filelist]
    pathnames = sum(pathnames, [])

    return pathnames


def _load(pathnames, labels_of_interest=None):
    """
    Prepare x,y as signal,target pairs for machine learning
    """
  
    x = numpy.empty((len(pathnames),) + _shape(pathnames[0]), dtype=numpy.uint8)
    
    y = numpy.empty((len(pathnames),), dtype=numpy.uint8)
    
    if labels_of_interest != None:
        label_to_index = {label: index for index, label in enumerate(labels_of_interest)}    

    # Metadata book-keeper:
    metadata_bkpr = []

    for index, pathname in enumerate(pathnames):
        if os.path.isfile(pathname) == True:

            x[index] = numpy.load(pathname)
            
            if labels_of_interest != None:
                label = [l for l in labels_of_interest for spl in split_all(pathname)[-6:-1] if l in spl.lower()][0]
                y[index] = label_to_index[label]

            metadata_bkpr.append(split_all(pathname)[-6:-1])
        
    return x, y, metadata_bkpr


def quickload(directories, convert=True, sample=None):
    """
    Load image and label data.

    :param directories: List of directories. Subdirectories of ``directories`` are
        class labels and subdirectory contents are image data as NPY arrays.
    :param convert: Convert label strings to integers (default: ``True``).
    :param sample: Undersample image data per subdirectory (default: ``None``).
    :return: ``(x, y, units)`` where ``x`` is concatenated image data of shape
        ``(N samples, row, col, channels)``, ``y`` is a list of labels of length ``N samples``,
        and ``units`` is the number of unique labels.
    """
    paths, labels = _collect(directories, sample)
    label_to_index = {label: index for index, label in enumerate(labels)}

    x = numpy.empty((len(paths),) + _shape(paths[0]), dtype=numpy.uint8)

    if convert:
        y = numpy.empty((len(paths),), dtype=numpy.uint8)
    else:
        y = [None] * len(paths)

    for index, pathname in enumerate(paths):
        x[index] = numpy.load(pathname)

        label = os.path.split(os.path.dirname(pathname))[-1]

        if convert:
            y[index] = label_to_index[label]
        else:
            y[index] = label

    return x, y, len(labels)


def _shape(pathname):

    return numpy.load(pathname).shape


def split_all(path):
    """
    Break a path into unit components
    """

    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def _collect(directories, sample=None):
    paths = []

    for directory in directories:
        subdirectories = glob.glob(os.path.join(directory, "*"))
        subdirectory_paths = [_filter(glob.glob(os.path.join(subdirectory, "*"))) for subdirectory in subdirectories]

        if sample:
            if isinstance(sample, bool):
                n_samples = int(numpy.median([len(subdir_paths) for subdir_paths in subdirectory_paths]))
            else:
                n_samples = sample

            subdirectory_paths = [
                list(numpy.random.permutation(subdir_paths)[:n_samples]) for subdir_paths in subdirectory_paths
            ]

        paths += subdirectory_paths

    paths = sum(paths, [])
    labels = sorted(set([os.path.split(os.path.dirname(pathname))[-1] for pathname in paths]))

    return paths, labels


def _filter(paths):
    return [path for path in paths if os.path.splitext(path)[-1].lower() == ".npy"]
