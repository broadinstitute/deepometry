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
        filelist.append([i for i in all_npy for j in split_all(i)[-6:-1] if label in j]) 
    
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
                label = [l for l in labels_of_interest for spl in split_all(pathname)[-6:-1] if l in spl][0]
                y[index] = label_to_index[label]

            metadata_bkpr.append(split_all(pathname)[-6:-1])
        
    return x, y, metadata_bkpr


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