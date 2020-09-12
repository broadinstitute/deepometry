# coding: utf-8

import os, glob
import click
import itertools

import numpy
import pandas
import seaborn
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn import preprocessing
import sklearn.metrics

import tensorflow
import keras
import deepometry.utils
import deepometry.model


@click.command(
    "evaluate",
    help="""
    Compute loss & accuracy values.

    INPUT should be a directory that contains parsed NPY arrays.

    OUTPUT directory where the evaluation outputs are saved.

    TRAINING_INPUT is the directory that was used during model training, which contains parsed NPY arrays,
    organized in subdirectories of corresponding experiments, days, samples, replicates, classes.    

    CLASS_OPTION target category to train the classifier, choose among "experiment", "day", "sample", "replicate", "class"
    For instance, put "sample" to instruct the model to categorize Sample A, Sample B, Sample C;
    choose "class" to instruct the model to classify "Class Control_cells", "Class Treated_cells" etc...
    """
)
@click.argument(
    "input",
    required=True,
    type=click.Path(exists=True)
)
@click.argument(
    "output",
    required=True,
    type=click.Path()
)
@click.option(
    "--training-input",
    required=True, 
    help="The directory that was used during model training",
    type=click.Path()
)
@click.option(
    "--batch-size",
    default=32,
    required=True,
    help="Number of samples per gradient update.",
    type=click.INT
)
@click.option(
    "--class-option",
    default=32,
    help="Target classification, = experiment / day / sample / replicate / class",
    type=click.STRING    
)
@click.option(
    "--model-location",
    default=None,
    help="Location of saved model(s)",    
    type=click.Path()
)
@click.option(
    "--name",
    default=None,
    help="A unique identifier for referencing this model.",
    type=click.STRING
)
@click.option(
    "--verbose",
    is_flag=True
)

def command(input, output, training_input, model_location, class_option, batch_size, name, verbose):  

    all_subdirs = [x[0] for x in os.walk(training_input)]
    possible_labels = sorted(list(set([os.path.basename(i) for i in all_subdirs])))
    labels_of_interest = [i for i in possible_labels if class_option.lower() in i.lower()]

    pathnames_of_interest = deepometry.utils.collect_pathnames(input, labels_of_interest, n_samples=None)

    x, y, _ = deepometry.utils._load(pathnames_of_interest, labels_of_interest)

    units = len(list(set(labels_of_interest)))

    apply_session()

    model = deepometry.model.Model(shape=x.shape[1:], units=units, name=name)

    model.compile()

    predicted = model.predict(x, model_location, batch_size=32, verbose=1 if verbose else 0)
    
    predicted = numpy.argmax(predicted, -1)
    
    expected = y
    
    confusion = sklearn.metrics.confusion_matrix(expected, predicted)

    # Normalize values in confusion matrix
    confusion = confusion.astype('float') / confusion.sum(axis=1)[:, numpy.newaxis]

    confusion = pandas.DataFrame(confusion)
    confusion = confusion.rename(index={index: label for index, label in enumerate(labels_of_interest)}, columns={index: label for index, label in enumerate(labels_of_interest)})

    # Plot confusion matrix
    fig, _ = plt.subplots()
    fig.set_size_inches(10, 10) 
    plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues )
    plt.colorbar()
    plt.xticks(numpy.arange(len(labels_of_interest)), labels_of_interest, rotation=45)
    plt.yticks(numpy.arange(len(labels_of_interest)), labels_of_interest)

    fmt = '.2f'
    thresh = confusion.max() / 2.
    for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
        plt.text(j, i, format(confusion.iloc[i, j], fmt),
                horizontalalignment="center",
                color="white" if numpy.all(confusion.iloc[i, j] > thresh) else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    matplotlib.rcParams.update({'font.size': 15})

    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(left=0.27)
    plt.subplots_adjust(bottom=0.27)

    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.1)

    print("Save plots and values")

    if not os.path.exists(output):
        os.makedirs(output)   

    fig.savefig(os.path.join(output,'confusion_matrix_plot.png'), dpi = 300)

    # save the confusion matrix values to a csv_file.
    confusion.to_csv( os.path.join(output,'confusion_matrix.csv') )

    # save the accuracy metrics to a csv_file.
    val = sklearn.metrics.accuracy_score(expected, predicted)
    file_to_use = os.path.join(output,'accuracy_values.csv')
    with open(file_to_use, 'w') as f:
        f.write(str(val))
  
    print('Done')


def apply_session(gpu=None):
    configuration = tensorflow.ConfigProto()
    configuration.gpu_options.allow_growth = True

    if gpu!=None:
        configuration.gpu_options.visible_device_list = str(gpu)

    session = tensorflow.Session(config = configuration)
    keras.backend.set_session(session)

    return True
