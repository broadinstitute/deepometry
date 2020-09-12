# coding: utf-8

import os, glob
import click
import tensorflow
import keras
import deepometry.utils
import deepometry.model

@click.command(
    "fit",
    help="""
    Train a model.

    INPUT should be a directory that contains parsed NPY arrays, organized in subdirectories of corresponding
    experiments, days, samples, replicates, classes.

    OUTPUT directory where the trained model should be saved.

    CLASS_OPTION target category to train the classifier, choose among "experiment", "day", "sample", "replicate", "class"
    For instance, put "sample" to instruct the model to learn to categorize Sample A, Sample B, Sample C;
    choose "class" to train the model to distinguish "Class Control_cells", "Class Treated_cells" etc...
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
    "--epochs",
    default=128,
    help="Number of iterations over training data.",
    type=click.INT
)
@click.option(
    "--name",
    default=None,
    help="A unique identifier for referencing this model.",
    type=click.STRING
)
@click.option(
    "--n-samples",
    default=None,
    help="Sampling size for over-representing classes",
    type=click.STRING
)
@click.option(
    "--validation-split",
    default=0.2,
    help="Fraction of training data withheld for validation.",
    type=click.FLOAT
)
@click.option(
    "--verbose",
    is_flag=True
)

def command(input, output, class_option, n_samples, validation_split, batch_size, epochs, name, verbose):

    input_dir = os.path.realpath(input)

    output_dir = os.path.realpath(output)

    all_subdirs = [x[0] for x in os.walk(input_dir)]
    possible_labels = sorted(list(set([os.path.basename(i) for i in all_subdirs])))
    labels_of_interest = [i for i in possible_labels if class_option.lower() in i.lower()]

    pathnames_of_interest = deepometry.utils.collect_pathnames(input_dir, labels_of_interest, n_samples=n_samples)

    x, y, _ = deepometry.utils._load(pathnames_of_interest, labels_of_interest)

    units = len(list(set(labels_of_interest)))

    apply_session()

    model = deepometry.model.Model(
        name=name,
        shape=x.shape[1:],
        units=units
    )

    model.compile()

    print('Model training... Please wait!')

    model.fit(
        x,
        y,
        balance_train=False,
        class_weight=None,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1 if verbose else 0
    )

    print("Save model to ", output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.model.save( os.path.join(output_dir, 'model_'+ class_option + '_' + str(units) + '_categories.h5') ) # The 'units' in model name is very important for feature extraction module!

    print('Done')


def apply_session(gpu=None):
    configuration = tensorflow.ConfigProto()
    configuration.gpu_options.allow_growth = True

    if gpu!=None:
        configuration.gpu_options.visible_device_list = str(gpu)

    session = tensorflow.Session(config = configuration)
    keras.backend.set_session(session)

    return True
