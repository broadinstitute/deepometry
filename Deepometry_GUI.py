# Native
from random import randint
import sys, os, glob, re
import time
from time import strftime
import pickle
import itertools
from collections import Counter
import pkg_resources
from itertools import groupby

# Libraries
import javabridge
import bioformats
import numpy
import pandas
import seaborn
import matplotlib
import matplotlib.pyplot as plt
import tensorflow
import keras

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn import preprocessing
import sklearn.metrics

# Flask
from flask import Flask, render_template, flash, Response, request, redirect, url_for 
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from jinja2 import Environment
from jinja2.loaders import FileSystemLoader

# Deepometry
import deepometry.parse
import deepometry.utils
import deepometry.model
import deepometry.visualize


# Custom functions

def apply_session(gpu=None):
    configuration = tensorflow.ConfigProto()
    configuration.gpu_options.allow_growth = True

    if gpu!=None:
        configuration.gpu_options.visible_device_list = str(gpu)

    session = tensorflow.Session(config = configuration)
    keras.backend.set_session(session)

    return True


def parsing(input_parse, output_parse, frame, channels, montage_size):
    '''
    Convert images from .CIF or .TIF formats to numpy arrays, organized in a structured folder tree.
    '''

    print('Parsing... Please wait!')
    if channels != None:
        channels = re.findall(r'\d+', channels)
        channels = [int(i) for i in channels]

    pathnames_tif = glob.glob(os.path.join(input_parse, '**', '*.tif'), recursive = True)
    pathnames_tiff = glob.glob(os.path.join(input_parse, '**', '*.tiff'), recursive = True)
    pathnames_cif = glob.glob(os.path.join(input_parse, '**', '*.cif'), recursive = True)
    if len(pathnames_cif) > 0:
        javabridge.start_vm(class_path=bioformats.JARS, max_heap_size="8G")       

    for paths in [pathnames_tif, pathnames_tiff, pathnames_cif]:
        if len(paths) > 0:
            keyf = lambda path: os.path.dirname(path)
            grouped_paths = [list(items) for gr, items in groupby(paths, key=keyf)]

            for group in grouped_paths:

                meta_as_path = os.path.relpath(os.path.dirname(group[0]),input_parse)

                dest_dir = os.path.join(output_parse,meta_as_path)

                deepometry.parse.parse(
                    paths=group, 
                    output_directory=dest_dir,  
                    meta = '_'.join(os.path.normpath(meta_as_path).split(os.path.sep)),                                 
                    size=int(frame),
                    channels=channels,
                    montage_size=int(montage_size)
                    )

    print('Done.')


def training(input_train, output_train, class_option, n_samples=None, validation_split=0.2, batch_size=32, epochs=512):
    '''
    Train a deep neural network
    '''
    
    print('Model training... Please wait!')

    # Turn user's input, which is a string "['foo','bar',...]", into a list without quotations "": ['foo','bar',...]
    labels_of_interest = [i.lower() for i in re.split("[\'\"]", class_option)[1:-1] if ', ' not in i]

    # We'll search within the parsed database, any categories that match the prefixes:
    pathnames_of_interest = deepometry.utils.collect_pathnames(input_train, labels_of_interest, n_samples=n_samples)

    x, y, _ = deepometry.utils._load(pathnames_of_interest, labels_of_interest)
    print(x.shape, y.shape)

    units = len(list(set(labels_of_interest)))

    apply_session()

    model = deepometry.model.Model(shape=x.shape[1:], units=units)

    model.compile()

    model.fit(
        x,
        y,
        balance_train=False,
        class_weight=None,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )

    print("Save model to disk...")
    if not os.path.exists(output_train):
        os.makedirs(output_train)

    prefix_class_option = re.split('\s|(?<!\d)[,._-]|[,._-](?!\d)', labels_of_interest[0])[0]
    model.model.save( os.path.join(output_train, 'model_' + str(units) + '-' + prefix_class_option + '_categories.h5') ) # The 'units' in model name is very important for feature extraction module!

    try:
        csv = pandas.read_csv("./deepometry/data/training.csv")
        _, (ax0, ax1) = plt.subplots(ncols=2, figsize=(16, 4))

        ax0.plot(csv["acc"], c="r")
        ax0.plot(csv["val_acc"], c="b")
        ax0.set_ylabel('Categorical accuracy')
        ax0.set_xlabel('Epoch')
        ax0.legend(["Train", "Val"], loc='upper left')

        ax1.plot(csv["loss"], c="r")
        ax1.plot(csv["val_loss"], c="b");        
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend(["Train", "Val"], loc='upper right')

        plt.savefig(os.path.join(output_train, 'train_val_loss.png'), dpi = 300, bbox_inches='tight')    
    except:
        pass

    print('Done')


def evaluating(input_predict, output_predict, modellocation, class_option, unannotated_box_ticked, n_samples=None):  
    '''
    Evaluate a trained deep neural network on annotated ground truth
    '''
    print('Evaluating... Please wait.')

    if not os.path.exists(output_predict):
        os.makedirs(output_predict) 

    # Turn user's input, which is a string "['foo','bar',...]", into a list without quotations "": ['foo','bar',...]
    labels_of_interest = [i.lower() for i in re.split("[\'\"]", class_option)[1:-1] if ', ' not in i]

    pathnames_of_interest = deepometry.utils.collect_pathnames(input_predict, labels_of_interest, n_samples=n_samples)

    if unannotated_box_ticked == 'on':
        x, _, _ = deepometry.utils._load(pathnames_of_interest, labels_of_interest)
    else:
        x, y, _ = deepometry.utils._load(pathnames_of_interest, labels_of_interest)

    units = len(list(set(labels_of_interest)))

    apply_session()

    model = deepometry.model.Model(shape=x.shape[1:], units=units)

    model.compile()

    predicted = model.predict(x, modellocation, batch_size=32)
    
    predicted = numpy.argmax(predicted, -1)
    
    if unannotated_box_ticked != 'on':
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
        plt.subplots_adjust(left=0.37)
        plt.subplots_adjust(bottom=0.27)

        # Pad margins so that markers don't get clipped by the axes
        plt.margins(0.1)

        print("Save plots and values")

        fig.savefig(os.path.join(output_predict,'confusion_matrix_plot.png'), dpi = 300)

        # save the confusion matrix values to a csv_file.
        confusion.to_csv( os.path.join(output_predict,'confusion_matrix.csv') )

        # save the evaluation metrics to a csv_file.
        report = pandas.DataFrame(sklearn.metrics.classification_report(expected, predicted, output_dict=True)).transpose()
        report.index = labels_of_interest + ['accuracy', 'macro avg', 'weighted avg']
        report.to_csv(os.path.join(output_predict,'classification_report.csv'))

    else:
        predicted_classes = pandas.DataFrame()
        predicted_classes['numeric_class'] = predicted
        predicted_classes['label'] = [labels_of_interest[i] for i in predicted]
        predicted_classes.to_csv(os.path.join(output_predict,'predicted.csv'), index=True, index_label='ID')

        # Simple count plot:
        plt.figure(figsize = (9, 6))
        ax = seaborn.countplot(x="label", data=predicted_classes)
        plt.xticks(rotation=60)
        # Pad margins so that markers don't get clipped by the axes
        plt.margins(0.2)
        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(bottom=0.15)
        
        print("Save predicted values")
        plt.savefig(os.path.join(output_predict, 'count_plot.png'), dpi = 300, bbox_inches='tight')        
    
    print('Done')


def extracting(input_extract, output_extract, modellocation, feature_extraction_layer):
    '''
    Use a trained deep neural network to extract deep learning embeddings
    '''

    print('Extracting features... Please wait.')
    # Collect data
    pathnames_of_interest = [j for i in [x[0] for x in os.walk(input_extract)] for j in glob.glob(os.path.join(i,'*')) if '.npy' in j]

    x, _, metadata_bkpr = deepometry.utils._load(pathnames_of_interest)

    if os.path.isdir(modellocation):
        list_of_files = glob.glob(os.path.join(modellocation, '*_categories.h5')) # only file with correct naming, i.e.***_categories.h5, is accepted
        modellocation = max(list_of_files, key=os.path.getctime)

    units = int( re.search('model_([0-9]*)-.*_categories.h5', os.path.basename(modellocation)).group(1) )
    
    apply_session()

    model = deepometry.model.Model(shape=x.shape[1:], units=units)

    model.compile()

    # Extract the features
    features = model.extract(x, feature_extraction_layer, modellocation, batch_size=32, verbose=1)

    if len(features.shape) > 2:
        a = numpy.mean(features, axis = 1)
        del(features)
        features = numpy.mean(a, axis = 1)  

    if not os.path.exists(output_extract):
        os.makedirs(output_extract)   

    # Export features to .TSV file, to be used on http://projector.tensorflow.org
    numpy.savetxt( os.path.join(output_extract, 'features_extracted_by_'+ feature_extraction_layer +'.txt'), features, delimiter='\t')
    
    # Save labels, to be used as "metadata" on http://projector.tensorflow.org
    save_metadata(os.path.join(output_extract, 'metadata.tsv'), metadata_bkpr, metadata_depth=split_all(pathnames_of_interest[0])[-6:-1])
  
    print('Done')


def save_metadata(file, metadata_bkpr, metadata_depth):

    meta = [re.split('\s|(?<!\d)[,._-]|[,._-](?!\d)', i)[0] for i in metadata_depth]

    with open(file, 'w') as f:
        f.write( meta[0] + '\t' + meta[1] + '\t' + meta[2] + '\t' + meta[3] + '\t' + meta[4] + '\n')
        for i in range(len(metadata_bkpr)):
            f.write('{}\t{}\t{}\t{}\t{}\n'.format( metadata_bkpr[i][0], metadata_bkpr[i][1], metadata_bkpr[i][2], metadata_bkpr[i][3], metadata_bkpr[i][4]))


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


DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'SjdnUends821Jsdlkvxh391ksdODnejdDw'


class parse_essential(Form):
    input_parse = TextField('input_parse:', validators=[validators.required()])
    output_parse = TextField('output_parse:', validators=[validators.required()])


class train_essential(Form):
    input_train = TextField('input_train:', validators=[validators.required()])
    # output_train = TextField('output_train:', validators=[validators.required()])
    # class_option = TextField('class_option:', validators=[validators.required()])


class predict_essential(Form):
    input_predict = TextField('input_predict:', validators=[validators.required()])
    output_predict = TextField('output_predict:', validators=[validators.required()])
    class_option = TextField('class_option:', validators=[validators.required()])

    
class extract_essential(Form):
    input_extract = TextField('input_extract:', validators=[validators.required()])
    output_extract = TextField('output_extract:', validators=[validators.required()])
    modellocation = TextField('modellocation:', validators=[validators.required()]) # this time it has to be a .h5 or .hdf5 file, named as ***_categories.h5
    feature_extraction_layer = TextField('feature_extraction_layer:', validators=[validators.required()])


    @app.route("/", methods=['GET', 'POST'])
    def gui():

        # Parse training data   
        if parse_essential(request.form).validate():                
        
            input_parse = request.form['input_parse']
            output_parse = request.form['output_parse']                
            frame = request.form['frame']            
            channels = request.form['channels']
            montage_size = request.form['montage_size']
            train_test_split = request.form['train_test_split']

            if frame=='':
                frame = 48

            if channels=='':
                channels = None

            if montage_size=='':
                # print('No stitching')
                montage_size = 0

            if train_test_split !='':
                output_parse_train = os.path.join(output_parse, 'Train')
                output_parse_test = os.path.join(output_parse, 'Hold_out')

                parsing(input_parse, output_parse_train, frame, channels, montage_size)

                # Transform string train_test_split into a ratio eg. 0.2, 0.3
                split_portion = [float(i) for i in re.split("[,_:/-]", train_test_split)]
                train_test_split = split_portion[1]/sum(split_portion)

                filelist = glob.glob(os.path.join(output_parse_train,'**','*.npy'), recursive=True)
                split_index = int(len(filelist) * (1.0 - train_test_split))
                indexes = numpy.random.permutation(len(filelist))

                test_set = [filelist[index] for index in indexes[split_index:]]  

                # Move some of the files from parsed location to 'hold_out' folder; the rest is for 'train'
                for i in test_set:
                    dest = os.path.dirname(i).replace(output_parse_train, output_parse_test)
                    if not os.path.exists(dest):
                        os.makedirs(dest)

                    os.rename(i, os.path.join(dest, os.path.basename(i)) )
                    
            else:
                parsing(input_parse, output_parse, frame, channels, montage_size)
 
        # Model training   
        elif train_essential(request.form).validate():

            input_train = request.form['input_train']
            output_train = request.form['output_train']
            class_option = request.form['class_option']
            epochs = request.form['epochs']
            if epochs=='':
                epochs = 512

            if ((class_option=='') or (os.path.exists(class_option))):
                all_subdirs = [x[0] for x in os.walk(input_train)]
                list1 = sorted(list(set([os.path.basename(i.lower()) for i in all_subdirs[1:]])))
                keyf = lambda text: re.split('\s|(?<!\d)[,._-]|[,._-](?!\d)', text)[0]
                possible_labels = sorted([sorted(list(items)) for gr, items in groupby(list1, key=keyf)])

                def inner():
                    for x in possible_labels:
                        yield x

                env = Environment(loader=FileSystemLoader('templates'))
                tmpl = env.get_template('Deepometry_GUI.html')
                return Response(tmpl.generate(possible_targets=inner()))
            else:
                if output_train=='':
                    output_train = os.path.join(input_train, 'model_trained_from_this_dataset')

                training(
                    input_train=input_train, 
                    output_train=output_train, 
                    class_option=class_option, 
                    epochs=int(epochs)
                )


        # Evaluate trained model 
        elif predict_essential(request.form).validate():

            input_predict = request.form['input_predict']
            output_predict = request.form['output_predict']
            class_option = request.form['class_option']
            unannotated_box_ticked = request.form['unannotated_box_ticked']
            modellocation = request.form['modellocation']
            if modellocation=='':
                modellocation = None            
            
            evaluating(input_predict, output_predict, modellocation, class_option, unannotated_box_ticked)


        # Feature extraction 
        elif extract_essential(request.form).validate():

            input_extract = request.form['input_extract']
            output_extract = request.form['output_extract']
            feature_extraction_layer = request.form['feature_extraction_layer']
            modellocation=request.form['modellocation']  # this time it has to be a .h5 or .hdf5 file, named as ***_categories.h5

            extracting(input_extract, output_extract, modellocation, feature_extraction_layer)

        else:
            flash('Error: Some critical fields are missing!')
        
        return render_template('Deepometry_GUI.html',form = parse_essential(request.form))


if __name__ == "__main__":
    app.run()