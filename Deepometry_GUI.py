# Native
from random import randint
import sys, os, glob, re
import time
from time import strftime
import pickle
import itertools
from collections import Counter
import pkg_resources

# Libraries
import javabridge
import bioformats
import numpy
import pandas
import seaborn
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


def parsing(input_dir, output_dir, frame, channels, montage_size):
    '''
    Convert images from .CIF or .TIF formats to numpy arrays, organized in a structured folder tree.
    '''

    channels = re.findall(r'\d+', channels)
    channels = [int(i) for i in channels]

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
                                        size=int(frame),
                                        channels=channels,
                                        montage_size=int(montage_size)
                                    )

    print('Done')


def training(input_dir4, output_dir4, class_option, n_samples=None, validation_split=0.2, batch_size=32, epochs=512):
    '''
    Train a deep neural network
    '''

    all_subdirs = [x[0] for x in os.walk(input_dir4)]
    possible_labels = sorted(list(set([os.path.basename(i) for i in all_subdirs])))
    labels_of_interest = [i for i in possible_labels if class_option.lower() in i.lower()]

    pathnames_of_interest = deepometry.utils.collect_pathnames(input_dir4, labels_of_interest, n_samples=n_samples)

    x, y, _ = deepometry.utils._load(pathnames_of_interest, labels_of_interest)

    units = len(list(set(labels_of_interest)))

    apply_session()

    model = deepometry.model.Model(shape=x.shape[1:], units=units)

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
        verbose=1
    )

    print("Save model to disk...")
    if not os.path.exists(output_dir4):
        os.makedirs(output_dir4)
    
    model.model.save( os.path.join(output_dir4, 'model_'+ class_option + '_' + str(units) + '_categories.h5') ) # The 'units' in model name is very important for feature extraction module!

    print('Done')


def evaluating(input_dir4, input_dir5, output_dir5, modellocation, class_option):  
    '''
    Evaluate a trained deep neural network on annotated ground truth
    '''

    all_subdirs = [x[0] for x in os.walk(input_dir4)]
    possible_labels = sorted(list(set([os.path.basename(i) for i in all_subdirs])))
    labels_of_interest = [i for i in possible_labels if class_option.lower() in i.lower()]

    pathnames_of_interest = deepometry.utils.collect_pathnames(input_dir5, labels_of_interest, n_samples=None)

    x, y, _ = deepometry.utils._load(pathnames_of_interest, labels_of_interest)

    units = len(list(set(labels_of_interest)))

    apply_session()

    model = deepometry.model.Model(shape=x.shape[1:], units=units)

    model.compile()

    predicted = model.predict(x, modellocation, batch_size=32)
    
    predicted = numpy.argmax(predicted, -1)
    expected = y
    
    confusion = sklearn.metrics.confusion_matrix(expected, predicted)

    # Normalize values in confusion matrix
    confusion = confusion.astype('float') / confusion.sum(axis=1)[:, numpy.newaxis]

    confusion = pandas.DataFrame(confusion)
    confusion = confusion.rename(index={index: label for index, label in enumerate(labels_of_interest)}, columns={index: label for index, label in enumerate(labels_of_interest)})

    fig = plt.figure(figsize=(12, 8))

    seaborn.heatmap(confusion, annot=True) 

    print("Saved plots and values")

    if not os.path.exists(output_dir5):
        os.makedirs(output_dir5)   

    fig.savefig(os.path.join(output_dir5,'confusion_matrix_plot.png'), dpi = 300)

    # save the confusion matrix values to a csv_file.
    confusion.to_csv( os.path.join(output_dir5,'confusion_matrix.csv') )

    # save the accuracy metrics to a csv_file.
    val = sklearn.metrics.accuracy_score(expected, predicted)
    file_to_use = os.path.join(output_dir5,'accuracy_values.csv')
    with open(file_to_use, 'w') as f:
        f.write(str(val))
  
    print('Done')


def predicting(input_dir4, input_dir6, output_dir6, modellocation2, class_option):
    '''
    Use a trained deep neural network to predict unannotated ground truth
    '''

    # Collect labels from training data
    all_subdirs = [x[0] for x in os.walk(input_dir4)]
    possible_labels = sorted(list(set([os.path.basename(i) for i in all_subdirs])))
    labels_of_interest = [i for i in possible_labels if class_option.lower() in i.lower()]

    # Collect unannotated data
    
    pathnames_of_interest = deepometry.utils.collect_pathnames(input_dir6, labels_of_interest, n_samples=None)

    x, _, _ = deepometry.utils._load(pathnames_of_interest, labels_of_interest)

    units = len(list(set(labels_of_interest)))

    apply_session()

    model = deepometry.model.Model(shape=x.shape[1:], units=units)

    model.compile()
   
    predicted = model.predict(x, modellocation2, batch_size=32)
    
    predicted = numpy.argmax(predicted, -1)


    print("Save predicted values")

    if not os.path.exists(output_dir6):
        os.makedirs(output_dir6)    

    predicted_classes = pandas.DataFrame()
    predicted_classes['numeric_class'] = predicted
    predicted_classes['label'] = [labels_of_interest[i] for i in predicted]

    predicted_classes.to_csv(os.path.join(output_dir6,'predicted.csv'), index=True, index_label='ID')
    
    print('Done')


def extracting(input_dir5, output_dir5, modellocation, feature_extraction_layer):
    '''
    Use a trained deep neural network to extract deep learning embeddings
    '''

    # Collect data
    pathnames_of_interest = [j for i in [x[0] for x in os.walk(input_dir5)] for j in glob.glob(os.path.join(i,'*')) if '.npy' in j]

    x, _, metadata_bkpr = deepometry.utils._load(pathnames_of_interest)

    if os.path.isdir(modellocation):
        list_of_files = glob.glob(os.path.join(modellocation, '*_categories.h5')) # only file with correct naming, i.e.***_categories.h5, is accepted
        modellocation = max(list_of_files, key=os.path.getctime)

    units = int( re.search('model_.*_([0-9]*)_categories.h5', os.path.basename(modellocation)).group(1) )
    
    apply_session()

    model = deepometry.model.Model(shape=x.shape[1:], units=units)

    model.compile()

    # Extract the features
    features = model.extract(x, feature_extraction_layer, modellocation, batch_size=32, standardize=True, verbose=1)

    if len(features.shape) > 2:
        a = numpy.mean(features, axis = 1)
        del(features)
        features = numpy.mean(a, axis = 1)  

    if not os.path.exists(output_dir5):
        os.makedirs(output_dir5)   

    # Export features to .TSV file, to be used as "metadata" on http://projector.tensorflow.org
    numpy.savetxt( os.path.join(output_dir5, 'features_extracted_by_'+ feature_extraction_layer +'.txt'), features, delimiter='\t')
    
    # Save labels, to be used as "metadata" on http://projector.tensorflow.org
    save_metadata(os.path.join(output_dir5, 'metadata.tsv'), metadata_bkpr)
  
    print('Done')


def save_metadata(file, metadata_bkpr):
    with open(file, 'w') as f:
        f.write('Experiments\tDays\tSamples\tReplicates\tClasses\n')
        for i in range(len(metadata_bkpr)):
            f.write('{}\t{}\t{}\t{}\t{}\n'.format( metadata_bkpr[i][0], metadata_bkpr[i][1], metadata_bkpr[i][2], metadata_bkpr[i][3], metadata_bkpr[i][4]))


# def write_to_disk(name, output_dir, email):
#     data = open('file.log', 'a')
#     timestamp = get_time()
#     data.write('DateStamp={}, Name={}, output_dir={}, Email={} \n'.format(timestamp, name, output_dir, email))
#     data.close()


DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'SjdnUends821Jsdlkvxh391ksdODnejdDw'


class parse_train_essential(Form):
    input_dir = TextField('input_dir:', validators=[validators.required()])
    output_dir = TextField('output_dir:', validators=[validators.required()])
    frame = TextField('frame:', validators=[validators.required()])
    channels = TextField('channels:', validators=[validators.required()])


class parse_eval_essential(Form):
    input_dir2 = TextField('input_dir2:', validators=[validators.required()])
    output_dir2 = TextField('output_dir2:', validators=[validators.required()])
    frame = TextField('frame:', validators=[validators.required()])
    channels = TextField('channels:', validators=[validators.required()])


class parse_predict_essential(Form):
    input_dir3 = TextField('input_dir3:', validators=[validators.required()])
    output_dir3 = TextField('output_dir3:', validators=[validators.required()])
    frame = TextField('frame:', validators=[validators.required()])
    channels = TextField('channels:', validators=[validators.required()])


class train_essential(Form):
    input_dir4 = TextField('input_dir4:', validators=[validators.required()])
    output_dir4 = TextField('output_dir4:', validators=[validators.required()])
    class_option = TextField('class_option:', validators=[validators.required()])


class eval_essential(Form):
    input_dir4 = TextField('input_dir4:', validators=[validators.required()])
    input_dir5 = TextField('input_dir5:', validators=[validators.required()])
    output_dir5 = TextField('output_dir5:', validators=[validators.required()])
    class_option = TextField('class_option:', validators=[validators.required()])
  

class predict_essential(Form):
    input_dir4 = TextField('input_dir4:', validators=[validators.required()])
    input_dir6 = TextField('input_dir6:', validators=[validators.required()])
    output_dir6 = TextField('output_dir6:', validators=[validators.required()])
    class_option = TextField('class_option:', validators=[validators.required()])

    
class extract_essential(Form):
    input_dir5 = TextField('input_dir5:', validators=[validators.required()])
    output_dir5 = TextField('output_dir5:', validators=[validators.required()])
    modellocation = TextField('modellocation:', validators=[validators.required()]) # this time it has to be a .h5 or .hdf5 file, named as ***_categories.h5
    feature_extraction_layer = TextField('feature_extraction_layer:', validators=[validators.required()])


    @app.route("/", methods=['GET', 'POST'])
    def gui():

        # Parse training data   
        if parse_train_essential(request.form).validate():                
        
            input_dir = request.form['input_dir']
            output_dir=request.form['output_dir']
            frame=request.form['frame']            
            channels=request.form['channels']
            montage_size=request.form['montage_size']
            if montage_size=='':
                # print('No stitching')
                montage_size = 0

            parsing(input_dir, output_dir, frame, channels, montage_size)

            flash('Parsing data: {} {}'.format(input_dir, output_dir))


        # Parse evaluating data   
        elif parse_eval_essential(request.form).validate():
            input_dir2 = request.form['input_dir2']
            output_dir2=request.form['output_dir2']
            frame=request.form['frame']         
            channels=request.form['channels']

            montage_size=request.form['montage_size']
            if montage_size=='':
                montage_size = 0

            parsing(input_dir2, output_dir2, frame, channels, montage_size)


        # Parse unannotated data   
        elif parse_predict_essential(request.form).validate():
            input_dir3 = request.form['input_dir3']
            output_dir3=request.form['output_dir3']
            frame=request.form['frame']         
            channels=request.form['channels']

            montage_size=request.form['montage_size']
            if montage_size=='':
                montage_size = 0

            parsing(input_dir3, output_dir3, frame, channels, montage_size)


        # Model training   
        elif train_essential(request.form).validate():

            input_dir4 = request.form['input_dir4']
            output_dir4=request.form['output_dir4']
            class_option = request.form['class_option']

            epochs = request.form['epochs']
            if epochs=='':
                epochs = 512

            training(
                input_dir4=input_dir4, 
                output_dir4=output_dir4, 
                class_option=class_option, 
                epochs=int(epochs)
            )


        # Evaluate trained model 
        elif eval_essential(request.form).validate():

            input_dir4 = request.form['input_dir4'] 
            input_dir5 = request.form['input_dir5']
            output_dir5=request.form['output_dir5']
            class_option = request.form['class_option']
            modellocation=request.form['modellocation']
            if modellocation=='':
                modellocation = None            
            
            evaluating(input_dir4, input_dir5, output_dir5, modellocation, class_option)


        # Predict unannotated data
        elif predict_essential(request.form).validate():

            input_dir4 = request.form['input_dir4'] 
            input_dir6 = request.form['input_dir6']          
            output_dir6=request.form['output_dir6']
            modellocation2=request.form['modellocation2']
            class_option = request.form['class_option']
            if modellocation2=='':
                modellocation2 = None     

            predicting(input_dir4, input_dir6, output_dir6, modellocation2, class_option)


        # Feature extraction 
        elif extract_essential(request.form).validate():

            input_dir5 = request.form['input_dir5']
            output_dir5 = request.form['output_dir5']
            feature_extraction_layer = request.form['feature_extraction_layer']
            modellocation=request.form['modellocation']  # this time it has to be a .h5 or .hdf5 file, named as ***_categories.h5

            extracting(input_dir5, output_dir5, modellocation, feature_extraction_layer)

        else:
            flash('Error: Some critical fields are missing!')

        return render_template('Deepometry_GUI.html',form = parse_train_essential(request.form))

if __name__ == "__main__":
    app.run()