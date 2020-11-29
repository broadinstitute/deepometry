# coding: utf-8

import sys, os, glob, gc
import collections
import csv

import keras
import keras.models
import keras.preprocessing.image
import keras_resnet.models
import numpy
import pkg_resources
import sklearn.preprocessing

import deepometry.image.generator


class Model(object):
    """
    Create a model for single-cell image classification.
    :param shape: Input image shape, including channels. Grayscale data should specify channels as 1. Check your
                  keras configuration for channel order (e.g., ``"image_data_format": "channels_last"``). Usually,
                  this configuration is defined at ``$HOME/.keras/keras.json``, or ``%USERPROFILE%\.keras\keras.json``
                  on Windows.
    :param units: Number of predictable classes.
    :param directory: (Optional) Output directory for model checkpoints, metrics, and metadata. Otherwise, the
                      package's data directory is used.
    :param name: (Optional) A unique identifier for referencing this model.
    """
    def __init__(self, shape, units, directory=None, name=None):
        self.directory = directory

        self.name = name

        self.units = units

        x = keras.layers.Input(shape)

        self.model = keras_resnet.models.ResNet50(x, classes=units)
        

    def compile(self, lr=0.0001):
        """
        Configure the model.
        """
        self.model.compile(
            loss="categorical_crossentropy",
            metrics=[
                "accuracy"
            ],
            optimizer=keras.optimizers.Adam(lr=lr)
        )


    def evaluate(self, x, y, saved_model_location=None, batch_size=32, verbose=0):
        """
        Compute the loss value & metrics values for the model in test mode.
        Computation is done in batches.
        :param x: NumPy array of test data.
        :param y: NumPy array of target data.
        :param batch_size: Number of samples evaluated per batch.
        :param verbose: Verbosity mode, 0 = silent, or 1 = verbose.
        :return: Tuple of scalars: (loss, accuracy).
        """

        if saved_model_location==None:
            self.model.load_weights(self._resource("checkpoint.hdf5"))
        elif os.path.isfile(saved_model_location):
            self.model.load_weights(saved_model_location)
        elif os.path.isdir(saved_model_location):
            list_of_files = glob.glob(os.path.join(saved_model_location, '*5')) # for h5 and hdf5
            latest_file = max(list_of_files, key=os.path.getctime)
            self.model.load_weights(latest_file)

        return self.model.evaluate(
            x=self._center(x),
            y=keras.utils.to_categorical(y, num_classes=self.units),
            batch_size=batch_size,
            verbose=verbose
        )


    def extract(self, x, selected_layer='pool5', saved_model_location=None, batch_size=32, standardize=False, verbose=0):
        """
        Extract learned features from the model.
        Computation is done in batches.
        :param x: NumPy array of data.
        :param selected_layer: name of the layer to be used as feature extractor (default:``'pool5'``),
        :param saved_model_location: location of the saved weights of a trained model. If multiple files exist, use the last weights.
        :param batch_size: Number of samples evaluated per batch.
        :param standardize: If ``True``, center to the mean and component wise scale to unit variance
            (default: ``False``).
        :param verbose: Verbosity mode, 0 = silent, or 1 = verbose.
        :return: NumPy array of shape ``(samples, features)``.
        """

        if saved_model_location==None:
            self.model.load_weights(self._resource("checkpoint.hdf5"))
        elif os.path.isfile(saved_model_location):
            self.model.load_weights(saved_model_location)
        elif os.path.isdir(saved_model_location):
            list_of_files = glob.glob(os.path.join(saved_model_location, '*5')) # for h5 and hdf5
            latest_file = max(list_of_files, key=os.path.getctime)
            self.model.load_weights(latest_file)

        str_layers = [str(i) for i in self.model.layers]

        response_model = keras.models.Model(
            inputs=self.model.input,
            outputs=self.model.layers[str_layers.index(str(self.model.get_layer(selected_layer)))].output
        )

        features = response_model.predict(
            self._center(x),
            batch_size=batch_size,
            verbose=verbose
        )

        if standardize:
            if len(features.shape) > 2:
                a = numpy.mean(features, axis = 1)
                del(features)
                features = numpy.mean(a, axis = 1)              
            return sklearn.preprocessing.scale(features)

        return features
    

    def fit(self, x, y, x_valid=None, y_valid=None, batch_size=32, class_weight="auto", balance_train=True, balance_valid=False, train_valid_sameset=True, validation_split=0.2, mixup_alpha=0.0, finetune=False, freeze_until_layer='pool5', old_shape=None, old_class=None, old_model=None, trainable_layer='pool5', epochs=512, verbose=0):
        """
        Train the model for a fixed number of epochs (iterations on a dataset). Training will automatically stop
        if the validation loss fails to improve for 20 epochs.

        :param x: NumPy array of training data.
        :param y: NumPy array of target data.
        :param x_valid: NumPy array of data for validation, sometimes from a different dataset.
        :param y_valid: NumPy array of target data for validation, sometimes from a different dataset.       
        :param batch_size: Number of samples per gradient update.
        :param class_weight: Dictionary mapping labels to weights. Use ``"auto"`` to automatically compute weights.
            Use ``None`` to ignore weights.
        :param balance_train: Set to ``True`` to balance class presentation for each batch during training.
        :param balance_valid: Set to ``True`` to balance class presentation for each batch during validation.
        :param train_valid_sameset: Set to ``False`` to perform validation on a data pool that is different from training set.    
        :param validation_split: Fraction of the training data to be used as validation data.
        :param finetune: Set to ``True`` to perform tranfer learning and/or finetuning.
        :param old_shape: Specify the input shape of the transfered model.
        :param old_class: Specify the number of output classes of the transfered model.
        :param old_model: Specify the location of the model to be transfered.
        :param trainable_layer: Specify the layer from which finetuning can occur.
        :param epochs: Number of times to iterate over the training data arrays.        
        :param verbose: Verbosity mode. 0 = silent, 1 = verbose, 2 = one log line per epoch.
        """
        
        if train_valid_sameset:
        
            x_train, y_train, x_valid, y_valid = _split(x, y, validation_split)

        else:
            x_train = x
            y_train = y
            
        del(x,y)    
        gc.collect()
            
        train_generator = self._create_generator()

        valid_generator = self._create_generator()            
            
        self._calculate_means(x_train)

        options = {
            "callbacks": [
                keras.callbacks.CSVLogger(
                    self._resource("training.csv")
                ),
                keras.callbacks.EarlyStopping(patience=50),
                keras.callbacks.ModelCheckpoint(
                    self._resource("checkpoint.hdf5")
                ),
                keras.callbacks.ReduceLROnPlateau(),
                keras.callbacks.TensorBoard(log_dir= pkg_resources.resource_filename("deepometry", os.path.join("./data")), histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False)
            ],
            "class_weight": _class_weights(y) if class_weight == "auto" else class_weight,
            "epochs": epochs,
            "steps_per_epoch": len(x_train) // batch_size,
            "validation_steps": len(x_valid) // batch_size,
            "verbose": verbose
        }

        if finetune:
            trained_model = keras_resnet.models.ResNet50(keras.layers.Input(old_shape), classes=old_class)
            trained_model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
            trained_model.load_weights(old_model)
            trained_model.layers.pop()

            str_layers = [str(i) for i in trained_model.layers]
            print('Old model ', old_model , ' will be transferred. Every layers before ', freeze_until_layer, ' will be frozen. Later layers will be trainable.' )
            for layer in trained_model.layers[:str_layers.index(str(trained_model.get_layer(freeze_until_layer)))+1]:
            #for layer in trained_model.layers:
                layer.trainable=False
            last = trained_model.layers[-1].output
            xx = keras.layers.Dense(self.units, activation="softmax")(last)
            self.model = keras.models.Model(trained_model.input, xx)
            self.model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])        
        
        self.model.fit_generator(
            generator=train_generator.flow(
                x=x_train,
                y=keras.utils.to_categorical(y_train, num_classes=self.units),
                batch_size=batch_size,
                balance=balance_train,
                mixup_alpha = mixup_alpha # change value for alpha
            ),
            validation_data=valid_generator.flow(
                x=x_valid,
                y=keras.utils.to_categorical(y_valid, num_classes=self.units),
                batch_size=batch_size,
                balance=balance_valid,                
                mixup_alpha = 0.0
            ),
            **options
        )
            

    def predict(self, x, saved_model_location=None, batch_size=32, verbose=0):
        """
        Make predictions for the input samples.
        Computation is done in batches.
        :param x: NumPy array of input data.
        :param batch_size: Number of samples predicted per batch.
        :param verbose: Verbosity mode, 0 = silent, or 1 = verbose.
        :return: NumPy array of predictions.
        """

        if saved_model_location==None:
            self.model.load_weights(self._resource("checkpoint.hdf5"))
        elif os.path.isfile(saved_model_location):
            self.model.load_weights(saved_model_location)
        elif os.path.isdir(saved_model_location):
            list_of_files = glob.glob(os.path.join(saved_model_location, '*5')) # for h5 and hdf5
            latest_file = max(list_of_files, key=os.path.getctime)
            self.model.load_weights(latest_file)

        return self.model.predict(self._center(x), batch_size=batch_size, verbose=verbose)
    

    def _calculate_means(self, x):
        reshaped = x.reshape(-1, x.shape[-1])

        means = numpy.mean(reshaped, axis=0)

        with open(self._resource("means.csv"), "w") as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(means)

        return means


    def _center(self, x):
        xc = x.reshape(-1, x.shape[-1])

        xc = ((xc - self._means() + 255.0) / (2.0 * 255.0))
        #xc = xc - self._means()

        return xc.reshape(x.shape)


    def _create_generator(self, balancing=True):

        means = self._means()

        generator_options = {
            "height_shift_range": 0.5,
            "horizontal_flip": True,
            "preprocessing_function": lambda data: ((data - means + 255.0) / (2.0 * 255.0)), # better when you want to neutralize batch effect
            #"preprocessing_function": lambda data: data - means, # better when you care about intensity
            "rotation_range": 180,
            "vertical_flip": True,
            "width_shift_range": 0.5
        }
        
        return deepometry.image.generator.ImageDataGenerator(
            **generator_options
        )


    def _means(self):
        means = None

        with open(self._resource("means.csv"), "r") as csvfile:
            reader = csv.reader(csvfile)

            for row in reader:
                means = [float(mean) for mean in row]

                break

        return means


    def _resource(self, filename):
        if self.name is None:
            resource_filename = filename
        else:
            resource_filename = "{:s}_{:s}".format(self.name, filename)

        if self.directory is None:
            return pkg_resources.resource_filename(
                "deepometry",
                os.path.join("./data", resource_filename)
            )
        
        return os.path.join(self.directory, resource_filename)


def _class_weights(y):
    counter = collections.Counter(y)

    majority = max(counter.values())

    return {cls: float(majority / count) for cls, count in counter.items()}


def _split(x, y, validation_split=0.2):
    split_index = int(len(x) * (1.0 - validation_split))

    indexes = numpy.random.permutation(len(x))

    x_train = numpy.asarray([x[index] for index in indexes[:split_index]])
    x_valid = numpy.asarray([x[index] for index in indexes[split_index:]])

    y_train = numpy.asarray([y[index] for index in indexes[:split_index]])
    y_valid = numpy.asarray([y[index] for index in indexes[split_index:]])

    return x_train, y_train, x_valid, y_valid
