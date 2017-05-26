# deepometry
Image classification for imaging flow cytometry.

## Installation

### Dependencies
```
git clone https://github.com/broadinstitute/deepometry.git
cd deepometry
pip install --editable .
```

## Parse

Supports converting .CIF files to NumPy arrays.

```python
import bioformats
import javabridge

import deepometry.parse


javabridge.start_vm(class_path=bioformats.JARS)

data = deepometry.parse("cells.cif", size=48, channels=[2, 11])

javabridge.kill_vm()
```

## Fit

### Using the model

```python
import keras.callbacks
import keras.losses
import keras.optimizers
import numpy

import deepometry.model


model = deepometry.model.Model(shape=(48, 48, 2), classes=2)

model.compile(
    loss=keras.losses.categorical_crossentropy,
    metrics=[
        "accuracy"
    ],
    optimizer=keras.optimizers.Adam(0.00001)
)

x = numpy.load("x.npy")  # Input data (N_samples, 48, 48, 2)
y = numpy.load("y.npy")  # Target labels (N_samples,)

model.fit(
    x,
    y,
    batch_size=32,
    callbacks=[
        keras.callbacks.ModelCheckpoint("checkpoint.hdf5")  # save model
    ]
    epochs=8,
    shuffle=True,
    validation_split=0.2
)
```

### Using the classifier

```python
import numpy

import deepometry.classifier


classifier = deepometry.classifier.Classifier(input_shape=(48, 48, 2), classes=2)

x = numpy.load("x.npy")  # Input data (N_samples, 48, 48, 2)
y = numpy.load("y.npy")  # Target labels (N_samples,)

classifier.fit(x, y)
```
