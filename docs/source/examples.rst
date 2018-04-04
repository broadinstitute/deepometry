Examples
========

.. toctree::
   :maxdepth: 2

Parse
^^^^^
*Parsing CIF files is only supported with Python 2.7!*

Before analyzing your images, it is important to preprocess your data and transform it into a format the ``deepometry``
model can use for training, evaluation, and prediction. Parse your data from ``CIF`` and ``TIF`` formats into ``NPY``
data with consistent dimensionality and appropriate rescaling.

Use the ``deepometry parse`` subcommand, or the ``deepometry.parse.parse`` function to preprocess a directory of raw
image data. Subdirectories of the top-level directory are class labels and subdirectory contents are .CIF or
.TIF files containing raw data corresponding to that label.

For the following examples, let's assume ``/data/raw/experiment_00`` contains two subdirectories,
``/data/raw/experiment_00/A`` and ``/data/raw/experiment_00/B``. Each subdirectory contains a series of 12-channel
images. We want to resize our images to a consistent shape of 48x48 pixels. Furthermore, we want to extract only three
channels. (Channels are 0-indexed!) The resized images from each subdirectory will be exported to
``/data/raw/experiment_00/A`` and ``/data/raw/experiment_00/B``, respectively.

CLI
---

::

    deepometry parse                            \
        --channels "0,5-6"                      \
        --image-size 48                         \
        /data/raw/experiment_00/                \
        /data/parsed/experiment_00/


Python
------

.. code-block:: python

   import glob
   import os

   import deepometry.parse


   subdirectories = glob.glob('/data/raw/experiment_00/*')

   for subdirectory in subdirectories:
       output_directory = os.path.join(
           '/data/parsed/experiment_00',
           os.path.split(subdirectory)[-1]  # Append the subdirectory name to the output directory
       )

       # Create the output subdirectory, if it does not exist.
       if not os.path.exists(output_directory):
           os.makedirs(output_directory)

       deepometry.parse.parse(
           paths=glob.glob(os.path.join(subdirectory, '*')),
           output_directory=output_directory,
           size=48,
           channels=[0, 5, 6]
       )

Notebooks
---------

For in-depth examples on pre-processing image data using ``deepometry``, check out these examples:

.. toctree::

  examples/parse_CIF.ipynb
  examples/parse_TIF.ipynb

Fit & Evaluate
^^^^^^^^^^^^^^

Use parsed data to fit and evaluate a ``deepometry`` model.

Data for fitting is expected to be saved as consistently sized NumPy arrays (the ``deepometry parse`` subcommand or
``deepometry.parse.parse`` function does this for you automatically). ``deepometry`` can fit to and evaluate data
across multiple directories. Subdirectories of the top-level directories are class labels and subdirectory contents are
.NPY files containing preprocessed image data corresponding to that label.

For the following examples, let's assume ``/data/parsed/experiment_00``, ``/data/parsed/experiment_01`` and
``/data/parsed/experiment_02`` each contain two subdirectories, ``A`` and ``B``. Each subdirectory contains a series
of 3-channel images as NPY arrays. Each array is 48x48 pixels.

CLI
---

::

   deepometry fit                            \
      --verbose                              \
      /data/parsed/experiment_00             \
      /data/parsed/experiment_01

   deepometry evaluate                       \
      /data/parsed/experiment_02


Python
------

.. code-block:: python

   import deepometry.model
   import deepometry.utils


   # Load training data `x_fit` and labels `y_fit`. Additionally, infer the number of
   # unique labels `units`.
   #
   # `sample=True` performs undersampling to mitigate class imbalance.
   directories = ['/data/parsed/experiment_00', '/data/parsed/experiment_01']
   x_fit, y_fit, units = deepometry.utils.load(directories, sample=True)

   # Instantiate the model.
   model = deepometry.model.Model(shape=x.shape[1:], units=units)
   model.compile()

   # Fit the model to the data.
   model.fit(x_fit, y_fit, verbose=1)

   # Evaluate the model's accuracy.
   x_eval, y_eval, _ = deepometry.utils.load(['/data/parsed/experiment_02'])
   model.evaluate(x_eval, y_eval)

Notebooks
---------

For in-depth examples on fitting and evaluating a model using ``deepometry``, check out these examples:

.. toctree::

   examples/fit.ipynb
   examples/evaluate.ipynb

Visualize
^^^^^^^^^

Extract and visualize learned features.

Explore learned features of a trained model using ``deepometry`` and TensorBoard. TensorBoard can run on your local
machine, and allows you to visualize how well classes are grouped together by a trained model. ``deepometry`` can
extract and visualize features from data across multiple directories. Subdirectories of the top-level directories are
class labels and subdirectory contents are .NPY files containing preprocessed image data corresponding to that label.

For the following examples, let's assume we have already fit a model to some data. Features will be extracted from
``/data/parsed/experiment_00``, which contains two subdirectories, ``A`` and ``B``. Each subdirectory contains a series
of 3-channel images as NPY arrays. Each array is 48x48 pixels.

CLI
---

The ``deepometry extract`` subcommand will output the location of the features and metadata TSVs. Navigate to the
address provided by ``deepometry visualize`` to view the extracted features on TensorBoard.

::

   deepometry extract                        \
      --output-directory ./extracted         \
      --verbose                              \
      /data/parsed/experiment_00

   deepometry visualize                      \
      --metadata ./extracted/metadata.tsv    \
      ./extracted/features.tsv

Python
------

.. code-block:: python

   import pandas

   import deepometry.utils
   import deepometry.visualize


   # Load training data `x` and labels `labels`. Additionally, infer the number of
   # unique labels `units`.
   #
   # `convert=False` won't convert class labels to integers, so `labels` is a list of strings.
   # `sample=256` limits the maximum number of samples per class.
   x, labels, units = deepometry.utils.load(['/data/parsed/experiment_00'], convert=False, sample=256)

   # Instantiate a pre-trained model.
   model = deepometry.model.Model(shape=x.shape[1:], units=units)

   # Extract features from a pre-trained model.
   features = model.extract(x, batch_size=32, standardize=False, verbose=1)

   # Export features to a TSV, exclude column headers.
   features_df = pandas.DataFrame(data=features)
   features_df.to_csv('./extracted/features.tsv', header=False, index=False, sep='\t')

   # Export metadata to a TSV, exclude column headers.
   metadata_df = pandas.DataFrame(data=labels)
   metadata_df.to_csv('./extracted/metadata.tsv', header=False, index=False, sep='\t')

   # Initialize TensorBoard visualization.
   log_directory = deepometry.visualize.make_projection(
      './extracted/features.tsv',
      metadata='./extracted/metadata.tsv'
   )

   # Run the output of this in a terminal to start TensorBoard.
   # Navigate to the address provided to view the extracted features.
   print('tensorboard --logdir {:s}'.format(log_directory))

Notebooks
---------

For in-depth examples on extracting and visualizing learned features using ``deepometry``, check out these examples:

.. toctree::

   examples/extract.ipynb
   examples/visualize.ipynb