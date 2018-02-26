import os.path

import setuptools


setuptools.setup(
    author="Claire McQuin",
    author_email="mcquincl@gmail.com",
    entry_points="""
    [console_scripts]
    deepometry=deepometry.command:command
    """,
    extras_require={
        "test": [
            "pytest",
            "pytest-mock"
        ]
    },
    install_requires=[
        "click",
        "javabridge",
        "Keras==2.1.5",  # TODO: See deepometry.iterator.NumpyArrayIterator's `_get_batches_of_transformed_samples`.
        "keras-resnet>=0.0.7",
        "numpy",
        "pandas",
        "python-bioformats",
        "scipy",
        "scikit-image",
        "scikit-learn"
    ],
    license="BSD",
    name="deepometry",
    packages=setuptools.find_packages(
        exclude=[
            "examples",
            "tests"
        ]
    ),
    package_data={
        "deepometry": [
            os.path.join("data", "checkpoint.hdf5"),
            os.path.join("data", "means.csv"),
            os.path.join("data", "training.csv"),
            os.path.join("resources", "logback.xml")
        ]
    },
    url="https://github.com/broadinstitute/deepometry",
    version="0.0.1"
)
