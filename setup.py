import os.path

import setuptools


setuptools.setup(
    author="Minh Doan",
    author_email="deepometry@gmail.com",
    entry_points="""
    [console_scripts]
    deepometry=deepometry.command:command
    """,
    extras_require={
        "development": [
            "ipython==7.13.0",
            "ipykernel",
            "nbsphinx",
            "pandoc",
            "pytest",
            "pytest-mock",
            "sphinx",
            "sphinx_rtd_theme",
            "tensorflow-gpu==1.9.0"
        ]
    },
    install_requires=[
        "click",
        "flask==1.1.2",
        "javabridge==1.0.19",
        "Keras==2.1.5",  # TODO: See deepometry.iterator.NumpyArrayIterator's `_get_batches_of_transformed_samples`.
        "keras-resnet==0.0.7",
        "numpy==1.18.1",
        "opencv-python==4.2.0.34",
        "pandas==1.0.3",
        "python-bioformats==1.5.2",
        "scipy==1.4.1",
        "scikit-image==0.16.2",
        "scikit-learn==0.22.1",
        "seaborn==0.10.1",
        "wtforms==2.3.3"
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
    version="1.0.0"
)
