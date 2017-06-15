import setuptools


setuptools.setup(
    author="Claire McQuin",
    author_email="mcquincl@gmail.com",
    extras_require={
        "test": [
            "pytest"
        ]
    },
    install_requires=[
        "javabridge",
        "keras",
        "keras-resnet",
        "numpy",
        "python-bioformats",
        "scikit-learn",
        "scipy"
    ],
    license="BSD",
    name="deepometry",
    package_data={
        "deepometry": [
            "data/checkpoint.hdf5"
        ],
    },
    packages=setuptools.find_packages(
        exclude=[
            "tests"
        ]
    ),
    url="https://github.com/broadinstitute/deepometry",
    version="0.0.0"
)
