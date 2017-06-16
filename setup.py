import setuptools


setuptools.setup(
    author="Claire McQuin",
    author_email="mcquincl@gmail.com",
    extras_require={
        "test": [
            "pytest",
            "pytest-mock"
        ]
    },
    install_requires=[
        "javabridge",
        "numpy",
        "python-bioformats",
        "scipy",
        "scikit-image"
    ],
    license="BSD",
    name="deepometry",
    packages=setuptools.find_packages(
        exclude=[
            "examples",
            "tests"
        ]
    ),
    url="https://github.com/broadinstitute/deepometry",
    version="0.0.1"
)
