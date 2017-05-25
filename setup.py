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
        "imbalanced-learn",
        "keras"
    ],
    license="BSD",
    name="deepometry",
    packages=setuptools.find_packages(
        exclude=[
            "tests"
        ]
    ),
    url="https://github.com/broadinstitute/deepometry",
    version="0.0.0"
)
