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
    package_data={
        "deepometry": [
            "resources/logback.xml"
        ]
    },
    url="https://github.com/broadinstitute/deepometry",
    version="0.0.1"
)
