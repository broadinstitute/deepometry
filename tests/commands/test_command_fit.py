import os

import click.testing
import numpy
import pytest

import deepometry.command
import deepometry.model


def create_samples(directory):
    pathnames = []

    for label in ["g1", "g2", "m", "s"]:
        label_dir = directory.mkdir(label)

        label_pathnames = []

        for i in range(numpy.random.randint(1, 10)):
            data = numpy.random.randint(256, size=(48, 48, 3))

            pathname = str(label_dir.join("{:02d}.npy".format(i)))

            numpy.save(pathname, data)

            label_pathnames.append(pathname)

        pathnames.append(label_pathnames)

    return pathnames


def select_samples(pathnames):
    n_samples = min([len(subpathnames) for subpathnames in pathnames])

    sample_pathnames = [list(numpy.random.permutation(subpathnames)[:n_samples]) for subpathnames in pathnames]
    sample_pathnames = sum(sample_pathnames, [])

    samples = numpy.empty((4 * n_samples, 48, 48, 3), dtype=numpy.uint8)
    targets = numpy.empty((4 * n_samples,), dtype=numpy.uint8)

    for index, sample_pathname in enumerate(sample_pathnames):
        samples[index] = numpy.load(sample_pathname)

        label = os.path.split(os.path.dirname(sample_pathname))[-1]
        if label == "g1":
            targets[index] = 0
        elif label == "g2":
            targets[index] = 1
        elif label == "m":
            targets[index] = 2
        elif label == "s":
            targets[index] = 3

    return samples, targets


@pytest.fixture()
def cli_runner():
    return click.testing.CliRunner()


def test_fit_help(cli_runner):
    result = cli_runner.invoke(deepometry.command.command, ["fit", "--help"])

    assert "fit [OPTIONS] INPUT..." in result.output

    assert "--batch-size INTEGER" in result.output

    assert "--directory PATH" in result.output

    assert "--epochs INTEGER" in result.output

    assert "--name TEXT" in result.output

    assert "--validation-split FLOAT" in result.output


def test_fit(cli_runner, mocker, tmpdir):
    input1 = tmpdir.mkdir("experiment_01")
    input1_pathnames = create_samples(input1)

    input2 = tmpdir.mkdir("experiment_02")
    input2_pathnames = create_samples(input2)

    numpy.random.seed(17)

    input1_samples, input1_targets = select_samples(input1_pathnames)
    input2_samples, input2_targets = select_samples(input2_pathnames)

    expected_samples = numpy.concatenate((input1_samples, input2_samples))
    expected_targets = numpy.concatenate((input1_targets, input2_targets))

    model_dir = tmpdir.mkdir("models")

    with mocker.patch("deepometry.model.Model") as model_mock:
        numpy.random.seed(17)

        deepometry.model.Model.return_value = model_mock

        cmd = cli_runner.invoke(
            deepometry.command.command,
            [
                "fit",
                "--batch-size", "128",
                "--directory", str(model_dir),
                "--epochs", "512",
                "--name", "zoolander",
                "--validation-split", "0.5",
                "--verbose",
                str(input1),
                str(input2)
            ]
        )

        assert cmd.exit_code == 0, cmd.exception

        model_mock.compile.assert_called_once_with()

        model_mock.fit.assert_called_once_with(
            mocker.ANY,
            mocker.ANY,
            batch_size=128,
            epochs=512,
            validation_split=0.5,
            verbose=1
        )

        (samples, targets), _ = model_mock.fit.call_args

        assert len(samples) >= 8
        assert len(targets) >= 8
        assert len(samples) == len(targets)
        assert sum(targets == 0) >= 2
        assert sum(targets == 0) == sum(targets == 1) == sum(targets == 2) == sum(targets == 3)

        numpy.testing.assert_array_equal(samples, expected_samples)
        numpy.testing.assert_array_equal(targets, expected_targets)
