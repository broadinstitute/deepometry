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


def load_samples(pathnames):
    sample_pathnames = sum(pathnames, [])

    samples = numpy.empty((len(sample_pathnames), 48, 48, 3), dtype=numpy.uint8)
    targets = numpy.empty((len(sample_pathnames),), dtype=numpy.uint8)

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


def test_evaluate_help(cli_runner):
    result = cli_runner.invoke(deepometry.command.command, ["evaluate", "--help"])

    assert "evaluate [OPTIONS] INPUT..." in result.output

    assert "--batch-size INTEGER" in result.output

    assert "--directory PATH" in result.output

    assert "--name TEXT" in result.output

    assert "--samples INTEGER" in result.output


def test_evaluate(cli_runner, mocker, tmpdir):
    input1 = tmpdir.mkdir("experiment_01")
    input1_pathnames = create_samples(input1)

    input2 = tmpdir.mkdir("experiment_02")
    input2_pathnames = create_samples(input2)

    input1_samples, input1_targets = load_samples(input1_pathnames)
    input2_samples, input2_targets = load_samples(input2_pathnames)

    expected_samples = numpy.concatenate((input1_samples, input2_samples))
    expected_targets = numpy.concatenate((input1_targets, input2_targets))

    assert len(expected_samples) == len(expected_targets), \
        "Expected sample size ({}) and target size ({}) mismatch".format(expected_samples.shape, expected_targets.shape)

    model_dir = tmpdir.mkdir("models")

    with mocker.patch("deepometry.model.Model") as model_mock:
        deepometry.model.Model.return_value = model_mock

        cmd = cli_runner.invoke(
            deepometry.command.command,
            [
                "evaluate",
                "--batch-size", "128",
                "--directory", str(model_dir),
                "--name", "zoolander",
                "--verbose",
                str(input1),
                str(input2)
            ]
        )

        assert cmd.exit_code == 0, cmd.exception

        model_mock.compile.assert_called_once_with()

        model_mock.evaluate.assert_called_once_with(
            mocker.ANY,
            mocker.ANY,
            batch_size=128,
            verbose=1
        )

        (samples, targets), _ = model_mock.evaluate.call_args

        assert len(samples) == len(expected_samples)
        numpy.testing.assert_array_equal(samples, expected_samples)

        assert len(targets) == len(expected_targets)
        numpy.testing.assert_array_equal(targets, expected_targets)
