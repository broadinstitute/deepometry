import os.path
import time

import bioformats
import click.testing
import pytest

import deepometry.command
import deepometry.parse


@pytest.fixture()
def filenames():
    return ["bar.cif", "baz.cif", "foo.cif"]


@pytest.fixture()
def input_directory(filenames, labels, tmpdir):
    directory = tmpdir.mkdir("original")

    for label in labels:
        label_directory = directory.mkdir(label)

        for filename in filenames:
            pathname = label_directory.join(filename)

            with open(str(pathname), "w") as f:
                f.write(str(time.time()))

    return directory


@pytest.fixture()
def labels():
    return ["bats", "cats", "hats"]


@pytest.fixture()
def output_directory(tmpdir):
    return tmpdir.mkdir("parsed")


def test_parse_help():
    runner = click.testing.CliRunner()

    result = runner.invoke(deepometry.command.command, ["parse", "--help"])

    assert "parse [OPTIONS] INPUT OUTPUT" in result.output

    assert "--channels TEXT" in result.output

    assert "--image-size INTEGER" in result.output


def test_parse_default(filenames, input_directory, labels, mocker, output_directory):
    runner = click.testing.CliRunner()

    start_vm = mocker.patch("javabridge.start_vm")

    stop_vm = mocker.patch("javabridge.kill_vm", unsafe=True)

    parser = mocker.patch("deepometry.parse.parse")

    deepometry.parse.parse.return_value = parser

    cmd = runner.invoke(deepometry.command.command, ["parse", str(input_directory), str(output_directory)])

    # Assert the command exited successfully.
    assert cmd.exit_code == 0, cmd.output

    # Assert the JVM was started.
    start_vm.assert_called_once_with(class_path=bioformats.JARS)

    # Assert we called deepometry.parse.parse with the expected parameters.
    expected_calls = []

    for label in labels:
        for filename in filenames:
            expected_calls.append(mocker.call(os.path.join(str(input_directory), label, filename), 48, None))

    assert parser.call_args_list == expected_calls

    # Assert the parsed images were saved as NumPy arrays
    for label in labels:
        expected_subdirectory = os.path.join(str(output_directory), label)

        assert os.path.exists(expected_subdirectory)

        assert os.path.isdir(expected_subdirectory)

        for filename in filenames:
            name, _ = os.path.splitext(filename)

            expected_pathname = os.path.join(str(output_directory), label, "{}.npy".format(name))

            assert os.path.exists(expected_pathname)

            assert os.path.isfile(expected_pathname)

    # Assert we stopped the JVM.
    stop_vm.assert_called_once()


def test_parse_with_image_size(filenames, input_directory, labels, mocker, output_directory):
    runner = click.testing.CliRunner()

    start_vm = mocker.patch("javabridge.start_vm")

    stop_vm = mocker.patch("javabridge.kill_vm", unsafe=True)

    parser = mocker.patch("deepometry.parse.parse")

    deepometry.parse.parse.return_value = parser

    cmd = runner.invoke(
        deepometry.command.command,
        ["parse", str(input_directory), str(output_directory), "--image-size", 56]
    )

    # Assert the command exited successfully.
    assert cmd.exit_code == 0, cmd.output

    # Assert the JVM was started.
    start_vm.assert_called_once_with(class_path=bioformats.JARS)

    # Assert we called deepometry.parse.parse with the expected parameters.
    expected_calls = []

    for label in labels:
        for filename in filenames:
            expected_calls.append(mocker.call(os.path.join(str(input_directory), label, filename), 56, None))

    assert parser.call_args_list == expected_calls

    # Assert the parsed images were saved as NumPy arrays
    for label in labels:
        expected_subdirectory = os.path.join(str(output_directory), label)

        assert os.path.exists(expected_subdirectory)

        assert os.path.isdir(expected_subdirectory)

        for filename in filenames:
            name, _ = os.path.splitext(filename)

            expected_pathname = os.path.join(str(output_directory), label, "{}.npy".format(name))

            assert os.path.exists(expected_pathname)

            assert os.path.isfile(expected_pathname)

    # Assert we stopped the JVM.
    stop_vm.assert_called_once()


def test_parse_with_channels(filenames, input_directory, labels, mocker, output_directory):
    runner = click.testing.CliRunner()

    start_vm = mocker.patch("javabridge.start_vm")

    stop_vm = mocker.patch("javabridge.kill_vm", unsafe=True)

    parser = mocker.patch("deepometry.parse.parse")

    deepometry.parse.parse.return_value = parser

    cmd = runner.invoke(
        deepometry.command.command,
        ["parse", str(input_directory), str(output_directory), "--channels", "0,3-7,9"]
    )

    # Assert the command exited successfully.
    assert cmd.exit_code == 0, cmd.output

    # Assert the JVM was started.
    start_vm.assert_called_once_with(class_path=bioformats.JARS)

    # Assert we called deepometry.parse.parse with the expected parameters.
    expected_calls = []

    expected_channels = [0, 3, 4, 5, 6, 7, 9]

    for label in labels:
        for filename in filenames:
            expected_calls.append(
                mocker.call(os.path.join(str(input_directory), label, filename), 48, expected_channels)
            )

    assert parser.call_args_list == expected_calls

    # Assert the parsed images were saved as NumPy arrays
    for label in labels:
        expected_subdirectory = os.path.join(str(output_directory), label)

        assert os.path.exists(expected_subdirectory)

        assert os.path.isdir(expected_subdirectory)

        for filename in filenames:
            name, _ = os.path.splitext(filename)

            expected_pathname = os.path.join(str(output_directory), label, "{}.npy".format(name))

            assert os.path.exists(expected_pathname)

            assert os.path.isfile(expected_pathname)

    # Assert we stopped the JVM.
    stop_vm.assert_called_once()
