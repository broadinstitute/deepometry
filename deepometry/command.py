import os
import os.path
import sys

import click


class Command(click.MultiCommand):
    def list_commands(self, context):
        rv = []

        commands = os.path.abspath(os.path.join(os.path.dirname(__file__), "commands"))

        for filename in os.listdir(commands):
            if filename.endswith(".py") and filename.startswith("command_"):
                _, name = filename.split("command_")

                name, _ = name.split(".py")

                rv.append(name)

        rv.sort()

        return rv

    def get_command(self, context, name):
        try:
            if sys.version_info[0] == 2:
                name = name.encode("ascii", "replace")

            name = "deepometry.commands.command_" + name

            mod = __import__(name, None, None, ["command"])
        except ImportError:
            return

        return mod.command


command = Command()