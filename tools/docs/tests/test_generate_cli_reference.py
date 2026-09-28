"""Generated CLI reference tracks the public command contract without execution."""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_cli_reference as reference


class CLIReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = reference.load_cli()

    def test_every_public_registered_command_has_one_section_and_index_link(self):
        content = reference.render(self.cli)
        paths = []

        def collect(command, path):
            paths.append(path)
            if isinstance(command, click.Group):
                for name, child in command.commands.items():
                    if not child.hidden:
                        collect(child, f"{path} {name}")

        collect(self.cli, "vllm-sr")
        self.assertGreater(len(paths), 50)
        self.assertEqual(len(paths), len(list(reference.command_tree(self.cli))))
        for path in paths:
            anchor = path.replace(" ", "-")
            self.assertEqual(content.count(f"`{path}` {{#{anchor}}}"), 1)
            self.assertEqual(content.count(f"[`{path}`](#{anchor})"), 1)
        self.assertIn("vllm-sr recipe builtin init", paths)
        self.assertIn("vllm-sr benchmark plan", paths)
        self.assertIn("vllm-sr benchmark dataset prepare", paths)
        self.assertIn("vllm-sr benchmark replay", paths)
        self.assertNotIn("vllm-sr benchmark intelligence", paths)

    def test_document_contains_real_cli_defaults_choices_arguments_and_help(self):
        content = reference.render(self.cli)
        self.assertIn("--startup-timeout SECONDS", content)
        self.assertIn("--image-pull-policy CHOICE", content)
        self.assertIn("Choices: always, ifnotpresent, never.", content)
        self.assertIn("[default: config.yaml]", content)
        self.assertIn("| `[MESSAGE]...` | Optional argument. Type: text.", content)
        self.assertIn("Explicitly bind one host environment variable", content)
        self.assertNotIn("Sentinel.UNSET", content)
        self.assertNotIn("\b", content)
        self.assertNotIn("/Users/", content)

    def test_generation_never_invokes_commands_parameter_callbacks_or_defaults(self):
        def forbidden(*args, **kwargs):
            self.fail("Documentation generation executed runtime code")

        cli = click.Group("example", callback=forbidden)
        cli.add_command(
            click.Command(
                "run",
                callback=forbidden,
                params=[
                    click.Option(["--dynamic"], default=forbidden, callback=forbidden),
                    click.Option(["--mode"], type=click.Choice(["a", "b"])),
                    click.Option(["--secret"], hidden=True),
                ],
                help="Literal <value> and {name}.\n\n\b\nexample run --mode a",
            )
        )
        cli.add_command(click.Command("internal", hidden=True, callback=forbidden))
        content = reference.render(cli)
        self.assertIn("computed at runtime", content)
        self.assertIn("Literal &lt;value&gt; and &#123;name&#125;.", content)
        self.assertIn("```text\nexample run --mode a\n```", content)
        self.assertNotIn("--secret", content)
        self.assertNotIn("internal", content)

    def test_output_is_deterministic_across_terminal_and_token_environment(self):
        first = reference.render(self.cli)
        with patch.dict(
            os.environ, {"COLUMNS": "30", "VSR_MGMT_TOKEN": "never-publish"}
        ):
            self.assertEqual(reference.render(self.cli), first)

    def test_registered_source_option_change_fails_check_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(
                reference.ROOT / "src/vllm-sr/cli",
                root / "src/vllm-sr/cli",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            schema_name = "router-config-v0.3.schema.json"
            shutil.copyfile(
                reference.ROOT / "src/semantic-router/pkg/configschema" / schema_name,
                root / "src/vllm-sr/cli/config_schema" / schema_name,
            )
            script = root / "tools/docs/generate_cli_reference.py"
            script.parent.mkdir(parents=True)
            shutil.copyfile(reference.__file__, script)
            output = root / "cli.md"
            reference.sync(output, reference.render(self.cli), check=False)
            before = output.read_bytes()
            source = root / "src/vllm-sr/cli/commands/runtime.py"
            original = source.read_text()
            updated = original.replace(
                "Local Docker startup readiness budget in seconds,",
                "Updated local startup readiness budget in seconds,",
            )
            self.assertNotEqual(original, updated)
            source.write_text(updated)
            result = subprocess.run(
                [sys.executable, str(script), "--check", "--output", str(output)],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("missing or outdated", result.stderr)
            self.assertEqual(output.read_bytes(), before)

    def test_missing_reference_fails_check_without_creating_it(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "missing/cli.md"
            with contextlib.redirect_stderr(io.StringIO()):
                result = reference.main(["--check", "--output", str(output)])
            self.assertEqual(result, 1)
            self.assertFalse(output.parent.exists())

    def test_repository_reference_is_current(self):
        self.assertFalse(
            reference.sync(reference.OUTPUT, reference.render(self.cli), check=True)
        )


if __name__ == "__main__":
    unittest.main()
