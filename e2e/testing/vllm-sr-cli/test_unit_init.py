#!/usr/bin/env python3
"""
test_vllm_sr_init.py - Tests for 'vllm-sr init' command.

This test validates the initialization command:
- Creates config.yaml from template
- Creates .vllm-sr/ directory with default files
- Handles --force flag for overwriting
- Validates created file contents

Signed-off-by: vLLM-SR Team
"""

import os
import sys
import unittest

from cli_test_base import CLITestBase


class TestVllmSRInit(CLITestBase):
    """Tests for the vllm-sr init command."""

    def test_init_creates_config_yaml(self):
        """Test that init creates config.yaml file."""
        self.print_test_header(
            "Init Creates config.yaml",
            "Validates that 'vllm-sr init' creates a config.yaml file in current directory",
        )

        # Run init command
        return_code, stdout, stderr = self.run_cli(["init"])

        # Verify command succeeded
        self.assertEqual(return_code, 0, f"init command failed: {stderr}")

        # Verify config.yaml was created
        config_path = os.path.join(self.test_dir, "config.yaml")
        self.assertFileExists(config_path, "config.yaml was not created")

        # Verify it contains expected content
        self.assertFileContains(
            config_path, "version:", "config.yaml missing version field"
        )
        self.assertFileContains(
            config_path, "listeners:", "config.yaml missing listeners field"
        )

        self.print_test_result(True, "config.yaml created successfully")

    def test_init_creates_vllm_sr_directory(self):
        """Test that init creates .vllm-sr/ directory with template files."""
        self.print_test_header(
            "Init Creates .vllm-sr/ Directory",
            "Validates that 'vllm-sr init' creates .vllm-sr/ directory with defaults",
        )

        # Run init command
        return_code, stdout, stderr = self.run_cli(["init"])

        # Verify command succeeded
        self.assertEqual(return_code, 0, f"init command failed: {stderr}")

        # Verify .vllm-sr directory was created
        vllm_sr_dir = os.path.join(self.test_dir, ".vllm-sr")
        self.assertDirExists(vllm_sr_dir, ".vllm-sr/ directory was not created")

        # Verify expected files exist in .vllm-sr/
        expected_files = ["router-defaults.yaml", "envoy.template.yaml"]
        for expected_file in expected_files:
            file_path = os.path.join(vllm_sr_dir, expected_file)
            # Note: Not all files may be present depending on template structure
            # This is a soft check
            if os.path.exists(file_path):
                print(f"  ✓ Found {expected_file}")

        # Verify at least some files were created
        files_created = os.listdir(vllm_sr_dir)
        self.assertGreater(len(files_created), 0, ".vllm-sr/ directory is empty")
        print(f"  Created {len(files_created)} files in .vllm-sr/")

        self.print_test_result(True, ".vllm-sr/ directory created successfully")

    def test_init_preserves_existing_config_without_force(self):
        """Test that init does not overwrite existing config.yaml without --force."""
        self.print_test_header(
            "Init Preserves Existing Config",
            "Validates that init does not overwrite existing config without --force",
        )

        # Create existing config.yaml with unique marker
        config_path = os.path.join(self.test_dir, "config.yaml")
        original_content = (
            "# Existing config - DO NOT OVERWRITE\nversion: v0.1\ncustom_field: true\n"
        )
        with open(config_path, "w") as f:
            f.write(original_content)

        # Run init command without --force
        return_code, stdout, stderr = self.run_cli(["init"])

        # Command may succeed (skipping existing file) or fail - both are acceptable
        # The key is that the original file is NOT overwritten
        with open(config_path, "r") as f:
            content = f.read()

        # Verify original file content is preserved
        self.assertIn("Existing config", content, "Original config was overwritten")
        self.assertIn("custom_field", content, "Original config content was lost")

        print(f"  Return code: {return_code}")
        print(f"  Original content preserved: ✓")

        self.print_test_result(True, "init correctly preserved existing config")

    def test_init_force_overwrites_existing_config(self):
        """Test that init --force overwrites existing files."""
        self.print_test_header(
            "Init --force Overwrites",
            "Validates that 'vllm-sr init --force' overwrites existing config",
        )

        # Create existing config.yaml with custom content
        config_path = os.path.join(self.test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write("# This should be overwritten\nold_field: true\n")

        # Run init with --force
        return_code, stdout, stderr = self.run_cli(["init", "--force"])

        # Should succeed
        self.assertEqual(return_code, 0, f"init --force failed: {stderr}")

        # Verify file was overwritten with template content
        with open(config_path, "r") as f:
            content = f.read()
        self.assertNotIn("old_field", content, "Old config content still present")
        self.assertIn("listeners:", content, "New config missing listeners field")

        self.print_test_result(True, "init --force successfully overwrote config")

    def test_init_output_messages(self):
        """Test that init provides informative output messages."""
        self.print_test_header(
            "Init Output Messages",
            "Validates that init provides helpful feedback to user",
        )

        # Run init command
        return_code, stdout, stderr = self.run_cli(["init"])

        # Combine stdout and stderr for message checking
        output = stdout + stderr

        # Should contain success indicators
        success_indicators = ["config.yaml", ".vllm-sr", "vllm-sr serve"]
        found_indicators = []

        for indicator in success_indicators:
            if indicator.lower() in output.lower():
                found_indicators.append(indicator)
                print(f"  ✓ Found '{indicator}' in output")

        self.assertGreater(
            len(found_indicators),
            0,
            f"Output missing expected indicators. Got: {output[:200]}",
        )

        self.print_test_result(True, "init provides informative output")


if __name__ == "__main__":
    unittest.main()
