package handlers

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const (
	managedContainerSourceConfigPath = "/app/config.yaml"
	dashboardVenvPythonPath          = "/opt/vllm-sr-dashboard-venv/bin/python3"
)

func configuredRuntimeConfigPath(configPath string) string {
	if runtimePath := strings.TrimSpace(os.Getenv("VLLM_SR_RUNTIME_CONFIG_PATH")); runtimePath != "" {
		return filepath.Clean(runtimePath)
	}
	return filepath.Clean(configPath)
}

func syncRuntimeConfigForCurrentRuntime(configPath string) (string, error) {
	if isRunningInContainer() && isManagedContainerConfigPath(configPath) {
		return syncRuntimeConfigLocally(configPath)
	}

	if getDockerContainerStatus(managedRuntimeSyncContainerName()) == "running" {
		return syncRuntimeConfigInManagedContainer()
	}

	return filepath.Clean(configPath), nil
}

func syncRuntimeConfigLocally(configPath string) (string, error) {
	targetPath := configuredRuntimeConfigPath(configPath)
	if targetPath == filepath.Clean(configPath) && !hasRuntimeOverrideEnv() {
		return targetPath, nil
	}

	cliRoot := detectPythonCLIRoot()
	if cliRoot == "" {
		return "", fmt.Errorf("python CLI root not found for runtime config sync")
	}

	output, err := runRuntimeSyncPython(
		30*time.Second,
		cliRoot,
		configPath,
		filepath.Dir(configPath),
	)
	if err != nil {
		return "", fmt.Errorf(
			"failed to sync runtime config locally: %w (output: %s)",
			err,
			strings.TrimSpace(output),
		)
	}
	return parseRuntimeSyncOutput(output, targetPath), nil
}

func syncRuntimeConfigInManagedContainer() (string, error) {
	containerName := managedRuntimeSyncContainerName()
	output, err := execInManagedContainer(
		containerName,
		30*time.Second,
		"python3",
		"-c",
		buildRuntimeSyncPythonScript("/app", managedContainerSourceConfigPath),
	)
	if err != nil {
		return "", fmt.Errorf("failed to sync runtime config in %s: %w (output: %s)", containerName, err, strings.TrimSpace(output))
	}
	return parseRuntimeSyncOutput(output, configuredRuntimeConfigPath(managedContainerSourceConfigPath)), nil
}

func hasRuntimeOverrideEnv() bool {
	return strings.TrimSpace(os.Getenv("VLLM_SR_RUNTIME_CONFIG_PATH")) != "" ||
		strings.TrimSpace(os.Getenv("VLLM_SR_ALGORITHM_OVERRIDE")) != "" ||
		strings.TrimSpace(os.Getenv("VLLM_SR_PLATFORM")) != "" ||
		strings.TrimSpace(os.Getenv("DASHBOARD_PLATFORM")) != ""
}

func runRuntimeSyncPython(timeout time.Duration, cliRoot string, configPath string, workDir string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	pythonBinary, err := runtimeSyncPythonBinary()
	if err != nil {
		return "", err
	}

	// #nosec G204 -- pythonBinary is resolved through LookPath and constrained to python interpreters; the script is repository-owned.
	cmd := exec.CommandContext(
		ctx,
		pythonBinary,
		"-c",
		buildRuntimeSyncPythonScript(cliRoot, configPath),
	)
	cmd.Dir = workDir
	output, err := cmd.CombinedOutput()
	return string(output), err
}

func runtimeSyncPythonBinary() (string, error) {
	if configured := strings.TrimSpace(os.Getenv("VLLM_SR_PYTHON_BIN")); configured != "" {
		return resolveRuntimeSyncPythonBinary(configured)
	}

	if venv := strings.TrimSpace(os.Getenv("VIRTUAL_ENV")); venv != "" {
		for _, candidate := range []string{
			filepath.Join(venv, "bin", "python3"),
			filepath.Join(venv, "bin", "python"),
		} {
			resolved, err := resolveRuntimeSyncPythonBinary(candidate)
			if err == nil {
				return resolved, nil
			}
		}
	}

	for _, candidate := range []string{dashboardVenvPythonPath, "python", "python3"} {
		resolved, err := resolveRuntimeSyncPythonBinary(candidate)
		if err == nil {
			return resolved, nil
		}
	}

	return "", fmt.Errorf("python interpreter not found for runtime config sync")
}

func resolveRuntimeSyncPythonBinary(candidate string) (string, error) {
	resolved, err := exec.LookPath(candidate)
	if err != nil {
		return "", err
	}

	if !strings.HasPrefix(strings.ToLower(filepath.Base(resolved)), "python") {
		return "", fmt.Errorf("unsupported runtime sync python binary %q", resolved)
	}

	return resolved, nil
}

func buildRuntimeSyncPythonScript(cliRoot string, configPath string) string {
	return fmt.Sprintf(`
import os
import sys
from pathlib import Path

sys.path.insert(0, %q)

from cli.commands.runtime_support import sync_runtime_config

config_path = Path(%q)
algorithm = (os.getenv("VLLM_SR_ALGORITHM_OVERRIDE") or "").strip() or None
platform = (os.getenv("VLLM_SR_PLATFORM") or os.getenv("DASHBOARD_PLATFORM") or "").strip() or None
effective = sync_runtime_config(config_path, algorithm=algorithm, platform=platform)
print(str(effective))
`, cliRoot, configPath)
}

func parseRuntimeSyncOutput(output string, fallback string) string {
	trimmed := strings.TrimSpace(output)
	if trimmed == "" {
		return fallback
	}

	lines := strings.Split(trimmed, "\n")
	return strings.TrimSpace(lines[len(lines)-1])
}
