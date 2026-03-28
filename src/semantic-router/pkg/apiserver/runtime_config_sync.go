//go:build !windows && cgo

package apiserver

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
	runtimeConfigPathEnv = "VLLM_SR_RUNTIME_CONFIG_PATH"
	sourceConfigPathEnv  = "VLLM_SR_SOURCE_CONFIG_PATH"
	runtimeAlgorithmEnv  = "VLLM_SR_ALGORITHM_OVERRIDE"
	runtimePlatformEnv   = "VLLM_SR_PLATFORM"
	dashboardPlatformEnv = "DASHBOARD_PLATFORM"
	defaultPythonCLIPath = "/app"
)

type configPersistencePaths struct {
	activePath  string
	sourcePath  string
	runtimePath string
}

var runtimeConfigSyncRunner = syncRuntimeConfigForCurrentRuntime

func resolveConfigPersistencePaths(activeConfigPath string) configPersistencePaths {
	activePath := filepath.Clean(activeConfigPath)
	sourcePath := strings.TrimSpace(os.Getenv(sourceConfigPathEnv))
	if sourcePath == "" {
		sourcePath = deriveSourceConfigPath(activePath)
	}
	if sourcePath == "" {
		sourcePath = activePath
	}
	sourcePath = filepath.Clean(sourcePath)

	runtimePath := strings.TrimSpace(os.Getenv(runtimeConfigPathEnv))
	if runtimePath == "" {
		runtimePath = activePath
	}
	runtimePath = filepath.Clean(runtimePath)

	return configPersistencePaths{
		activePath:  activePath,
		sourcePath:  sourcePath,
		runtimePath: runtimePath,
	}
}

func deriveSourceConfigPath(activeConfigPath string) string {
	activePath := filepath.Clean(activeConfigPath)
	if filepath.Base(activePath) == "config.yaml" {
		return activePath
	}

	parent := filepath.Base(filepath.Dir(activePath))
	base := filepath.Base(activePath)
	if parent == ".vllm-sr" && strings.HasPrefix(base, "runtime-config") && strings.HasSuffix(base, ".yaml") {
		return filepath.Join(filepath.Dir(filepath.Dir(activePath)), "config.yaml")
	}

	return ""
}

func (p configPersistencePaths) usesRuntimeOverride() bool {
	return p.sourcePath != "" && p.runtimePath != "" && p.sourcePath != p.runtimePath
}

func configuredRuntimeConfigPath(sourceConfigPath string) string {
	if runtimePath := strings.TrimSpace(os.Getenv(runtimeConfigPathEnv)); runtimePath != "" {
		return filepath.Clean(runtimePath)
	}
	return filepath.Clean(sourceConfigPath)
}

func hasRuntimeOverrideEnv() bool {
	return strings.TrimSpace(os.Getenv(runtimeConfigPathEnv)) != "" ||
		strings.TrimSpace(os.Getenv(runtimeAlgorithmEnv)) != "" ||
		strings.TrimSpace(os.Getenv(runtimePlatformEnv)) != "" ||
		strings.TrimSpace(os.Getenv(dashboardPlatformEnv)) != ""
}

func syncRuntimeConfigForCurrentRuntime(sourceConfigPath string) (string, error) {
	targetPath := configuredRuntimeConfigPath(sourceConfigPath)
	if targetPath == filepath.Clean(sourceConfigPath) && !hasRuntimeOverrideEnv() {
		return targetPath, nil
	}

	cliRoot := detectPythonCLIRoot()
	if cliRoot == "" {
		return "", fmt.Errorf("python CLI root not found for runtime config sync")
	}

	output, err := runRuntimeSyncPython(
		30*time.Second,
		cliRoot,
		sourceConfigPath,
		filepath.Dir(sourceConfigPath),
	)
	if err != nil {
		return "", fmt.Errorf(
			"failed to sync runtime config: %w (output: %s)",
			err,
			strings.TrimSpace(output),
		)
	}
	return parseRuntimeSyncOutput(output, targetPath), nil
}

func detectPythonCLIRoot() string {
	if configured := strings.TrimSpace(os.Getenv("VLLM_SR_CLI_PATH")); configured != "" {
		return configured
	}
	if info, err := os.Stat(defaultPythonCLIPath); err == nil && info.IsDir() {
		return defaultPythonCLIPath
	}
	return ""
}

func runRuntimeSyncPython(timeout time.Duration, cliRoot string, configPath string, workDir string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	// #nosec G204 -- python3 is fixed and the script is repository-owned runtime sync logic; inputs are local config paths.
	cmd := exec.CommandContext(
		ctx,
		"python3",
		"-c",
		buildRuntimeSyncPythonScript(cliRoot, configPath),
	)
	cmd.Dir = workDir
	output, err := cmd.CombinedOutput()
	return string(output), err
}

func buildRuntimeSyncPythonScript(cliRoot string, configPath string) string {
	return fmt.Sprintf(`
import os
import sys
from pathlib import Path

sys.path.insert(0, %q)

from cli.commands.runtime_support import sync_runtime_config

config_path = Path(%q)
algorithm = (os.getenv(%q) or "").strip() or None
platform = (os.getenv(%q) or os.getenv(%q) or "").strip() or None
effective = sync_runtime_config(config_path, algorithm=algorithm, platform=platform)
print(str(effective))
`, cliRoot, configPath, runtimeAlgorithmEnv, runtimePlatformEnv, dashboardPlatformEnv)
}

func parseRuntimeSyncOutput(output string, fallback string) string {
	trimmed := strings.TrimSpace(output)
	if trimmed == "" {
		return fallback
	}

	lines := strings.Split(trimmed, "\n")
	return strings.TrimSpace(lines[len(lines)-1])
}
