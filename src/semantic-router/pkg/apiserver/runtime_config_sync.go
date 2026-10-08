//go:build !windows

package apiserver

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
)

const (
	runtimeConfigPathEnv = configsnapshot.RuntimeConfigPathEnv
	sourceConfigPathEnv  = configsnapshot.SourceConfigPathEnv
	runtimeAlgorithmEnv  = "VLLM_SR_ALGORITHM_OVERRIDE"
	runtimePlatformEnv   = "VLLM_SR_PLATFORM"
	dashboardPlatformEnv = "DASHBOARD_PLATFORM"
	configBaseDirEnv     = config.ConfigBaseDirEnv
	configBackupDirEnv   = configsnapshot.HistoryDirEnv
	defaultPythonCLIPath = "/app"
)

// configBackupDir is the configuration history directory, which keeps the
// backups this API writes beside the snapshots the Router records.
func configBackupDir(sourceConfigPath string) string {
	return configsnapshot.HistoryDir(sourceConfigPath)
}

type configPersistencePaths struct {
	activePath  string
	sourcePath  string
	runtimePath string
}

var runtimeConfigSyncRunner = syncRuntimeConfigForCurrentRuntime

func resolveConfigPersistencePaths(activeConfigPath string) configPersistencePaths {
	persistence := configsnapshot.ResolvePersistence(activeConfigPath)
	return configPersistencePaths{
		activePath:  filepath.Clean(activeConfigPath),
		sourcePath:  persistence.Source,
		runtimePath: persistence.Runtime,
	}
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
		if hasRuntimeSyncModule(configured) {
			return configured
		}
		return ""
	}
	if hasRuntimeSyncModule(defaultPythonCLIPath) {
		return defaultPythonCLIPath
	}
	return ""
}

func hasRuntimeSyncModule(cliRoot string) bool {
	modulePath := filepath.Join(cliRoot, "cli", "commands", "runtime_support.py")
	info, err := os.Stat(modulePath)
	return err == nil && !info.IsDir()
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
