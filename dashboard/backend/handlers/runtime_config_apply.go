package handlers

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const (
	defaultEnvoyConfigPath      = "/etc/envoy/envoy.yaml"
	defaultSplitEnvoyConfigPath = "/app/.vllm-sr/envoy.yaml"
)

type runtimeConfigApplyError struct {
	applyErr   error
	restoreErr error
}

func (e *runtimeConfigApplyError) Error() string {
	if e == nil {
		return ""
	}
	if e.restoreErr != nil {
		return fmt.Sprintf("%v (restore failed: %v)", e.applyErr, e.restoreErr)
	}
	return e.applyErr.Error()
}

func (e *runtimeConfigApplyError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.applyErr
}

func applyWrittenConfig(configPath string, configDir string, previousData []byte, restoreOnFailure bool) error {
	if err := propagateConfigToRuntime(configPath, configDir); err != nil {
		if !restoreOnFailure || len(previousData) == 0 {
			return err
		}
		if restoreErr := restorePreviousRuntimeConfig(configPath, configDir, previousData); restoreErr != nil {
			return &runtimeConfigApplyError{applyErr: err, restoreErr: restoreErr}
		}
		return &runtimeConfigApplyError{applyErr: err}
	}

	return nil
}

func formatRuntimeApplyError(prefix string, err error) string {
	var applyErr *runtimeConfigApplyError
	if errors.As(err, &applyErr) {
		if applyErr.restoreErr != nil {
			return fmt.Sprintf("%s: %v. Failed to restore previous config: %v", prefix, applyErr.applyErr, applyErr.restoreErr)
		}
		return fmt.Sprintf("%s: %v. Previous config restored.", prefix, applyErr.applyErr)
	}

	return fmt.Sprintf("%s: %v", prefix, err)
}

func writeConfigAtomically(configPath string, yamlData []byte) error {
	tmpConfigFile := configPath + ".tmp"
	if err := os.WriteFile(tmpConfigFile, yamlData, 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmpConfigFile, configPath); err != nil {
		if writeErr := os.WriteFile(configPath, yamlData, 0o644); writeErr != nil {
			return writeErr
		}
	}
	return nil
}

func restorePreviousRuntimeConfig(configPath string, configDir string, previousData []byte) error {
	if len(previousData) == 0 {
		return nil
	}
	if err := writeConfigAtomically(configPath, previousData); err != nil {
		return err
	}
	return propagateConfigToRuntime(configPath, configDir)
}

func propagateConfigToRuntime(configPath string, configDir string) error {
	effectiveConfigPath, err := syncRuntimeConfigForCurrentRuntime(configPath)
	if err != nil {
		return fmt.Errorf("failed to sync runtime config: %w", err)
	}

	if isRunningInContainer() && isManagedContainerConfigPath(configPath) {
		if getDockerContainerStatus(managedContainerNameForService("envoy")) == "running" {
			return regenerateAndReloadManagedSplitEnvoyLocally(effectiveConfigPath)
		}
		return nil
	}

	if getDockerContainerStatus(managedContainerNameForService("envoy")) == "running" {
		return propagateConfigToManagedContainer()
	}

	return nil
}

func isManagedContainerConfigPath(configPath string) bool {
	return filepath.Clean(configPath) == "/app/config.yaml"
}

func regenerateAndReloadManagedSplitEnvoyLocally(configPath string) error {
	envoyConfigPath := detectEnvoyConfigPath()
	if envoyConfigPath == "" {
		log.Printf("Config propagation: Envoy config path not found, skipping managed Envoy reload")
		return nil
	}

	output, err := generateEnvoyConfigWithPython(configPath, envoyConfigPath)
	if err != nil {
		return fmt.Errorf("failed to regenerate Envoy config: %w (output: %s)", err, strings.TrimSpace(output))
	}
	log.Printf("Config propagation: %s", strings.TrimSpace(output))

	if err := restartManagedService("envoy", 20*time.Second); err != nil {
		return fmt.Errorf("failed to restart Envoy in %s: %w", managedContainerNameForService("envoy"), err)
	}

	return nil
}

func refreshManagedSplitEnvoyConfig(configPath string) error {
	if !managedRuntimeUsesSplitContainers() {
		return nil
	}
	if getDockerContainerStatus(managedContainerNameForService("envoy")) == "not found" {
		return nil
	}

	envoyConfigPath := splitEnvoyConfigPathForRuntimeConfig(configPath)
	if envoyConfigPath == "" {
		log.Printf("Config propagation: split Envoy config path not found, skipping setup-time refresh")
		return nil
	}

	output, err := generateEnvoyConfigWithPython(configPath, envoyConfigPath)
	if err != nil {
		return fmt.Errorf(
			"failed to regenerate split Envoy config: %w (output: %s)",
			err,
			strings.TrimSpace(output),
		)
	}

	if trimmed := strings.TrimSpace(output); trimmed != "" {
		log.Printf("Config propagation: %s", trimmed)
	}

	return nil
}

func propagateConfigToManagedContainer() error {
	effectiveConfigPath, err := syncRuntimeConfigInManagedContainer()
	if err != nil {
		return err
	}

	return regenerateAndReloadEnvoyInManagedContainer(effectiveConfigPath)
}

func regenerateAndReloadEnvoyInManagedContainer(configPath string) error {
	if output, err := generateEnvoyConfigInManagedContainer(configPath); err != nil {
		return fmt.Errorf("failed to regenerate Envoy config in %s: %w (output: %s)", managedContainerNameForService("envoy"), err, strings.TrimSpace(output))
	} else {
		log.Printf("Config propagation: %s", strings.TrimSpace(output))
	}

	if err := restartManagedService("envoy", 20*time.Second); err != nil {
		return fmt.Errorf("failed to restart Envoy in %s: %w", managedContainerNameForService("envoy"), err)
	}

	return nil
}

func generateEnvoyConfigWithPython(configPath string, outputPath string) (string, error) {
	cliRoot := detectPythonCLIRoot()
	if cliRoot == "" {
		return "SKIP: Python CLI not available, skipping Envoy config regeneration", nil
	}

	pythonBinary, err := runtimeSyncPythonBinary()
	if err != nil {
		return "", fmt.Errorf("python interpreter not found for Envoy config regeneration: %w", err)
	}

	pythonScript := fmt.Sprintf(`
import sys
sys.path.insert(0, %q)

from cli.config_generator import generate_envoy_config_from_user_config
from cli.parser import parse_user_config

user_config = parse_user_config(%q)
generate_envoy_config_from_user_config(user_config, %q)
print("Regenerated Envoy config: %s")
`, cliRoot, configPath, outputPath, outputPath)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, pythonBinary, "-c", pythonScript)
	cmd.Dir = filepath.Dir(configPath)
	output, err := cmd.CombinedOutput()
	return string(output), err
}

func detectEnvoyConfigPath() string {
	candidates := []string{}
	if envPath := strings.TrimSpace(os.Getenv("VLLM_SR_ENVOY_CONFIG_PATH")); envPath != "" {
		candidates = append(candidates, envPath)
	}
	if isRunningInContainer() && managedRuntimeUsesSplitContainers() {
		candidates = append(candidates, defaultSplitEnvoyConfigPath)
	}
	candidates = append(candidates, defaultEnvoyConfigPath)

	for _, candidate := range candidates {
		if candidate == "" {
			continue
		}
		if info, err := os.Stat(filepath.Dir(candidate)); err == nil && info.IsDir() {
			return candidate
		}
	}

	return ""
}

func splitEnvoyConfigPathForRuntimeConfig(configPath string) string {
	configDir := filepath.Dir(filepath.Clean(configPath))
	if filepath.Base(configDir) == ".vllm-sr" {
		return filepath.Join(configDir, "envoy.yaml")
	}
	return detectEnvoyConfigPath()
}

func generateEnvoyConfigInManagedContainer(configPath string) (string, error) {
	containerName := managedContainerNameForService("envoy")
	outputPath := defaultEnvoyConfigPath
	pythonBinary := "python3"
	if managedRuntimeUsesSplitContainers() {
		containerName = managedRuntimeSyncContainerName()
		outputPath = defaultSplitEnvoyConfigPath
		pythonBinary = dashboardVenvPythonPath
	}
	pythonScript := fmt.Sprintf(`
from cli.config_generator import generate_envoy_config_from_user_config
from cli.parser import parse_user_config

user_config = parse_user_config(%q)
generate_envoy_config_from_user_config(user_config, %q)
print("Regenerated Envoy config: %s")
`, configPath, outputPath, outputPath)

	return execInManagedContainer(containerName, 30*time.Second, pythonBinary, "-c", pythonScript)
}

func execInManagedContainer(containerName string, timeout time.Duration, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	if err := validateManagedContainerExecArgs(args); err != nil {
		return "", err
	}

	commandArgs := append([]string{"exec", containerName}, args...)
	// #nosec G204 -- commandArgs are validated against a strict allowlist above and the container name is constant.
	cmd := exec.CommandContext(ctx, "docker", commandArgs...)
	output, err := cmd.CombinedOutput()
	return string(output), err
}

func validateManagedContainerExecArgs(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("managed container command is required")
	}

	if isPythonCommand(args[0]) {
		return validateManagedContainerPythonArgs(args)
	}

	return fmt.Errorf("unsupported managed container command: %s", args[0])
}

func validateManagedContainerPythonArgs(args []string) error {
	if len(args) == 3 && args[1] == "-c" {
		return nil
	}

	return fmt.Errorf("unsupported python3 invocation in managed container")
}

func isPythonCommand(command string) bool {
	base := strings.ToLower(filepath.Base(strings.TrimSpace(command)))
	return base != "" && strings.HasPrefix(base, "python")
}

func restartManagedService(service string, timeout time.Duration) error {
	if !managedServiceUsesContainerLifecycle(service) {
		return fmt.Errorf("unsupported managed service restart: %s", service)
	}
	return restartOrStartManagedSplitContainerService(service, timeout)
}

func detectPythonCLIRoot() string {
	candidates := []string{}
	if envPath := strings.TrimSpace(os.Getenv("VLLM_SR_CLI_PATH")); envPath != "" {
		candidates = append(candidates, envPath)
	}
	candidates = append(candidates, "/app")

	if wd, err := os.Getwd(); err == nil {
		candidates = append(
			candidates,
			filepath.Clean(filepath.Join(wd, "..", "..", "..", "src", "vllm-sr")),
			filepath.Clean(filepath.Join(wd, "..", "..", "src", "vllm-sr")),
			filepath.Clean(filepath.Join(wd, "src", "vllm-sr")),
		)
	}
	if _, thisFile, _, ok := runtime.Caller(0); ok {
		candidates = append(
			candidates,
			filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", "..", "..", "src", "vllm-sr")),
		)
	}

	seen := map[string]bool{}
	for _, candidate := range candidates {
		if candidate == "" || seen[candidate] {
			continue
		}
		seen[candidate] = true
		if info, err := os.Stat(candidate); err == nil && info.IsDir() {
			return candidate
		}
	}

	return ""
}
