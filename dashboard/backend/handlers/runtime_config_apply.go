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
		if !managedServiceCanBeControlledLocally("envoy") && getDockerContainerStatus(managedContainerNameForService("envoy")) == "running" {
			return regenerateAndReloadManagedSplitEnvoyLocally(effectiveConfigPath)
		}
		return regenerateAndReloadEnvoyLocally(effectiveConfigPath)
	}

	if getDockerContainerStatus(managedContainerNameForService("envoy")) == "running" {
		return propagateConfigToManagedContainer()
	}

	return nil
}

func isManagedContainerConfigPath(configPath string) bool {
	return filepath.Clean(configPath) == "/app/config.yaml"
}

func regenerateAndReloadEnvoyLocally(configPath string) error {
	if _, err := exec.LookPath("supervisorctl"); err != nil {
		log.Printf("Config propagation: supervisorctl not available, skipping managed Envoy reload")
		return nil
	}

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

	if err := restartOrStartSupervisorService("envoy", 20*time.Second); err != nil {
		return fmt.Errorf("failed to restart Envoy: %w", err)
	}

	return nil
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

	switch args[0] {
	case "supervisorctl":
		return validateManagedContainerSupervisorArgs(args)
	default:
		return fmt.Errorf("unsupported managed container command: %s", args[0])
	}
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

func validateManagedContainerSupervisorArgs(args []string) error {
	if len(args) != 3 {
		return fmt.Errorf("unsupported supervisorctl invocation in managed container")
	}

	switch args[1] {
	case "restart", "start", "status":
	default:
		return fmt.Errorf("unsupported supervisorctl action: %s", args[1])
	}

	switch args[2] {
	case "router", "envoy", "dashboard":
		return nil
	default:
		return fmt.Errorf("unsupported supervisorctl service: %s", args[2])
	}
}

func restartOrStartSupervisorService(service string, timeout time.Duration) error {
	cmd := exec.Command("supervisorctl", "restart", service)
	if output, err := cmd.CombinedOutput(); err != nil {
		startCmd := exec.Command("supervisorctl", "start", service)
		if startOutput, startErr := startCmd.CombinedOutput(); startErr != nil {
			return fmt.Errorf("%s restart failed: %s / start failed: %s", service, strings.TrimSpace(string(output)), strings.TrimSpace(string(startOutput)))
		}
	}

	return waitForSupervisorService(service, timeout)
}

func restartOrStartManagedContainerService(service string, timeout time.Duration) error {
	containerName := managedContainerNameForService(service)
	if output, err := execInManagedContainer(containerName, 15*time.Second, "supervisorctl", "restart", service); err != nil {
		startOutput, startErr := execInManagedContainer(containerName, 15*time.Second, "supervisorctl", "start", service)
		if startErr != nil {
			return fmt.Errorf("%s restart failed: %s / start failed: %s", service, strings.TrimSpace(output), strings.TrimSpace(startOutput))
		}
	}

	return waitForManagedContainerService(containerName, service, timeout)
}

func restartManagedService(service string, timeout time.Duration) error {
	if managedServiceCanBeControlledLocally(service) {
		return restartOrStartSupervisorService(service, timeout)
	}
	if managedServiceUsesContainerLifecycle(service) {
		return restartOrStartManagedSplitContainerService(service, timeout)
	}
	return restartOrStartManagedContainerService(service, timeout)
}

func waitForSupervisorService(service string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	lastStatus := ""

	for time.Now().Before(deadline) {
		output, err := exec.Command("supervisorctl", "status", service).CombinedOutput()
		lastStatus = strings.TrimSpace(string(output))
		if err == nil && strings.Contains(lastStatus, "RUNNING") {
			return nil
		}
		if strings.Contains(lastStatus, "FATAL") || strings.Contains(lastStatus, "EXITED") || strings.Contains(lastStatus, "BACKOFF") {
			return fmt.Errorf("%s failed to start: %s", service, lastStatus)
		}
		time.Sleep(500 * time.Millisecond)
	}

	return fmt.Errorf("timed out waiting for %s to become RUNNING (last status: %s)", service, lastStatus)
}

func waitForManagedContainerService(containerName string, service string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	lastStatus := ""

	for time.Now().Before(deadline) {
		output, err := execInManagedContainer(containerName, 10*time.Second, "supervisorctl", "status", service)
		lastStatus = strings.TrimSpace(output)
		if err == nil && strings.Contains(lastStatus, "RUNNING") {
			return nil
		}
		if strings.Contains(lastStatus, "FATAL") || strings.Contains(lastStatus, "EXITED") || strings.Contains(lastStatus, "BACKOFF") {
			return fmt.Errorf("%s failed to start: %s", service, lastStatus)
		}
		time.Sleep(500 * time.Millisecond)
	}

	return fmt.Errorf("timed out waiting for %s in %s to become RUNNING (last status: %s)", service, containerName, lastStatus)
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
