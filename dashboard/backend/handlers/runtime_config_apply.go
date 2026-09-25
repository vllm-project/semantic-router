package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/k8s/configwriter"
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

// atomicRename is os.Rename by default; tests override it to simulate a rename failure.
var atomicRename = os.Rename

const configMapWriteTimeout = 10 * time.Second

var errConfigRolloutRequired = errors.New("the saved ConfigMap differs from this pod's mounted config; roll out the deployment before another mutation")

func configActivationDeferred() bool {
	_, ok := configwriter.ConfigMapTargetFromEnv()
	return ok
}

func writeDeferredConfigResponse(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	_ = json.NewEncoder(w).Encode(map[string]string{
		"status":  "persisted",
		"message": "Configuration saved to the Kubernetes ConfigMap. Roll out the Router and Envoy deployments to activate it.",
	})
}

func writeConfigPersistenceError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, errConfigRolloutRequired):
		http.Error(w, err.Error(), http.StatusConflict)
	case errors.Is(err, configwriter.ErrConfigMapChanged):
		http.Error(w, "the ConfigMap changed during this request; reload the configuration and retry", http.StatusConflict)
	case errors.Is(err, configwriter.ErrConfigMapControllerOwned):
		http.Error(w, "this ConfigMap is controller-owned; edit its owning resource", http.StatusForbidden)
	default:
		http.Error(w, "failed to persist configuration", http.StatusInternalServerError)
		log.Printf("Configuration persistence failed: %v", err)
	}
}

var (
	configMapWriterMu       sync.Mutex
	configMapWriterResult   *configwriter.ConfigMapWriter
	configMapWriterErr      error
	configMapWriterResolved bool
	// newInClusterConfigMapWriter is a seam for tests; production always uses
	// configwriter.NewInClusterConfigMapWriter.
	newInClusterConfigMapWriter = configwriter.NewInClusterConfigMapWriter
)

// resolvedConfigMapWriter builds the in-cluster ConfigMap client once and
// reuses it. Every shipped Kubernetes deployment mounts the config file
// read-only (issue #3688); this is that mount's write path.
func resolvedConfigMapWriter() (*configwriter.ConfigMapWriter, error) {
	configMapWriterMu.Lock()
	defer configMapWriterMu.Unlock()
	if !configMapWriterResolved {
		configMapWriterResult, configMapWriterErr = newInClusterConfigMapWriter()
		configMapWriterResolved = true
	}
	return configMapWriterResult, configMapWriterErr
}

// readPersistedDashboardConfig reads the saved document, which may be newer
// than the ConfigMap subPath mount used by the still-running Router.
func readPersistedDashboardConfig(configPath string) ([]byte, error) {
	target, ok := configwriter.ConfigMapTargetFromEnv()
	if !ok {
		return os.ReadFile(configPath)
	}
	writer, err := resolvedConfigMapWriter()
	if err != nil {
		return nil, fmt.Errorf("resolve config ConfigMap client: %w", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), configMapWriteTimeout)
	defer cancel()
	data, found, err := writer.Read(ctx, target)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, os.ErrNotExist
	}
	return data, nil
}

// writeConfigAtomically persists a canonical config document. On a
// Kubernetes deployment that has declared a ConfigMap write target (see
// configwriter.ConfigMapTargetFromEnv), it writes there via the Kubernetes API instead
// of the local file, since that file is a read-only ConfigMap mount on every
// shipped manifest (issue #3688). Every other deployment (local CLI, VM,
// plain Docker) keeps writing the local file exactly as before.
func writeConfigAtomically(configPath string, yamlData []byte) error {
	if target, ok := configwriter.ConfigMapTargetFromEnv(); ok {
		mounted, err := checkConfigMapMutationFresh(configPath)
		if err != nil {
			return err
		}
		writer, err := resolvedConfigMapWriter()
		if err != nil {
			return fmt.Errorf("config write target is declared but no Kubernetes client is available: %w", err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), configMapWriteTimeout)
		defer cancel()
		return writer.WriteIfUnchanged(ctx, target, mounted, yamlData)
	}

	tmpConfigFile := configPath + ".tmp"
	tmpFile, err := os.OpenFile(tmpConfigFile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	if _, err := tmpFile.Write(yamlData); err != nil {
		tmpFile.Close()
		os.Remove(tmpConfigFile)
		return err
	}
	if err := tmpFile.Sync(); err != nil {
		tmpFile.Close()
		os.Remove(tmpConfigFile)
		return err
	}
	if err := tmpFile.Close(); err != nil {
		os.Remove(tmpConfigFile)
		return err
	}
	if err := atomicRename(tmpConfigFile, configPath); err != nil {
		os.Remove(tmpConfigFile)
		return err
	}
	// Best-effort: fsync the directory too so the rename is durable, not just the bytes.
	if dir, derr := os.Open(filepath.Dir(configPath)); derr == nil {
		_ = dir.Sync()
		_ = dir.Close()
	}
	return nil
}

// checkConfigMapMutationFresh must run before creating a backup or DSL archive
// as well as at the final write. A stale subPath mount otherwise lets a
// rejected second request replace a valid backup with newer ConfigMap bytes.
// The final WriteIfUnchanged still protects the gap after this read.
func checkConfigMapMutationFresh(configPath string) ([]byte, error) {
	target, ok := configwriter.ConfigMapTargetFromEnv()
	if !ok {
		return nil, nil
	}
	writer, err := resolvedConfigMapWriter()
	if err != nil {
		return nil, fmt.Errorf("resolve config ConfigMap client: %w", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), configMapWriteTimeout)
	defer cancel()
	mounted, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("read mounted config before ConfigMap update: %w", err)
	}
	persisted, found, err := writer.Read(ctx, target)
	if err != nil {
		return nil, fmt.Errorf("read config ConfigMap before update: %w", err)
	}
	if !found {
		return nil, os.ErrNotExist
	}
	if !bytes.Equal(mounted, persisted) {
		return nil, errConfigRolloutRequired
	}
	return mounted, nil
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
	if _, ok := configwriter.ConfigMapTargetFromEnv(); ok {
		// The ConfigMap subPath mount cannot update this process. The Router and
		// Envoy pick up the saved config only after a deployment rollout.
		return nil
	}
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
	cleaned := filepath.Clean(configPath)
	configured := configuredRuntimeConfigPath(legacyManagedContainerConfigPath)
	return cleaned == configured || cleaned == legacyManagedContainerConfigPath
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
