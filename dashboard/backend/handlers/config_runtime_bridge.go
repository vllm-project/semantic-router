package handlers

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const defaultEnvoyConfigPath = "/etc/envoy/envoy.yaml"

var (
	runtimeDockerContainerStatus = getDockerContainerStatus
	runtimeIsRunningInContainer  = isRunningInContainer
)

// regenerateRouterConfigSync calls the Python CLI to regenerate .vllm-sr/router-config.yaml
// from the user-facing config.yaml. This bridges the gap between the Dashboard
// (which edits the nested Python CLI format) and the Router (which reads the flat format).
// The Router's fsnotify watcher will detect the change and hot-reload automatically.
func regenerateRouterConfigSync(configPath string, configDir string) error {
	outputDir := filepath.Join(configDir, ".vllm-sr")
	if _, err := os.Stat(outputDir); os.IsNotExist(err) {
		log.Printf("Config propagation: .vllm-sr directory not found at %s, skipping router config regeneration (dev mode?)", outputDir)
		return nil
	}

	output, err := generateRouterConfigWithPython(configPath, outputDir)
	if err != nil {
		return fmt.Errorf("failed to regenerate router config: %w (output: %s)", err, strings.TrimSpace(output))
	}

	log.Printf("Config propagation: %s", strings.TrimSpace(output))
	return nil
}

func generateRouterConfigWithPython(configPath string, outputDir string) (string, error) {
	cliRoot := detectPythonCLIRoot()
	if cliRoot == "" {
		return "SKIP: Python CLI not available, skipping router config regeneration", nil
	}
	pythonBin := detectPythonCLIExecutable()

	pythonScript := fmt.Sprintf(`
import sys
sys.path.insert(0, %q)
try:
    from cli.commands.serve import generate_router_config
    result = generate_router_config(%q, %q, force=True)
    print(f"Regenerated router config: {result}")
except ImportError:
    print("SKIP: Python CLI not available, skipping router config regeneration")
except Exception as e:
    print(f"ERROR: Failed to regenerate router config: {e}", file=sys.stderr)
    sys.exit(1)
`, cliRoot, configPath, outputDir)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, pythonBin, "-c", pythonScript)
	cmd.Dir = filepath.Dir(configPath)
	output, err := cmd.CombinedOutput()
	return string(output), err
}

func propagateConfigToRuntime(configPath string, configDir string) error {
	if runtimeIsRunningInContainer() && isManagedContainerConfigPath(configPath) {
		if err := regenerateRouterConfigSync(configPath, configDir); err != nil {
			return err
		}
		return regenerateAndReloadEnvoyLocally(configPath)
	}

	if runtimeDockerContainerStatus(vllmSrContainerName) == "running" {
		return propagateConfigToManagedContainer()
	}

	return regenerateRouterConfigSync(configPath, configDir)
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

func propagateConfigToManagedContainer() error {
	if output, err := generateRouterConfigInManagedContainer(); err != nil {
		return fmt.Errorf("failed to regenerate router config in %s: %w (output: %s)", vllmSrContainerName, err, strings.TrimSpace(output))
	} else {
		log.Printf("Config propagation: %s", strings.TrimSpace(output))
	}

	if output, err := generateEnvoyConfigInManagedContainer(); err != nil {
		return fmt.Errorf("failed to regenerate Envoy config in %s: %w (output: %s)", vllmSrContainerName, err, strings.TrimSpace(output))
	} else {
		log.Printf("Config propagation: %s", strings.TrimSpace(output))
	}

	if err := restartOrStartManagedContainerService("envoy", 20*time.Second); err != nil {
		return fmt.Errorf("failed to restart Envoy in %s: %w", vllmSrContainerName, err)
	}

	return nil
}

func generateEnvoyConfigWithPython(configPath string, outputPath string) (string, error) {
	cliRoot := detectPythonCLIRoot()
	if cliRoot == "" {
		return "SKIP: Python CLI not available, skipping Envoy config regeneration", nil
	}
	pythonBin := detectPythonCLIExecutable()

	pythonScript := fmt.Sprintf(`
import sys
sys.path.insert(0, %q)
try:
    from cli.config_generator import generate_envoy_config_from_user_config
    from cli.parser import parse_user_config
    user_config = parse_user_config(%q)
    generate_envoy_config_from_user_config(user_config, %q)
    print("Regenerated Envoy config: %s")
except ImportError:
    print("SKIP: Python CLI not available, skipping Envoy config regeneration")
except Exception as e:
    print(f"ERROR: Failed to regenerate Envoy config: {e}", file=sys.stderr)
    sys.exit(1)
`, cliRoot, configPath, outputPath, outputPath)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, pythonBin, "-c", pythonScript)
	cmd.Dir = filepath.Dir(configPath)
	output, err := cmd.CombinedOutput()
	return string(output), err
}

func detectEnvoyConfigPath() string {
	candidates := []string{}
	if envPath := strings.TrimSpace(os.Getenv("VLLM_SR_ENVOY_CONFIG_PATH")); envPath != "" {
		candidates = append(candidates, envPath)
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

func generateRouterConfigInManagedContainer() (string, error) {
	pythonScript := `
from cli.commands.serve import generate_router_config
result = generate_router_config("/app/config.yaml", "/app/.vllm-sr", force=True)
print(f"Regenerated router config: {result}")
`
	return execInManagedContainer(30*time.Second, "python3", "-c", pythonScript)
}

func generateEnvoyConfigInManagedContainer() (string, error) {
	return execInManagedContainer(30*time.Second, "python3", "-m", "cli.config_generator", "/app/config.yaml", defaultEnvoyConfigPath)
}

func execInManagedContainer(timeout time.Duration, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	if err := validateManagedContainerExecArgs(args); err != nil {
		return "", err
	}

	commandArgs := append([]string{"exec", vllmSrContainerName}, args...)
	// #nosec G204 -- commandArgs are validated against a strict allowlist above and the container name is constant.
	cmd := exec.CommandContext(ctx, "docker", commandArgs...)
	output, err := cmd.CombinedOutput()
	return string(output), err
}

func validateManagedContainerExecArgs(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("managed container command is required")
	}

	switch args[0] {
	case "python3":
		return validateManagedContainerPythonArgs(args)
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

	if len(args) == 5 &&
		args[1] == "-m" &&
		args[2] == "cli.config_generator" &&
		args[3] == "/app/config.yaml" &&
		args[4] == defaultEnvoyConfigPath {
		return nil
	}

	return fmt.Errorf("unsupported python3 invocation in managed container")
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
	if output, err := execInManagedContainer(15*time.Second, "supervisorctl", "restart", service); err != nil {
		startOutput, startErr := execInManagedContainer(15*time.Second, "supervisorctl", "start", service)
		if startErr != nil {
			return fmt.Errorf("%s restart failed: %s / start failed: %s", service, strings.TrimSpace(output), strings.TrimSpace(startOutput))
		}
	}

	return waitForManagedContainerService(service, timeout)
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

func waitForManagedContainerService(service string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	lastStatus := ""

	for time.Now().Before(deadline) {
		output, err := execInManagedContainer(10*time.Second, "supervisorctl", "status", service)
		lastStatus = strings.TrimSpace(output)
		if err == nil && strings.Contains(lastStatus, "RUNNING") {
			return nil
		}
		if strings.Contains(lastStatus, "FATAL") || strings.Contains(lastStatus, "EXITED") || strings.Contains(lastStatus, "BACKOFF") {
			return fmt.Errorf("%s failed to start: %s", service, lastStatus)
		}
		time.Sleep(500 * time.Millisecond)
	}

	return fmt.Errorf("timed out waiting for %s in %s to become RUNNING (last status: %s)", service, vllmSrContainerName, lastStatus)
}
