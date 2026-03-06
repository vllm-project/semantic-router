package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ConfigHandler reads and serves the config as JSON from the local config file.
func ConfigHandler(configPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")

		data, err := os.ReadFile(configPath)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to read config: %v", err), http.StatusInternalServerError)
			return
		}

		var config interface{}
		if err := yaml.Unmarshal(data, &config); err != nil {
			http.Error(w, fmt.Sprintf("Failed to parse config: %v", err), http.StatusInternalServerError)
			return
		}

		if err := json.NewEncoder(w).Encode(config); err != nil {
			log.Printf("Error encoding config to JSON: %v", err)
		}
	}
}

// ConfigYAMLHandler reads and serves the config as raw YAML text.
// This is used by the DSL Builder to load the current router config
// and decompile it into DSL via WASM.
func ConfigYAMLHandler(configPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		w.Header().Set("Content-Type", "text/yaml; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")

		data, err := os.ReadFile(configPath)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to read config: %v", err), http.StatusInternalServerError)
			return
		}

		_, _ = w.Write(data)
	}
}

// UpdateConfigHandler updates the config.yaml file with validation.
// After writing, it synchronously propagates the change to the managed runtime
// so Router and Envoy pick up the new config before the API returns success.
func UpdateConfigHandler(configPath string, readonlyMode bool, configDir string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost && r.Method != http.MethodPut {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Check read-only mode
		if readonlyMode {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusForbidden)
			if err := json.NewEncoder(w).Encode(map[string]string{
				"error":   "readonly_mode",
				"message": "Dashboard is in read-only mode. Configuration editing is disabled.",
			}); err != nil {
				log.Printf("Error encoding readonly response: %v", err)
			}
			return
		}

		var configData map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&configData); err != nil {
			http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
			return
		}

		// Read existing config and merge with updates
		existingData, err := os.ReadFile(configPath)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to read existing config: %v", err), http.StatusInternalServerError)
			return
		}

		existingMap := make(map[string]interface{})
		if err = yaml.Unmarshal(existingData, &existingMap); err != nil {
			http.Error(w, fmt.Sprintf("Failed to parse existing config: %v", err), http.StatusInternalServerError)
			return
		}

		// Store original key count for validation
		originalKeyCount := len(existingMap)

		// Merge updates into existing config (recursive deep merge for nested maps)
		existingMap = deepMerge(existingMap, configData)

		// Safety check: merged config should have at least as many keys as original
		// (it might have more if new keys were added, but should never have fewer)
		if len(existingMap) < originalKeyCount {
			http.Error(w, fmt.Sprintf("Merge would result in data loss: original had %d keys, merged has %d keys. This indicates a bug. File: %s", originalKeyCount, len(existingMap), configPath), http.StatusInternalServerError)
			return
		}

		// Convert merged config to YAML
		yamlData, err := yaml.Marshal(existingMap)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to convert to YAML: %v", err), http.StatusInternalServerError)
			return
		}

		// Validate using router's config parser
		tempFile := filepath.Join(os.TempDir(), "config_validate.yaml")
		if err = os.WriteFile(tempFile, yamlData, 0o644); err != nil {
			http.Error(w, fmt.Sprintf("Failed to validate: %v", err), http.StatusInternalServerError)
			return
		}
		defer func() {
			if removeErr := os.Remove(tempFile); removeErr != nil {
				log.Printf("Warning: failed to remove temp file: %v", removeErr)
			}
		}()

		parsedConfig, err := routerconfig.Parse(tempFile)
		if err != nil {
			http.Error(w, fmt.Sprintf("Config validation failed: %v", err), http.StatusBadRequest)
			return
		}

		// Explicitly validate vLLM endpoints (Parse doesn't validate endpoints by default)
		if len(parsedConfig.VLLMEndpoints) > 0 {
			for _, endpoint := range parsedConfig.VLLMEndpoints {
				if err := validateEndpointAddress(endpoint.Address); err != nil {
					http.Error(w, fmt.Sprintf("Config validation failed: vLLM endpoint '%s' address validation failed: %v\n\nSupported formats:\n- IPv4: 192.168.1.1, 127.0.0.1\n- IPv6: ::1, 2001:db8::1\n- DNS names: localhost, example.com, api.example.com\n\nUnsupported formats:\n- Protocol prefixes: http://, https://\n- Paths: /api/v1, /health\n- Ports in address: use 'port' field instead", endpoint.Name, err), http.StatusBadRequest)
					return
				}
			}
		}

		if err := writeConfigAtomically(configPath, yamlData); err != nil {
			http.Error(w, fmt.Sprintf("Failed to write config: %v", err), http.StatusInternalServerError)
			return
		}

		if err := propagateConfigToRuntime(configPath, configDir); err != nil {
			if restoreErr := restorePreviousRuntimeConfig(configPath, configDir, existingData); restoreErr != nil {
				http.Error(w, fmt.Sprintf("Failed to apply config to runtime: %v. Failed to restore previous config: %v", err, restoreErr), http.StatusInternalServerError)
				return
			}
			http.Error(w, fmt.Sprintf("Failed to apply config to runtime: %v. Previous config restored.", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]string{"status": "success"}); err != nil {
			log.Printf("Error encoding response: %v", err)
		}
	}
}

// RouterDefaultsHandler reads and serves the router-defaults.yaml file as JSON
// This file is located in .vllm-sr/router-defaults.yaml relative to config directory
func RouterDefaultsHandler(configDir string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// router-defaults.yaml is in .vllm-sr directory relative to config
		routerDefaultsPath := filepath.Join(configDir, ".vllm-sr", "router-defaults.yaml")

		data, err := os.ReadFile(routerDefaultsPath)
		if err != nil {
			// If file doesn't exist, return empty config
			if os.IsNotExist(err) {
				w.Header().Set("Content-Type", "application/json")
				if encErr := json.NewEncoder(w).Encode(map[string]interface{}{}); encErr != nil {
					log.Printf("Error encoding empty response: %v", encErr)
				}
				return
			}
			http.Error(w, fmt.Sprintf("Failed to read router-defaults: %v", err), http.StatusInternalServerError)
			return
		}

		var config interface{}
		if err := yaml.Unmarshal(data, &config); err != nil {
			http.Error(w, fmt.Sprintf("Failed to parse router-defaults: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(config); err != nil {
			log.Printf("Error encoding router-defaults to JSON: %v", err)
		}
	}
}

// UpdateRouterDefaultsHandler updates the router-defaults.yaml file
func UpdateRouterDefaultsHandler(configDir string, readonlyMode bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost && r.Method != http.MethodPut {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Check read-only mode
		if readonlyMode {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusForbidden)
			if err := json.NewEncoder(w).Encode(map[string]string{
				"error":   "readonly_mode",
				"message": "Dashboard is in read-only mode. Configuration editing is disabled.",
			}); err != nil {
				log.Printf("Error encoding readonly response: %v", err)
			}
			return
		}

		var configData map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&configData); err != nil {
			http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
			return
		}

		routerDefaultsPath := filepath.Join(configDir, ".vllm-sr", "router-defaults.yaml")

		// Read existing config and merge with updates
		existingMap := make(map[string]interface{})
		existingData, err := os.ReadFile(routerDefaultsPath)
		if err == nil {
			if unmarshalErr := yaml.Unmarshal(existingData, &existingMap); unmarshalErr != nil {
				log.Printf("Warning: failed to parse existing router-defaults, starting fresh: %v", unmarshalErr)
			}
		}

		// Merge updates into existing config (deep merge for nested maps)
		for key, value := range configData {
			if existingValue, exists := existingMap[key]; exists {
				if existingMapValue, ok := existingValue.(map[string]interface{}); ok {
					if newMapValue, ok := value.(map[string]interface{}); ok {
						mergedMap := make(map[string]interface{})
						for k, v := range existingMapValue {
							mergedMap[k] = v
						}
						for k, v := range newMapValue {
							mergedMap[k] = v
						}
						existingMap[key] = mergedMap
						continue
					}
				}
			}
			existingMap[key] = value
		}

		// Convert to YAML
		yamlData, err := yaml.Marshal(existingMap)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to convert to YAML: %v", err), http.StatusInternalServerError)
			return
		}

		// Ensure .vllm-sr directory exists
		vllmSrDir := filepath.Join(configDir, ".vllm-sr")
		if mkdirErr := os.MkdirAll(vllmSrDir, 0o755); mkdirErr != nil {
			http.Error(w, fmt.Sprintf("Failed to create .vllm-sr directory: %v", mkdirErr), http.StatusInternalServerError)
			return
		}

		if err := os.WriteFile(routerDefaultsPath, yamlData, 0o644); err != nil {
			http.Error(w, fmt.Sprintf("Failed to write router-defaults: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]string{"status": "success"}); err != nil {
			log.Printf("Error encoding response: %v", err)
		}
	}
}

// validateEndpointAddress validates that an endpoint address is in a valid format.
// It allows:
// - IPv4 addresses (e.g., "192.168.1.1", "127.0.0.1")
// - IPv6 addresses (e.g., "::1", "2001:db8::1")
// - DNS names (e.g., "localhost", "example.com", "api.example.com")
// It rejects:
// - Protocol prefixes (e.g., "http://", "https://")
// - Paths (e.g., "/api/v1", "/health")
// - Ports in the address field (should use the 'port' field instead)
func validateEndpointAddress(address string) error {
	if address == "" {
		return fmt.Errorf("address cannot be empty")
	}

	// Reject protocol prefixes
	if strings.HasPrefix(address, "http://") || strings.HasPrefix(address, "https://") {
		return fmt.Errorf("protocol prefix not allowed in address (use 'port' field for port number)")
	}

	// Reject paths (contains '/')
	if strings.Contains(address, "/") {
		return fmt.Errorf("paths not allowed in address field")
	}

	// Reject ports (contains ':')
	// Note: IPv6 addresses contain ':' but we check for ':' that's not part of IPv6 format
	if strings.Contains(address, ":") {
		// Check if it's a valid IPv6 address (contains multiple colons or starts with '[')
		if net.ParseIP(address) == nil {
			// If it's not a valid IP, it might be an address with a port
			// Check if it looks like "host:port" format
			parts := strings.Split(address, ":")
			if len(parts) == 2 {
				// Could be IPv4:port or hostname:port
				// Try to parse the second part as a port number
				if len(parts[1]) > 0 && len(parts[1]) <= 5 {
					// Likely a port number, reject it
					return fmt.Errorf("port not allowed in address field (use 'port' field instead)")
				}
			}
		}
	}

	// Try to parse as IP address
	ip := net.ParseIP(address)
	if ip != nil {
		// Valid IP address
		return nil
	}

	// If not an IP, check if it's a valid DNS name
	// Basic DNS name validation: alphanumeric, dots, hyphens
	if len(address) > 253 {
		return fmt.Errorf("DNS name too long (max 253 characters)")
	}

	// Check for valid DNS name characters
	for _, char := range address {
		if (char < 'a' || char > 'z') &&
			(char < 'A' || char > 'Z') &&
			(char < '0' || char > '9') &&
			char != '.' && char != '-' {
			return fmt.Errorf("invalid character in DNS name: %c", char)
		}
	}

	// Basic DNS name format check
	if strings.HasPrefix(address, ".") || strings.HasSuffix(address, ".") ||
		strings.Contains(address, "..") {
		return fmt.Errorf("invalid DNS name format")
	}

	return nil
}

const defaultEnvoyConfigPath = "/etc/envoy/envoy.yaml"

// regenerateRouterConfigSync calls the Python CLI to regenerate .vllm-sr/router-config.yaml
// from the user-facing config.yaml. This bridges the gap between the Dashboard
// (which edits the nested Python CLI format) and the Router (which reads the flat format).
// The Router's fsnotify watcher will detect the change and hot-reload automatically.

func regenerateRouterConfigSync(configPath string, configDir string) error {
	outputDir := filepath.Join(configDir, ".vllm-sr")

	// Check if output directory exists; if not, this is likely a dev environment
	// without the Python CLI setup, so skip silently.
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

	cmd := exec.CommandContext(ctx, "python3", "-c", pythonScript)
	cmd.Dir = filepath.Dir(configPath)
	output, err := cmd.CombinedOutput()
	return string(output), err
}

func propagateConfigToRuntime(configPath string, configDir string) error {
	if isRunningInContainer() && isManagedContainerConfigPath(configPath) {
		if err := regenerateRouterConfigSync(configPath, configDir); err != nil {
			return err
		}
		return regenerateAndReloadEnvoyLocally(configPath)
	}

	if getDockerContainerStatus(vllmSrContainerName) == "running" {
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

	cmd := exec.CommandContext(ctx, "python3", "-c", pythonScript)
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
