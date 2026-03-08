package handlers

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func validateDashboardConfig(configPath string, configData map[string]interface{}) error {
	if shouldValidateWithPythonCLI(configData) {
		output, err := validateUserConfigWithPython(configPath)
		if err != nil {
			trimmed := strings.TrimSpace(output)
			if trimmed == "" {
				trimmed = err.Error()
			}
			return fmt.Errorf("config validation failed: %s", trimmed)
		}
		return nil
	}

	parsedConfig, err := routerconfig.Parse(configPath)
	if err != nil {
		return fmt.Errorf("config validation failed: %w", err)
	}

	for _, endpoint := range parsedConfig.VLLMEndpoints {
		if err := validateEndpointAddress(endpoint.Address); err != nil {
			return fmt.Errorf("config validation failed: vLLM endpoint %q address validation failed: %w\n\nSupported formats:\n- IPv4: 192.168.1.1, 127.0.0.1\n- IPv6: ::1, 2001:db8::1\n- DNS names: localhost, example.com, api.example.com\n\nUnsupported formats:\n- Protocol prefixes: http://, https://\n- Paths: /api/v1, /health\n- Ports in address: use 'port' field instead", endpoint.Name, err)
		}
	}

	return nil
}

func shouldValidateWithPythonCLI(configData map[string]interface{}) bool {
	if len(configData) == 0 {
		return false
	}
	_, hasProviders := configData["providers"]
	return hasProviders
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
	if strings.HasPrefix(address, "http://") || strings.HasPrefix(address, "https://") {
		return fmt.Errorf("protocol prefix not allowed in address (use 'port' field for port number)")
	}
	if strings.Contains(address, "/") {
		return fmt.Errorf("paths not allowed in address field")
	}
	if hasEndpointPort(address) {
		return fmt.Errorf("port not allowed in address field (use 'port' field instead)")
	}

	if ip := net.ParseIP(address); ip != nil {
		return nil
	}
	return validateDNSName(address)
}

func hasEndpointPort(address string) bool {
	if !strings.Contains(address, ":") || net.ParseIP(address) != nil {
		return false
	}

	parts := strings.Split(address, ":")
	return len(parts) == 2 && len(parts[1]) > 0 && len(parts[1]) <= 5
}

func validateDNSName(address string) error {
	if len(address) > 253 {
		return fmt.Errorf("DNS name too long (max 253 characters)")
	}
	for _, char := range address {
		if !isAllowedDNSCharacter(char) {
			return fmt.Errorf("invalid character in DNS name: %c", char)
		}
	}
	if strings.HasPrefix(address, ".") || strings.HasSuffix(address, ".") || strings.Contains(address, "..") {
		return fmt.Errorf("invalid DNS name format")
	}

	return nil
}

func isAllowedDNSCharacter(char rune) bool {
	if char == '.' || char == '-' {
		return true
	}
	if char >= 'a' && char <= 'z' {
		return true
	}
	if char >= 'A' && char <= 'Z' {
		return true
	}
	return char >= '0' && char <= '9'
}

func validateUserConfigWithPython(configPath string) (string, error) {
	cliRoot := detectPythonCLIRoot()
	if cliRoot == "" {
		return "", fmt.Errorf("python CLI not available for dashboard config validation")
	}
	pythonBin := detectPythonCLIExecutable()

	pythonScript := fmt.Sprintf(`
import sys
sys.path.insert(0, %q)
try:
    from cli.parser import parse_user_config
    from cli.validator import validate_user_config
    user_config = parse_user_config(%q)
    errors = validate_user_config(user_config)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        sys.exit(1)
    print("Validated user config")
except ImportError:
    print("Python CLI not available for dashboard config validation", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(e, file=sys.stderr)
    sys.exit(1)
`, cliRoot, configPath)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, pythonBin, "-c", pythonScript)
	cmd.Dir = filepath.Dir(configPath)
	output, err := cmd.CombinedOutput()
	return string(output), err
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
			filepath.Clean(filepath.Join(wd, "..", "..", "..", "..", "src", "vllm-sr")),
			filepath.Clean(filepath.Join(wd, "..", "..", "..", "src", "vllm-sr")),
			filepath.Clean(filepath.Join(wd, "..", "..", "src", "vllm-sr")),
			filepath.Clean(filepath.Join(wd, "src", "vllm-sr")),
		)
	}
	if _, thisFile, _, ok := runtime.Caller(0); ok {
		candidates = append(
			candidates,
			filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", "..", "..", "..", "src", "vllm-sr")),
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

func detectPythonCLIExecutable() string {
	candidates := []string{}
	if envPath := strings.TrimSpace(os.Getenv("VLLM_SR_PYTHON_BIN")); envPath != "" {
		candidates = append(candidates, envPath)
	}
	candidates = append(candidates, "python", "python3")

	for _, candidate := range candidates {
		if candidate == "" {
			continue
		}
		if resolved, err := exec.LookPath(candidate); err == nil {
			return resolved
		}
	}

	return "python3"
}
