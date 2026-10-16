package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// The CLI already supplies this shared resource root to Router and Dashboard.
// Keep the process working directory as the standalone/container default;
// ConfigDir describes writable Dashboard state and is not an asset root.
func resolveConfigBaseDir() (string, error) {
	const key = "VLLM_SR_CONFIG_BASE_DIR"
	base := strings.TrimSpace(os.Getenv(key))
	if base == "" {
		return os.Getwd()
	}
	if !filepath.IsAbs(base) {
		return "", fmt.Errorf("%s must be an absolute directory: %q", key, base)
	}
	base = filepath.Clean(base)
	info, err := os.Stat(base)
	if err != nil {
		return "", fmt.Errorf("invalid %s directory: %w", key, err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("%s must name a directory: %q", key, base)
	}
	return base, nil
}
