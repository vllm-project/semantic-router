package config

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"gopkg.in/yaml.v2"
)

var (
	config     *RouterConfig
	configOnce sync.Once
	configErr  error
	configMu   sync.RWMutex
)

// Load loads the configuration from the specified YAML file once and caches it globally.
func Load(configPath string) (*RouterConfig, error) {
	configOnce.Do(func() {
		cfg, err := Parse(configPath)
		if err != nil {
			configErr = err
			return
		}
		configMu.Lock()
		config = cfg
		configMu.Unlock()
	})
	if configErr != nil {
		return nil, configErr
	}
	configMu.RLock()
	defer configMu.RUnlock()
	return config, nil
}

// Parse parses the YAML config file without touching the global cache.
func Parse(configPath string) (*RouterConfig, error) {
	// Resolve symlinks to handle Kubernetes ConfigMap mounts
	resolved, _ := filepath.EvalSymlinks(configPath)
	if resolved == "" {
		resolved = configPath
	}
	data, err := os.ReadFile(resolved)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	cfg := &RouterConfig{}
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	// Validation after parsing
	if err := validateConfigStructure(cfg); err != nil {
		return nil, err
	}

	return cfg, nil
}

// Replace replaces the globally cached config. It is safe for concurrent readers.
func Replace(newCfg *RouterConfig) {
	configMu.Lock()
	defer configMu.Unlock()
	config = newCfg
	// Do not reset configOnce to avoid racing re-parses via LoadConfig; callers should use ParseConfigFile for fresher reads.
	configErr = nil
}

// Get returns the current configuration
func Get() *RouterConfig {
	configMu.RLock()
	defer configMu.RUnlock()
	return config
}
