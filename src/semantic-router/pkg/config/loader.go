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

	// Config change notification channel
	configUpdateCh chan *RouterConfig
	configUpdateMu sync.Mutex
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

	// Apply default model registry if not specified in config
	// If user specifies model_registry in config.yaml, it completely replaces the defaults
	if len(cfg.MoMRegistry) == 0 {
		cfg.MoMRegistry = make(map[string]string)
		for k, v := range DefaultModelRegistry {
			cfg.MoMRegistry[k] = v
		}
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
	config = newCfg
	configErr = nil
	configMu.Unlock()

	// Notify listeners of config change
	configUpdateMu.Lock()
	if configUpdateCh != nil {
		select {
		case configUpdateCh <- newCfg:
		default:
			// Channel full or no listener, skip
		}
	}
	configUpdateMu.Unlock()
}

// Get returns the current configuration
func Get() *RouterConfig {
	configMu.RLock()
	defer configMu.RUnlock()
	return config
}

// WatchConfigUpdates returns a channel that receives config updates
// Only one watcher is supported at a time
func WatchConfigUpdates() <-chan *RouterConfig {
	configUpdateMu.Lock()
	defer configUpdateMu.Unlock()

	if configUpdateCh == nil {
		configUpdateCh = make(chan *RouterConfig, 1)
	}
	return configUpdateCh
}
