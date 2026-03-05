package config

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
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
	logging.Debugf("[config.Parse] Loading config: path=%s, resolved=%s", configPath, resolved)

	data, err := os.ReadFile(resolved)
	if err != nil {
		logging.Debugf("[config.Parse] ERROR reading config file: %v", err)
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}
	logging.Debugf("[config.Parse] Read config file: size=%d bytes", len(data))

	cfg := &RouterConfig{}
	if err := yaml.Unmarshal(data, cfg); err != nil {
		logging.Debugf("[config.Parse] ERROR parsing YAML: %v", err)
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	// Log decisions found after YAML unmarshal
	logging.Debugf("[config.Parse] After unmarshal: decisions=%d", len(cfg.Decisions))
	for i, d := range cfg.Decisions {
		logging.Debugf("[config.Parse]   decision[%d]: name=%q, modelRefs=%d, priority=%d", i, d.Name, len(d.ModelRefs), d.Priority)
	}

	// Apply default model registry if not specified in config
	// If user specifies mom_registry in config.yaml, it completely replaces the defaults
	if len(cfg.MoMRegistry) == 0 {
		cfg.MoMRegistry = ToLegacyRegistry()
	}

	// Validation after parsing
	if err := validateConfigStructure(cfg); err != nil {
		logging.Debugf("[config.Parse] ERROR validation failed: %v", err)
		return nil, err
	}

	logging.Debugf("[config.Parse] Config loaded successfully: decisions=%d", len(cfg.Decisions))
	return cfg, nil
}

// Replace replaces the globally cached config. It is safe for concurrent readers.
func Replace(newCfg *RouterConfig) {
	logging.Debugf("[config.Replace] Replacing global config: decisions=%d", len(newCfg.Decisions))
	for i, d := range newCfg.Decisions {
		logging.Debugf("[config.Replace]   decision[%d]: name=%q, modelRefs=%d", i, d.Name, len(d.ModelRefs))
	}

	configMu.Lock()
	config = newCfg
	configErr = nil
	configMu.Unlock()

	// Notify listeners of config change
	configUpdateMu.Lock()
	if configUpdateCh != nil {
		select {
		case configUpdateCh <- newCfg:
			logging.Debugf("[config.Replace] Notified config update listener")
		default:
			logging.Debugf("[config.Replace] WARNING: config update channel full or no listener, notification skipped")
		}
	} else {
		logging.Debugf("[config.Replace] No config update channel registered")
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
