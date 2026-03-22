package config

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
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

	return ParseYAMLBytes(data)
}

// ParseYAMLBytes parses config YAML content without touching the filesystem.
func ParseYAMLBytes(data []byte) (*RouterConfig, error) {
	raw, err := parseRawConfigMap(data)
	if err != nil {
		return nil, err
	}
	if rejectErr := rejectDeprecatedUserConfigFields(raw); rejectErr != nil {
		return nil, rejectErr
	}

	cfg, err := parseRouterConfigPayload(data, raw)
	if err != nil {
		return nil, err
	}
	if err := finalizeParsedConfig(cfg); err != nil {
		return nil, err
	}

	logging.Debugf("[config.Parse] Config loaded successfully: decisions=%d", len(cfg.Decisions))
	return cfg, nil
}

func parseRawConfigMap(data []byte) (map[string]interface{}, error) {
	var raw map[string]interface{}
	if unmarshalErr := yaml.Unmarshal(data, &raw); unmarshalErr != nil {
		logging.Debugf("[config.Parse] ERROR parsing YAML map: %v", unmarshalErr)
		return nil, fmt.Errorf("failed to parse config file: %w", unmarshalErr)
	}
	return raw, nil
}

func rejectDeprecatedUserConfigFields(raw map[string]interface{}) error {
	if deprecated := deprecatedUserConfigFields(raw); len(deprecated) > 0 {
		return fmt.Errorf(
			"deprecated config fields are no longer supported: %s; rewrite the file to canonical v0.3 providers/routing/global or run `vllm-sr config migrate --config old-config.yaml`",
			strings.Join(deprecated, ", "),
		)
	}
	return nil
}

func parseRouterConfigPayload(data []byte, raw map[string]interface{}) (*RouterConfig, error) {
	if !isCanonicalConfig(raw) {
		return nil, canonicalConfigRequiredError(raw)
	}
	return parseCanonicalConfigPayload(data, raw)
}

func parseCanonicalConfigPayload(data []byte, raw map[string]interface{}) (*RouterConfig, error) {
	canonical := &CanonicalConfig{}
	if unmarshalErr := yaml.Unmarshal(data, canonical); unmarshalErr != nil {
		logging.Debugf("[config.Parse] ERROR parsing canonical YAML: %v", unmarshalErr)
		return nil, fmt.Errorf("failed to parse canonical config file: %w", unmarshalErr)
	}
	if err := attachCanonicalGlobalOverride(raw, canonical); err != nil {
		return nil, err
	}

	cfg, err := normalizeCanonicalConfig(canonical)
	if err != nil {
		logging.Debugf("[config.Parse] ERROR normalizing canonical YAML: %v", err)
		return nil, err
	}
	return cfg, nil
}

func attachCanonicalGlobalOverride(raw map[string]interface{}, canonical *CanonicalConfig) error {
	rawGlobal, ok := raw["global"]
	if !ok {
		return nil
	}

	payload, err := NewStructuredPayload(rawGlobal)
	if err != nil {
		return fmt.Errorf("failed to encode canonical global override: %w", err)
	}
	canonical.globalOverrideRaw = payload
	return nil
}

func canonicalConfigRequiredError(raw map[string]interface{}) error {
	unsupported := unsupportedTopLevelConfigFields(raw)
	detail := "missing canonical routing/global sections"
	if len(unsupported) > 0 {
		detail = fmt.Sprintf("unexpected top-level keys: %s", strings.Join(unsupported, ", "))
	}
	return fmt.Errorf(
		"config file must use canonical v0.3 version/listeners/providers/routing/global; %s; run `vllm-sr config migrate --config old-config.yaml` or rewrite the file to canonical v0.3 providers/routing/global",
		detail,
	)
}

func finalizeParsedConfig(cfg *RouterConfig) error {
	logParsedDecisions(cfg)

	// Apply default model registry if not specified in config.
	// If user specifies mom_registry in config.yaml, it completely replaces the defaults.
	if len(cfg.MoMRegistry) == 0 {
		cfg.MoMRegistry = ToLegacyRegistry()
	}
	if cfg.VectorStore != nil {
		cfg.VectorStore.ApplyDefaults()
	}
	if err := validateConfigStructure(cfg); err != nil {
		logging.Debugf("[config.Parse] ERROR validation failed: %v", err)
		return err
	}
	return nil
}

func logParsedDecisions(cfg *RouterConfig) {
	logging.Debugf("[config.Parse] After unmarshal: decisions=%d", len(cfg.Decisions))
	for i, d := range cfg.Decisions {
		logging.Debugf("[config.Parse]   decision[%d]: name=%q, modelRefs=%d, priority=%d", i, d.Name, len(d.ModelRefs), d.Priority)
	}
}

func deprecatedUserConfigFields(raw map[string]interface{}) []string {
	fields := []string{}

	routing := nestedStringMap(raw["routing"])
	if _, ok := routing["models"]; ok {
		fields = append(fields, "routing.models")
	}

	providers := nestedStringMap(raw["providers"])
	for _, key := range []string{
		"model_targets",
		"backends",
		"auth_profiles",
		"default_model",
		"reasoning_families",
		"default_reasoning_effort",
	} {
		if _, ok := providers[key]; ok {
			fields = append(fields, "providers."+key)
		}
	}

	if models, ok := providers["models"].([]interface{}); ok {
		for index, rawModel := range models {
			model := nestedStringMap(rawModel)
			for _, key := range []string{
				"access",
				"endpoints",
				"access_key",
				"param_size",
				"context_window_size",
				"description",
				"capabilities",
				"loras",
				"quality_score",
				"modality",
				"tags",
			} {
				if _, ok := model[key]; ok {
					fields = append(fields, fmt.Sprintf("providers.models[%d].%s", index, key))
				}
			}
		}
	}

	global := nestedStringMap(raw["global"])
	if _, ok := global["modules"]; ok {
		fields = append(fields, "global.modules")
	}

	fields = append(fields, deprecatedDecisionConfigFields(routing)...)

	return fields
}

func deprecatedDecisionConfigFields(routing map[string]interface{}) []string {
	decisions, ok := routing["decisions"].([]interface{})
	if !ok {
		return nil
	}

	fields := make([]string, 0)
	for index, rawDecision := range decisions {
		decision := nestedStringMap(rawDecision)
		if _, ok := decision["modelSelectionAlgorithm"]; ok {
			fields = append(fields, fmt.Sprintf("routing.decisions[%d].modelSelectionAlgorithm", index))
		}
	}
	return fields
}

func unsupportedTopLevelConfigFields(raw map[string]interface{}) []string {
	allowed := map[string]bool{
		"version":   true,
		"listeners": true,
		"providers": true,
		"routing":   true,
		"global":    true,
	}

	fields := make([]string, 0)
	for key := range raw {
		if !allowed[key] {
			fields = append(fields, key)
		}
	}
	sort.Strings(fields)
	return fields
}

func nestedStringMap(raw interface{}) map[string]interface{} {
	switch typed := raw.(type) {
	case map[string]interface{}:
		return typed
	case map[interface{}]interface{}:
		converted := make(map[string]interface{}, len(typed))
		for key, value := range typed {
			keyString, ok := key.(string)
			if !ok {
				continue
			}
			converted[keyString] = value
		}
		return converted
	default:
		return map[string]interface{}{}
	}
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
