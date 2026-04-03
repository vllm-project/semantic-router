package config

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

var (
	config     *RouterConfig
	configOnce sync.Once
	configErr  error
	configMu   sync.RWMutex

	configUpdateMu          sync.Mutex
	configUpdateSubscribers = map[uint64]chan *RouterConfig{}
	configUpdateNextID      uint64
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
	logging.ComponentDebugEvent("config", "config_parse_started", map[string]interface{}{
		"path":     configPath,
		"resolved": resolved,
	})

	data, err := os.ReadFile(resolved)
	if err != nil {
		logging.ComponentDebugEvent("config", "config_read_failed", map[string]interface{}{
			"resolved": resolved,
			"error":    err.Error(),
		})
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}
	logging.ComponentDebugEvent("config", "config_read_complete", map[string]interface{}{
		"resolved":   resolved,
		"size_bytes": len(data),
	})

	return parseYAMLBytesWithBaseDir(data, filepath.Dir(resolved))
}

// ParseYAMLBytes parses config YAML content without touching the filesystem.
func ParseYAMLBytes(data []byte) (*RouterConfig, error) {
	return parseYAMLBytesWithBaseDir(data, "")
}

func parseYAMLBytesWithBaseDir(data []byte, baseDir string) (*RouterConfig, error) {
	raw, err := parseRawConfigMap(data)
	if err != nil {
		return nil, err
	}
	if rejectErr := rejectDeprecatedUserConfigFields(raw); rejectErr != nil {
		return nil, rejectErr
	}
	if rejectErr := rejectRemovedStructureFields(raw); rejectErr != nil {
		return nil, rejectErr
	}
	if rejectErr := rejectRemovedTaxonomyLegacyFields(raw); rejectErr != nil {
		return nil, rejectErr
	}
	if rejectErr := rejectRemovedDecisionToolFields(raw); rejectErr != nil {
		return nil, rejectErr
	}

	// Warn about unknown YAML fields (typos) before parsing into typed structs.
	WarnUnknownFields(raw, reflect.TypeOf(CanonicalConfig{}))

	cfg, err := parseRouterConfigPayload(data, raw)
	if err != nil {
		return nil, err
	}
	cfg.ConfigBaseDir = baseDir
	if err := finalizeParsedConfig(cfg); err != nil {
		return nil, err
	}

	logging.ComponentDebugEvent("config", "config_parse_complete", map[string]interface{}{
		"decision_count": len(cfg.Decisions),
		"base_dir":       baseDir,
	})
	return cfg, nil
}

func parseRawConfigMap(data []byte) (map[string]interface{}, error) {
	var raw map[string]interface{}
	if unmarshalErr := yaml.Unmarshal(data, &raw); unmarshalErr != nil {
		logging.ComponentDebugEvent("config", "config_yaml_map_parse_failed", map[string]interface{}{
			"error": unmarshalErr.Error(),
		})
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

func rejectRemovedStructureFields(raw map[string]interface{}) error {
	if removed := removedStructureFields(raw); len(removed) > 0 {
		return fmt.Errorf(
			"removed config fields are no longer supported: %s; structure density now uses built-in multilingual normalization and no longer accepts feature.normalize_by",
			strings.Join(removed, ", "),
		)
	}
	return nil
}

func rejectRemovedTaxonomyLegacyFields(raw map[string]interface{}) error {
	routing := nestedStringMap(raw["routing"])
	signals := nestedStringMap(routing["signals"])
	if _, ok := signals["category_kb"]; ok {
		return fmt.Errorf(
			"routing.signals.category_kb is no longer supported; migrate to global.model_catalog.kbs[] plus routing.signals.kb[]",
		)
	}
	if _, ok := signals["taxonomy"]; ok {
		return fmt.Errorf(
			"routing.signals.taxonomy is no longer supported; migrate to routing.signals.kb[]",
		)
	}
	global := nestedStringMap(raw["global"])
	modelCatalog := nestedStringMap(global["model_catalog"])
	if _, ok := modelCatalog["classifiers"]; ok {
		return fmt.Errorf(
			"global.model_catalog.classifiers is no longer supported; migrate to global.model_catalog.kbs[]",
		)
	}
	return nil
}

func rejectRemovedDecisionToolFields(raw map[string]interface{}) error {
	routing := nestedStringMap(raw["routing"])
	decisions, ok := routing["decisions"].([]interface{})
	if !ok {
		return nil
	}

	removed := make([]string, 0)
	for index, rawDecision := range decisions {
		decision := nestedStringMap(rawDecision)
		for _, field := range []string{"tool_scope", "allow_tools", "block_tools"} {
			if _, ok := decision[field]; ok {
				removed = append(removed, fmt.Sprintf("routing.decisions[%d].%s", index, field))
			}
		}
	}
	if len(removed) == 0 {
		return nil
	}

	return fmt.Errorf(
		"removed config fields are no longer supported: %s; migrate to routing.decisions[].plugins[type=tools].configuration",
		strings.Join(removed, ", "),
	)
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
		logging.ComponentDebugEvent("config", "config_canonical_parse_failed", map[string]interface{}{
			"error": unmarshalErr.Error(),
		})
		return nil, fmt.Errorf("failed to parse canonical config file: %w", unmarshalErr)
	}
	if err := attachCanonicalGlobalOverride(raw, canonical); err != nil {
		return nil, err
	}

	cfg, err := normalizeCanonicalConfig(canonical)
	if err != nil {
		logging.ComponentDebugEvent("config", "config_normalize_failed", map[string]interface{}{
			"error": err.Error(),
		})
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
		logging.ComponentDebugEvent("config", "config_validation_failed", map[string]interface{}{
			"error": err.Error(),
		})
		return err
	}
	return nil
}

func logParsedDecisions(cfg *RouterConfig) {
	decisionNames := make([]string, 0, len(cfg.Decisions))
	for _, d := range cfg.Decisions {
		decisionNames = append(decisionNames, d.Name)
	}
	logging.ComponentDebugEvent("config", "config_decisions_parsed", map[string]interface{}{
		"decision_count": len(cfg.Decisions),
		"decision_names": decisionNames,
	})
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
	modelCatalog := nestedStringMap(global["model_catalog"])
	embeddings := nestedStringMap(modelCatalog["embeddings"])
	if _, ok := embeddings["bert"]; ok {
		fields = append(fields, "global.model_catalog.embeddings.bert")
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

func removedStructureFields(raw map[string]interface{}) []string {
	routing := nestedStringMap(raw["routing"])
	signals := nestedStringMap(routing["signals"])
	structureRules, ok := signals["structure"].([]interface{})
	if !ok {
		return nil
	}

	fields := make([]string, 0)
	for index, rawRule := range structureRules {
		rule := nestedStringMap(rawRule)
		feature := nestedStringMap(rule["feature"])
		if _, ok := feature["normalize_by"]; ok {
			fields = append(fields, fmt.Sprintf("routing.signals.structure[%d].feature.normalize_by", index))
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
	decisionNames := make([]string, 0, len(newCfg.Decisions))
	for _, d := range newCfg.Decisions {
		decisionNames = append(decisionNames, d.Name)
	}
	logging.ComponentDebugEvent("config", "config_replace_started", map[string]interface{}{
		"decision_count": len(newCfg.Decisions),
		"decision_names": decisionNames,
	})

	configMu.Lock()
	config = newCfg
	configErr = nil
	configMu.Unlock()

	// Notify listeners of config change.
	configUpdateMu.Lock()
	subscribers := make(map[uint64]chan *RouterConfig, len(configUpdateSubscribers))
	for id, ch := range configUpdateSubscribers {
		subscribers[id] = ch
	}
	configUpdateMu.Unlock()

	if len(subscribers) == 0 {
		logging.ComponentDebugEvent("config", "config_update_listener_missing", nil)
		return
	}
	for id, ch := range subscribers {
		select {
		case ch <- newCfg:
			logging.ComponentDebugEvent("config", "config_update_notified", map[string]interface{}{
				"subscriber_id":  id,
				"decision_count": len(newCfg.Decisions),
			})
		default:
			logging.ComponentWarnEvent("config", "config_update_notification_skipped", map[string]interface{}{
				"subscriber_id":  id,
				"reason":         "channel_full",
				"decision_count": len(newCfg.Decisions),
			})
		}
	}
}

// Get returns the current configuration
func Get() *RouterConfig {
	configMu.RLock()
	defer configMu.RUnlock()
	return config
}

type ConfigUpdateSubscription struct {
	ch        chan *RouterConfig
	closeOnce sync.Once
	closeFn   func()
}

func (s *ConfigUpdateSubscription) Updates() <-chan *RouterConfig {
	if s == nil {
		return nil
	}
	return s.ch
}

func (s *ConfigUpdateSubscription) Close() {
	if s == nil {
		return
	}
	s.closeOnce.Do(func() {
		if s.closeFn != nil {
			s.closeFn()
		}
	})
}

func SubscribeConfigUpdates(buffer int) *ConfigUpdateSubscription {
	if buffer <= 0 {
		buffer = 1
	}

	id := atomic.AddUint64(&configUpdateNextID, 1)
	ch := make(chan *RouterConfig, buffer)

	configUpdateMu.Lock()
	configUpdateSubscribers[id] = ch
	configUpdateMu.Unlock()

	return &ConfigUpdateSubscription{
		ch: ch,
		closeFn: func() {
			configUpdateMu.Lock()
			delete(configUpdateSubscribers, id)
			configUpdateMu.Unlock()
			close(ch)
		},
	}
}

// WatchConfigUpdates returns a compatibility channel that receives config
// updates. New code should prefer SubscribeConfigUpdates so callers can
// explicitly release their subscription.
func WatchConfigUpdates() <-chan *RouterConfig {
	return SubscribeConfigUpdates(1).Updates()
}
