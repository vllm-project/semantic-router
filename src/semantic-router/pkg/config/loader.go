package config

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
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
)

const ConfigBaseDirEnv = "VLLM_SR_CONFIG_BASE_DIR"

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

	baseDir, err := configBaseDir(filepath.Dir(resolved))
	if err != nil {
		return nil, err
	}
	return parseYAMLBytesWithBaseDir(data, baseDir)
}

func configBaseDir(defaultDir string) (string, error) {
	override := strings.TrimSpace(os.Getenv(ConfigBaseDirEnv))
	if override == "" {
		return filepath.Clean(defaultDir), nil
	}
	if !filepath.IsAbs(override) {
		return "", fmt.Errorf("%s must be an absolute directory: %q", ConfigBaseDirEnv, override)
	}
	cleaned := filepath.Clean(override)
	info, err := os.Stat(cleaned)
	if err != nil {
		return "", fmt.Errorf("invalid %s directory %q: %w", ConfigBaseDirEnv, cleaned, err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("%s must name a directory: %q", ConfigBaseDirEnv, cleaned)
	}
	return cleaned, nil
}

// ParseYAMLBytes parses config YAML content without touching the filesystem.
func ParseYAMLBytes(data []byte) (*RouterConfig, error) {
	return parseYAMLBytesWithOptions(data, "", true)
}

func parseYAMLBytesWithBaseDir(data []byte, baseDir string) (*RouterConfig, error) {
	return parseYAMLBytesWithOptions(data, baseDir, true)
}

// ParseYAMLBytesWithoutEnvExpansion validates in-memory YAML while preserving
// ${VAR} references verbatim. It is intended for read-only validation APIs
// that must not expose process environment values in normalized output.
func ParseYAMLBytesWithoutEnvExpansion(data []byte) (*RouterConfig, error) {
	return parseYAMLBytesWithOptions(data, "", false)
}

func parseYAMLBytesWithOptions(
	data []byte,
	baseDir string,
	expandEnvironment bool,
) (*RouterConfig, error) {
	raw, err := parseRawConfigMap(data)
	if err != nil {
		return nil, err
	}
	if normalizeErr := validateAndNormalizeRawConfig(raw); normalizeErr != nil {
		return nil, normalizeErr
	}

	if expandEnvironment {
		expandEnvSubstitutionsInMap(raw)
	}
	expandedData, marshalErr := yaml.Marshal(raw)
	if marshalErr != nil {
		return nil, fmt.Errorf("failed to marshal normalized config input: %w", marshalErr)
	}

	// Warn about unknown YAML fields (typos) before parsing into typed structs.
	WarnUnknownFields(raw, reflect.TypeOf(CanonicalConfig{}))

	cfg, err := parseRouterConfigPayload(expandedData, raw)
	if err != nil {
		return nil, err
	}
	cfg.ConfigBaseDir = baseDir
	documentDigest := sha256.Sum256(data)
	cfg.DocumentHash = hex.EncodeToString(documentDigest[:])
	cfg.SkipExternalAssetValidation = !expandEnvironment
	if err := finalizeParsedConfig(cfg); err != nil {
		return nil, err
	}

	logging.ComponentDebugEvent("config", "config_parse_complete", map[string]interface{}{
		"decision_count": len(cfg.Decisions),
		"base_dir":       baseDir,
	})
	return cfg, nil
}

func validateAndNormalizeRawConfig(raw map[string]interface{}) error {
	validators := []func(map[string]interface{}) error{
		normalizeResponseCacheAliases,
		rejectDeprecatedUserConfigFields,
		rejectRemovedStructureFields,
		rejectRemovedTaxonomyLegacyFields,
		rejectRemovedDecisionToolFields,
		rejectRemovedRouterLearningFields,
		rejectUnsupportedRouterLearningFields,
	}
	for _, validate := range validators {
		if err := validate(raw); err != nil {
			return err
		}
	}
	return nil
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

func rejectRemovedRouterLearningFields(raw map[string]interface{}) error {
	global := nestedStringMap(raw["global"])
	router := nestedStringMap(global["router"])
	modelSelection := nestedStringMap(router["model_selection"])

	removedGlobal := make([]string, 0)
	for _, field := range []string{
		"session_aware",
		"model_switch_gate",
		"lookup_tables",
		"elo",
		"rl_driven",
		"gmtrouter",
		"bandit",
		"personalization",
	} {
		if _, ok := modelSelection[field]; ok {
			removedGlobal = append(removedGlobal, "global.router.model_selection."+field)
		}
	}
	if method := strings.TrimSpace(fmt.Sprint(modelSelection["method"])); removedGlobalLearningSelector(method) {
		removedGlobal = append(removedGlobal, "global.router.model_selection.method="+method)
	}
	if len(removedGlobal) > 0 {
		return fmt.Errorf(
			"removed config fields are no longer supported: %s; use global.router.learning.adaptation and global.router.learning.protection for cross-request learning",
			strings.Join(removedGlobal, ", "),
		)
	}

	routing := nestedStringMap(raw["routing"])
	decisions, ok := routing["decisions"].([]interface{})
	if !ok {
		return nil
	}
	for index, rawDecision := range decisions {
		decision := nestedStringMap(rawDecision)
		if err := rejectRemovedDecisionLearningFields(index, nestedStringMap(decision["algorithm"])); err != nil {
			return err
		}
	}
	return nil
}

func rejectRemovedDecisionLearningFields(index int, algorithm map[string]interface{}) error {
	algorithmType := strings.TrimSpace(fmt.Sprint(algorithm["type"]))
	if err := removedLearningAlgorithmTypeError(index, algorithmType); err != nil {
		return err
	}
	if _, ok := algorithm["session_aware"]; ok {
		return fmt.Errorf(
			"routing.decisions[%d].algorithm.session_aware is no longer supported; remove algorithm.session_aware and configure global.router.learning.protection plus routing.decisions[].adaptations when this decision needs apply/observe/bypass control",
			index,
		)
	}
	for field, adaptation := range removedDecisionLearningBlocks() {
		if _, ok := algorithm[field]; ok {
			return fmt.Errorf(
				"routing.decisions[%d].algorithm.%s has moved to %s; remove the decision-local learning algorithm block",
				index,
				field,
				adaptation,
			)
		}
	}
	return nil
}

func removedLearningAlgorithmTypeError(index int, algorithmType string) error {
	if algorithmType == "session_aware" {
		return fmt.Errorf(
			"routing.decisions[%d].algorithm.type=session_aware is no longer supported; remove algorithm.type=session_aware and enable global.router.learning.protection",
			index,
		)
	}
	if target, ok := removedDecisionLearningBlocks()[algorithmType]; ok {
		return fmt.Errorf(
			"routing.decisions[%d].algorithm.type=%s has moved to %s; remove algorithm.type=%s and choose a request-time base algorithm only when needed",
			index,
			algorithmType,
			target,
			algorithmType,
		)
	}
	return nil
}

func removedGlobalLearningSelector(method string) bool {
	switch strings.TrimSpace(method) {
	case "session_aware", "lookup_tables", "elo", "rl_driven", "gmtrouter", "bandit", "personalization":
		return true
	default:
		return false
	}
}

func removedDecisionLearningBlocks() map[string]string {
	return map[string]string{
		"elo":             "global.router.learning.adaptation",
		"rl_driven":       "global.router.learning.adaptation",
		"gmtrouter":       "global.router.learning.adaptation",
		"bandit":          "global.router.learning.adaptation",
		"personalization": "global.router.learning.adaptation",
	}
}

func rejectUnsupportedRouterLearningFields(raw map[string]interface{}) error {
	if err := rejectUnsupportedGlobalRouterLearningFields(raw); err != nil {
		return err
	}
	return rejectUnsupportedDecisionAdaptationFields(raw)
}

func rejectUnsupportedGlobalRouterLearningFields(raw map[string]interface{}) error {
	global := nestedStringMap(raw["global"])
	router := nestedStringMap(global["router"])
	learning := nestedStringMap(router["learning"])
	if len(learning) == 0 {
		return nil
	}
	if err := rejectUnknownMapFields(
		"global.router.learning",
		learning,
		[]string{"enabled", "adaptation", "protection", "state_store"},
	); err != nil {
		return err
	}
	if err := rejectUnknownMapFields(
		"global.router.learning.adaptation",
		nestedStringMap(learning["adaptation"]),
		[]string{"enabled", "candidate_set", "strategy"},
	); err != nil {
		return err
	}
	if err := rejectUnsupportedProtectionLearningFields(
		"global.router.learning.protection",
		nestedStringMap(learning["protection"]),
		[]string{"enabled", "scope", "identity", "tuning"},
	); err != nil {
		return err
	}
	return rejectUnsupportedRouterLearningStateStoreFields(learning)
}

func rejectUnsupportedDecisionAdaptationFields(raw map[string]interface{}) error {
	routing := nestedStringMap(raw["routing"])
	decisions, ok := routing["decisions"].([]interface{})
	if !ok {
		return nil
	}
	for index, rawDecision := range decisions {
		decision := nestedStringMap(rawDecision)
		adaptations := nestedStringMap(decision["adaptations"])
		if len(adaptations) == 0 {
			continue
		}
		prefix := fmt.Sprintf("routing.decisions[%d].adaptations", index)
		if err := rejectUnknownMapFields(prefix, adaptations, []string{"mode", "adaptation", "protection"}); err != nil {
			return err
		}
		if err := rejectUnknownMapFields(
			prefix+".adaptation",
			nestedStringMap(adaptations["adaptation"]),
			[]string{"mode", "candidate_set"},
		); err != nil {
			return err
		}
		if err := rejectUnknownMapFields(
			prefix+".protection",
			nestedStringMap(adaptations["protection"]),
			[]string{"mode", "stability_weight", "switch_margin"},
		); err != nil {
			return err
		}
	}
	return nil
}

func rejectUnsupportedProtectionLearningFields(prefix string, raw map[string]interface{}, allowed []string) error {
	if len(raw) == 0 {
		return nil
	}
	if err := rejectUnknownMapFields(prefix, raw, allowed); err != nil {
		return err
	}
	if identity, ok := raw["identity"]; ok {
		identityMap := nestedStringMap(identity)
		if err := rejectUnknownMapFields(prefix+".identity", identityMap, []string{"headers"}); err != nil {
			return err
		}
		if err := rejectUnknownMapFields(
			prefix+".identity.headers",
			nestedStringMap(identityMap["headers"]),
			[]string{"session", "conversation"},
		); err != nil {
			return err
		}
	}
	if tuning, ok := raw["tuning"]; ok {
		if err := rejectUnknownMapFields(prefix+".tuning", nestedStringMap(tuning), []string{
			"idle_timeout_seconds",
			"min_turns_before_switch",
			"switch_margin",
			"stability_weight",
		}); err != nil {
			return err
		}
	}
	return nil
}

func rejectUnknownMapFields(prefix string, raw map[string]interface{}, allowed []string) error {
	allowedSet := make(map[string]struct{}, len(allowed))
	for _, key := range allowed {
		allowedSet[key] = struct{}{}
	}
	unknown := make([]string, 0)
	for key := range raw {
		if _, ok := allowedSet[key]; !ok {
			unknown = append(unknown, prefix+"."+key)
		}
	}
	if len(unknown) == 0 {
		return nil
	}
	sort.Strings(unknown)
	return fmt.Errorf("unsupported Router Learning config fields: %s", strings.Join(unknown, ", "))
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
