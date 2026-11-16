package config

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
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
	return parseYAMLBytesWithOptions(data, "", &processEnv)
}

func parseYAMLBytesWithBaseDir(data []byte, baseDir string) (*RouterConfig, error) {
	return parseYAMLBytesWithOptions(data, baseDir, &processEnv)
}

// ParseYAMLBytesWithoutEnvExpansion validates in-memory YAML while preserving
// ${VAR} references verbatim. It is intended for read-only validation APIs
// that must not expose process environment values in normalized output.
func ParseYAMLBytesWithoutEnvExpansion(data []byte) (*RouterConfig, error) {
	return parseYAMLBytesWithOptions(data, "", nil)
}

// ParseGlobalYAMLBytesWithoutEnvExpansion parses a document that holds a
// global block whose recipes are elsewhere, as ParseYAMLBytesWithoutEnvExpansion
// does, but runs only the checks of the global settings, as for a Kubernetes
// source. A check of what the recipes ask, such as which calls reach a
// delegated upstream, runs when the whole config loads.
func ParseGlobalYAMLBytesWithoutEnvExpansion(data []byte) (*RouterConfig, error) {
	cfg, err := decodeYAMLBytes(data, nil)
	if err != nil {
		return nil, err
	}
	cfg.SkipExternalAssetValidation = true
	applyParsedConfigDefaults(cfg)
	if err := validateConfigContractsAtStage(cfg, staticConfigValidation); err != nil {
		return nil, err
	}
	return cfg, nil
}

// ValidateYAMLBytesDeferringEnv validates a config that another process will
// load with its own environment, so it resolves no reference from this one:
// references take their defaults, and the config passes if it is valid with
// each variable referenced without a default either kept as written or empty,
// as that process reads an unset variable. External assets are checked as in
// ParseYAMLBytes.
func ValidateYAMLBytesDeferringEnv(data []byte) error {
	_, err := deferredEnvExpander(data)
	return err
}

// ParseYAMLBytesDeferringEnv parses a config for an editor of its routing,
// entrypoints and recipes, such as the DSL, when another process loads the
// config with its own environment. It checks the config as
// ValidateYAMLBytesDeferringEnv does. Those three sections keep ${VAR}
// references and $$ escapes as written, so the editor can write them back;
// the other sections resolve references as that check did.
func ParseYAMLBytesDeferringEnv(data []byte) (*RouterConfig, error) {
	env, err := deferredEnvExpander(data)
	if err != nil {
		return nil, err
	}
	// Only the resolved form is checked; references kept as written fail typed checks, such as a token count.
	cfg, err := decodeYAMLBytes(data, func(raw map[string]interface{}) {
		for key, value := range raw {
			if !routingSections[key] {
				raw[key] = env.expandValue(value)
			}
		}
	})
	if err != nil {
		return nil, err
	}
	applyParsedConfigDefaults(cfg)
	return cfg, nil
}

// routingSections are the top-level sections that hold routing state.
var routingSections = map[string]bool{"recipes": true, "entrypoints": true}

// deferredEnvExpander returns the expander with which data passes
// ValidateYAMLBytesDeferringEnv. Each variable starts kept as written. A check
// that rejects a value it parses, such as a token count or a duration, quotes
// the value, and each variable it quotes is then read as empty. If no such mix
// passes, every variable is tried empty.
func deferredEnvExpander(data []byte) (*envExpander, error) {
	keep := &envExpander{lookup: noEnv, keepUnset: true, unset: map[string]bool{}, kept: map[string]string{}}
	_, err := parseYAMLBytesWithOptions(data, "", keep)
	keptErr := err
	for err != nil && keep.unsetQuoted(err.Error()) {
		_, err = parseYAMLBytesWithOptions(data, "", keep)
	}
	if err == nil {
		return keep, nil
	}
	empty := &envExpander{lookup: noEnv}
	if _, emptyErr := parseYAMLBytesWithOptions(data, "", empty); emptyErr == nil {
		return empty, nil
	}
	return nil, keptErr
}

// parseYAMLBytesWithOptions keeps references as written and skips external
// assets when env is nil.
func parseYAMLBytesWithOptions(
	data []byte,
	baseDir string,
	env *envExpander,
) (*RouterConfig, error) {
	var expand func(map[string]interface{})
	if env != nil {
		expand = env.expandMap
	}
	cfg, err := decodeYAMLBytes(data, expand)
	if err != nil {
		return nil, err
	}
	cfg.ConfigBaseDir = baseDir
	cfg.SkipExternalAssetValidation = env == nil
	if err := finalizeParsedConfig(cfg); err != nil {
		return nil, err
	}

	logging.ComponentDebugEvent("config", "config_parse_complete", map[string]interface{}{
		"decision_count": len(cfg.Decisions),
		"base_dir":       baseDir,
	})
	return cfg, nil
}

// decodeYAMLBytes builds a config without the checks of finalizeParsedConfig.
// A non-nil expand rewrites the raw document before it is decoded.
func decodeYAMLBytes(data []byte, expand func(map[string]interface{})) (*RouterConfig, error) {
	written := data
	data, formWarnings, err := applyConfigForms(data, configForms)
	if err != nil {
		return nil, err
	}
	raw, err := parseRawConfigMap(data)
	if err != nil {
		return nil, err
	}
	if normalizeErr := validateAndNormalizeRawConfig(raw); normalizeErr != nil {
		return nil, normalizeErr
	}

	if expand != nil {
		expand(raw)
	}
	expandedData, marshalErr := yaml.Marshal(raw)
	if marshalErr != nil {
		return nil, fmt.Errorf("failed to marshal normalized config input: %w", marshalErr)
	}

	if !isCanonicalConfig(raw) {
		return nil, canonicalConfigRequiredError(raw)
	}
	if validationErr := validateKnownFields(raw, reflect.TypeOf(CanonicalConfig{})); validationErr != nil {
		return nil, validationErr
	}
	cfg, err := parseCanonicalConfigPayload(expandedData, raw)
	if err != nil {
		return nil, err
	}
	cfg.FormWarnings = formWarnings
	documentDigest := sha256.Sum256(written)
	cfg.DocumentHash = hex.EncodeToString(documentDigest[:])
	return cfg, nil
}

func validateAndNormalizeRawConfig(raw map[string]interface{}) error {
	validators := []func(map[string]interface{}) error{
		rejectDeprecatedUserConfigFields,
		rejectRemovedTaxonomyLegacyFields,
		rejectRemovedDecisionToolFields,
		rejectRemovedRouterLearningFields,
		rejectUnsupportedRouterLearningFields,
		rejectRemovedBackendRefFields,
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
			"deprecated config fields are no longer supported: %s; rewrite them in the canonical v0.5 providers/recipes/global layout",
			strings.Join(deprecated, ", "),
		)
	}
	return nil
}

func rejectRemovedTaxonomyLegacyFields(raw map[string]interface{}) error {
	for _, recipe := range rawRecipes(raw) {
		signals := nestedStringMap(recipe.fields["signals"])
		for _, field := range []string{"category_kb", "taxonomy"} {
			if _, ok := signals[field]; ok {
				return fmt.Errorf(
					"%s.signals.%s is no longer supported; write a systemone set or choice question instead",
					recipe.path, field,
				)
			}
		}
	}
	global := nestedStringMap(raw["global"])
	modelCatalog := nestedStringMap(global["model_catalog"])
	if _, ok := modelCatalog["classifiers"]; ok {
		return fmt.Errorf(
			"global.model_catalog.classifiers is no longer supported; write a systemone set or choice question instead",
		)
	}
	return nil
}

// rawRecipe is one recipe of a raw document, at its path.
type rawRecipe struct {
	path   string
	fields map[string]interface{}
}

func rawRecipes(raw map[string]interface{}) []rawRecipe {
	items, _ := raw["recipes"].([]interface{})
	recipes := make([]rawRecipe, 0, len(items))
	for index, item := range items {
		fields := nestedStringMap(item)
		name, _ := fields["name"].(string)
		if name == "" {
			name = strconv.Itoa(index)
		}
		recipes = append(recipes, rawRecipe{path: recipePath(name), fields: fields})
	}
	return recipes
}

// decisions are the recipe's decisions, each at its path.
func (r rawRecipe) decisions() ([]string, []map[string]interface{}) {
	items, _ := r.fields["decisions"].([]interface{})
	paths := make([]string, 0, len(items))
	decisions := make([]map[string]interface{}, 0, len(items))
	for index, item := range items {
		paths = append(paths, fmt.Sprintf("%s.decisions[%d]", r.path, index))
		decisions = append(decisions, nestedStringMap(item))
	}
	return paths, decisions
}

func rejectRemovedDecisionToolFields(raw map[string]interface{}) error {
	removed := make([]string, 0)
	for _, recipe := range rawRecipes(raw) {
		paths, decisions := recipe.decisions()
		for index, decision := range decisions {
			for _, field := range []string{"tool_scope", "allow_tools", "block_tools"} {
				if _, ok := decision[field]; ok {
					removed = append(removed, paths[index]+"."+field)
				}
			}
		}
	}
	if len(removed) == 0 {
		return nil
	}

	return fmt.Errorf(
		"removed config fields are no longer supported: %s; migrate to recipes[].decisions[].plugins[type=tools].configuration",
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
	if len(removedGlobal) > 0 {
		return fmt.Errorf(
			"removed config fields are no longer supported: %s; use global.router.learning.adaptation and global.router.learning.protection for cross-request learning",
			strings.Join(removedGlobal, ", "),
		)
	}

	for _, recipe := range rawRecipes(raw) {
		paths, decisions := recipe.decisions()
		for index, decision := range decisions {
			if err := rejectRemovedDecisionLearningFields(paths[index], nestedStringMap(decision["algorithm"])); err != nil {
				return err
			}
		}
	}
	return nil
}

func rejectRemovedDecisionLearningFields(path string, algorithm map[string]interface{}) error {
	algorithmType := strings.TrimSpace(fmt.Sprint(algorithm["type"]))
	if err := removedLearningAlgorithmTypeError(path, algorithmType); err != nil {
		return err
	}
	if _, ok := algorithm["session_aware"]; ok {
		return fmt.Errorf(
			"%s.algorithm.session_aware is no longer supported; remove algorithm.session_aware and configure global.router.learning.protection plus recipes[].decisions[].adaptations when this decision needs apply/observe/bypass control",
			path,
		)
	}
	for field, adaptation := range removedDecisionLearningBlocks() {
		if _, ok := algorithm[field]; ok {
			return fmt.Errorf(
				"%s.algorithm.%s has moved to %s; remove the decision-local learning algorithm block",
				path,
				field,
				adaptation,
			)
		}
	}
	return nil
}

func removedLearningAlgorithmTypeError(path string, algorithmType string) error {
	if algorithmType == "session_aware" {
		return fmt.Errorf(
			"%s.algorithm.type=session_aware is no longer supported; remove algorithm.type=session_aware and enable global.router.learning.protection",
			path,
		)
	}
	if target, ok := removedDecisionLearningBlocks()[algorithmType]; ok {
		return fmt.Errorf(
			"%s.algorithm.type=%s has moved to %s; remove algorithm.type=%s and choose a request-time base algorithm only when needed",
			path,
			algorithmType,
			target,
			algorithmType,
		)
	}
	return nil
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
		[]string{"enabled", "scope", "identity", "tuning", "release"},
	); err != nil {
		return err
	}
	return rejectUnsupportedRouterLearningStateStoreFields(learning)
}

func rejectUnsupportedDecisionAdaptationFields(raw map[string]interface{}) error {
	for _, recipe := range rawRecipes(raw) {
		if err := rejectUnsupportedRecipeAdaptationFields(recipe); err != nil {
			return err
		}
	}
	return nil
}

func rejectUnsupportedRecipeAdaptationFields(recipe rawRecipe) error {
	paths, decisions := recipe.decisions()
	for index, decision := range decisions {
		adaptations := nestedStringMap(decision["adaptations"])
		if len(adaptations) == 0 {
			continue
		}
		prefix := paths[index] + ".adaptations"
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
			[]string{"session", "conversation", "agent"},
		); err != nil {
			return err
		}
	}
	if release, ok := raw["release"]; ok {
		if err := rejectUnknownMapFields(
			prefix+".release",
			nestedStringMap(release),
			[]string{"enabled", "header", "min_user_turns_between"},
		); err != nil {
			return err
		}
	}
	if tuning, ok := raw["tuning"]; ok {
		tuningMap := nestedStringMap(tuning)
		if err := rejectUnknownMapFields(prefix+".tuning", tuningMap, []string{
			"idle_timeout_seconds",
			"min_turns_before_switch",
			"switch_margin",
			"stability_weight",
			"progress_gate",
		}); err != nil {
			return err
		}
		if gate, ok := tuningMap["progress_gate"]; ok {
			if err := rejectUnknownMapFields(
				prefix+".tuning.progress_gate",
				nestedStringMap(gate),
				[]string{
					"enabled",
					"mode",
					"calibration_id",
					"window_size",
					"window_ttl_seconds",
					"min_window_outcomes",
					"min_consecutive_regressions",
					"min_consecutive_recoveries",
					"cooldown_seconds",
					"max_switches_per_window",
				},
			); err != nil {
				return err
			}
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
	if err := refuseEmptyGlobalValue(rawGlobal); err != nil {
		return err
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
	detail := "missing canonical recipes/global sections"
	if len(unsupported) > 0 {
		detail = fmt.Sprintf("unexpected top-level keys: %s", strings.Join(unsupported, ", "))
	}
	return fmt.Errorf("config file must use the canonical v0.5 hierarchy; %s", detail)
}

func finalizeParsedConfig(cfg *RouterConfig) error {
	applyParsedConfigDefaults(cfg)
	if err := validateConfigStructure(cfg); err != nil {
		logging.ComponentDebugEvent("config", "config_validation_failed", map[string]interface{}{
			"error": err.Error(),
		})
		return err
	}
	return nil
}

func applyParsedConfigDefaults(cfg *RouterConfig) {
	logParsedDecisions(cfg)

	// Apply default model registry if not specified in config.
	// If user specifies mom_registry in config.yaml, it completely replaces the defaults.
	if len(cfg.MoMRegistry) == 0 {
		cfg.MoMRegistry = ToLegacyRegistry()
	}
	if cfg.VectorStore != nil {
		cfg.VectorStore.ApplyDefaults()
	}
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

// deprecatedUserConfigFields are earlier fields no migrate rewrites.
func deprecatedUserConfigFields(raw map[string]interface{}) []string {
	var fields []string
	embeddings := nestedStringMap(nestedStringMap(nestedStringMap(raw["global"])["model_catalog"])["embeddings"])
	if _, ok := embeddings["bert"]; ok {
		fields = append(fields, "global.model_catalog.embeddings.bert")
	}
	return append(fields, deprecatedDecisionConfigFields(raw)...)
}

func deprecatedDecisionConfigFields(raw map[string]interface{}) []string {
	fields := make([]string, 0)
	for _, recipe := range rawRecipes(raw) {
		paths, decisions := recipe.decisions()
		for index, decision := range decisions {
			if _, ok := decision["modelSelectionAlgorithm"]; ok {
				fields = append(fields, paths[index]+".modelSelectionAlgorithm")
			}
		}
	}
	return fields
}

func unsupportedTopLevelConfigFields(raw map[string]interface{}) []string {
	allowed := map[string]bool{
		"version":     true,
		"listeners":   true,
		"providers":   true,
		"evaluation":  true,
		"entrypoints": true,
		"recipes":     true,
		"global":      true,
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
