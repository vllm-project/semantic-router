package config

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	yamlv2 "gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelauthoring"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

var (
	config     *RouterConfig
	configOnce sync.Once
	configErr  error
	configMu   sync.RWMutex
)

const ConfigBaseDirEnv = "VLLM_SR_CONFIG_BASE_DIR"

// Parser owns the application-provided Provider Integration compiler used at
// the human-authoring boundary. A nil compiler remains valid for internal
// publication bootstrap documents, but file-authored Models fail closed.
type Parser struct {
	connectionCompiler modelauthoring.ConnectionCompiler
}

func NewParser(connectionCompiler modelauthoring.ConnectionCompiler) *Parser {
	return &Parser{connectionCompiler: connectionCompiler}
}

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
	return NewParser(nil).Parse(configPath)
}

func (parser *Parser) Parse(configPath string) (*RouterConfig, error) {
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
	return parser.parseYAMLBytesWithBaseDir(data, baseDir)
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
	return NewParser(nil).ParseYAMLBytes(data)
}

func (parser *Parser) ParseYAMLBytes(data []byte) (*RouterConfig, error) {
	return parser.parseYAMLBytesWithOptions(data, "", true)
}

func (parser *Parser) parseYAMLBytesWithBaseDir(data []byte, baseDir string) (*RouterConfig, error) {
	return parser.parseYAMLBytesWithOptions(data, baseDir, true)
}

// ParseYAMLBytesWithoutEnvExpansion validates in-memory YAML while preserving
// ${VAR} references verbatim. It is intended for read-only validation APIs
// that must not expose process environment values in normalized output.
func ParseYAMLBytesWithoutEnvExpansion(data []byte) (*RouterConfig, error) {
	return NewParser(nil).ParseYAMLBytesWithoutEnvExpansion(data)
}

func (parser *Parser) ParseYAMLBytesWithoutEnvExpansion(data []byte) (*RouterConfig, error) {
	return parser.parseYAMLBytesWithOptions(data, "", false)
}

func (parser *Parser) parseYAMLBytesWithOptions(
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
	expandedData, marshalErr := yamlv2.Marshal(raw)
	if marshalErr != nil {
		return nil, fmt.Errorf("failed to marshal normalized config input: %w", marshalErr)
	}

	cfg, err := parseRouterConfigPayload(expandedData, raw, parser.connectionCompiler)
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
		validateV03DocumentBoundary,
		validateCanonicalAuthoringFields,
		rejectUnsupportedTopLevelFields,
		rejectBootstrapSecretLiterals,
		validateBootstrapFieldNames,
		rejectRemovedStructureFields,
		rejectRemovedTaxonomyLegacyFields,
		rejectRemovedDecisionToolFields,
		rejectDecisionLearningFields,
		rejectUnsupportedRouterLearningFields,
		rejectRemovedRouterFields,
	}
	for _, validate := range validators {
		if err := validate(raw); err != nil {
			return err
		}
	}
	return nil
}

func rejectRemovedRouterFields(raw map[string]interface{}) error {
	router := nestedMapAt(nestedStringMap(raw["global"]), "router")
	if _, exists := router["skip_processing"]; exists {
		return fmt.Errorf("global.router.skip_processing has been removed because inference paths cannot bypass Router access enforcement")
	}
	return nil
}

func validateV03DocumentBoundary(raw map[string]interface{}) error {
	version, ok := raw["version"].(string)
	if !ok || version != "v0.3" {
		return fmt.Errorf("version must be v0.3 and use the current providers/routing/recipes/entrypoints authoring schema")
	}
	return nil
}

func parseRawConfigMap(data []byte) (map[string]interface{}, error) {
	raw, unmarshalErr := ParseYAML12Mapping(data)
	if unmarshalErr != nil {
		logging.ComponentDebugEvent("config", "config_yaml_map_parse_failed", map[string]interface{}{
			"error": unmarshalErr.Error(),
		})
		return nil, fmt.Errorf("failed to parse config file: %w", unmarshalErr)
	}
	return raw, nil
}

func rejectUnsupportedTopLevelFields(raw map[string]interface{}) error {
	if unsupported := unsupportedTopLevelConfigFields(raw); len(unsupported) > 0 {
		return canonicalConfigRequiredError(raw)
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
	for _, routing := range rawRoutingDocuments(raw) {
		signals := nestedStringMap(routing.document["signals"])
		if _, ok := signals["category_kb"]; ok {
			return fmt.Errorf(
				"%s.signals.category_kb is no longer supported; use global.model_catalog.kbs[] plus %s.signals.kb[]",
				routing.prefix,
				routing.prefix,
			)
		}
		if _, ok := signals["taxonomy"]; ok {
			return fmt.Errorf(
				"%s.signals.taxonomy is no longer supported; use %s.signals.kb[]",
				routing.prefix,
				routing.prefix,
			)
		}
	}
	global := nestedStringMap(raw["global"])
	modelCatalog := nestedStringMap(global["model_catalog"])
	if _, ok := modelCatalog["classifiers"]; ok {
		return fmt.Errorf(
			"global.model_catalog.classifiers is no longer supported; use global.model_catalog.kbs[]",
		)
	}
	return nil
}

func rejectRemovedDecisionToolFields(raw map[string]interface{}) error {
	removed := make([]string, 0)
	for _, routing := range rawRoutingDocuments(raw) {
		decisions, _ := routing.document["decisions"].([]interface{})
		for index, rawDecision := range decisions {
			decision := nestedStringMap(rawDecision)
			for _, field := range []string{"tool_scope", "allow_tools", "block_tools"} {
				if _, ok := decision[field]; ok {
					removed = append(removed, fmt.Sprintf("%s.decisions[%d].%s", routing.prefix, index, field))
				}
			}
		}
	}
	if len(removed) == 0 {
		return nil
	}

	return fmt.Errorf(
		"removed config fields are no longer supported: %s; use recipes[].routing.decisions[].plugins[type=tools].configuration",
		strings.Join(removed, ", "),
	)
}

func rejectDecisionLearningFields(raw map[string]interface{}) error {
	for _, routing := range rawRoutingDocuments(raw) {
		decisions, _ := routing.document["decisions"].([]interface{})
		for index, rawDecision := range decisions {
			decision := nestedStringMap(rawDecision)
			if err := rejectRemovedDecisionLearningFields(routing.prefix, index, nestedStringMap(decision["algorithm"])); err != nil {
				return err
			}
		}
	}
	return nil
}

func rejectRemovedDecisionLearningFields(prefix string, index int, algorithm map[string]interface{}) error {
	algorithmType := strings.TrimSpace(fmt.Sprint(algorithm["type"]))
	if err := removedLearningAlgorithmTypeError(prefix, index, algorithmType); err != nil {
		return err
	}
	if _, ok := algorithm["session_aware"]; ok {
		return fmt.Errorf(
			"%s.decisions[%d].algorithm.session_aware is no longer supported; remove algorithm.session_aware and configure global.router.learning.protection plus recipes[].routing.decisions[].adaptations when this decision needs apply/observe/bypass control",
			prefix,
			index,
		)
	}
	for field, adaptation := range removedDecisionLearningBlocks() {
		if _, ok := algorithm[field]; ok {
			return fmt.Errorf(
				"%s.decisions[%d].algorithm.%s has moved to %s; remove the decision-local learning algorithm block",
				prefix,
				index,
				field,
				adaptation,
			)
		}
	}
	return nil
}

func removedLearningAlgorithmTypeError(prefix string, index int, algorithmType string) error {
	if algorithmType == "session_aware" {
		return fmt.Errorf(
			"%s.decisions[%d].algorithm.type=session_aware is no longer supported; remove algorithm.type=session_aware and enable global.router.learning.protection",
			prefix,
			index,
		)
	}
	if target, ok := removedDecisionLearningBlocks()[algorithmType]; ok {
		return fmt.Errorf(
			"%s.decisions[%d].algorithm.type=%s has moved to %s; remove algorithm.type=%s and choose a request-time base algorithm only when needed",
			prefix,
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
	for _, routing := range rawRoutingDocuments(raw) {
		decisions, _ := routing.document["decisions"].([]interface{})
		for index, rawDecision := range decisions {
			decision := nestedStringMap(rawDecision)
			adaptations := nestedStringMap(decision["adaptations"])
			if len(adaptations) == 0 {
				continue
			}
			prefix := fmt.Sprintf("%s.decisions[%d].adaptations", routing.prefix, index)
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

func parseRouterConfigPayload(
	data []byte,
	raw map[string]interface{},
	connectionCompiler modelauthoring.ConnectionCompiler,
) (*RouterConfig, error) {
	if !isCanonicalConfig(raw) {
		return nil, canonicalConfigRequiredError(raw)
	}
	return parseCanonicalConfigPayload(data, raw, connectionCompiler)
}

func parseCanonicalConfigPayload(
	data []byte,
	raw map[string]interface{},
	connectionCompiler modelauthoring.ConnectionCompiler,
) (*RouterConfig, error) {
	canonical := &CanonicalConfig{}
	if unmarshalErr := yamlv2.UnmarshalStrict(data, canonical); unmarshalErr != nil {
		logging.ComponentDebugEvent("config", "config_canonical_parse_failed", map[string]interface{}{
			"error": unmarshalErr.Error(),
		})
		return nil, fmt.Errorf("failed to parse canonical config file: %w", unmarshalErr)
	}
	if err := attachCanonicalGlobalOverride(raw, canonical); err != nil {
		return nil, err
	}

	cfg, err := normalizeCanonicalConfig(canonical, connectionCompiler)
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
		"config file must use the current v0.3 version/listeners/providers/routing/recipes/entrypoints/global authoring schema; %s",
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
	if err := cfg.ValidateRuntimeBootstrap(); err != nil {
		logging.ComponentDebugEvent("config", "config_bootstrap_validation_failed", map[string]interface{}{
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

func removedStructureFields(raw map[string]interface{}) []string {
	fields := make([]string, 0)
	for _, routing := range rawRoutingDocuments(raw) {
		signals := nestedStringMap(routing.document["signals"])
		structureRules, _ := signals["structure"].([]interface{})
		for index, rawRule := range structureRules {
			rule := nestedStringMap(rawRule)
			feature := nestedStringMap(rule["feature"])
			if _, ok := feature["normalize_by"]; ok {
				fields = append(fields, fmt.Sprintf("%s.signals.structure[%d].feature.normalize_by", routing.prefix, index))
			}
		}
	}
	return fields
}

type rawRoutingDocument struct {
	prefix   string
	document map[string]interface{}
}

// rawRoutingDocuments is the single path-aware traversal for routing authoring
// values. It covers the top-level v0.3 routing profile and reusable Recipe
// fragments. Unsupported top-level layouts are rejected before this traversal.
func rawRoutingDocuments(raw map[string]interface{}) []rawRoutingDocument {
	documents := make([]rawRoutingDocument, 0, 2)
	if routing, ok := raw["routing"]; ok {
		documents = append(documents, rawRoutingDocument{prefix: "routing", document: nestedStringMap(routing)})
	}
	if recipes, ok := raw["recipes"].([]interface{}); ok {
		for index, rawRecipe := range recipes {
			recipe := nestedStringMap(rawRecipe)
			documents = append(documents, rawRoutingDocument{
				prefix:   fmt.Sprintf("recipes[%d].routing", index),
				document: nestedStringMap(recipe["routing"]),
			})
		}
	}
	return documents
}

func unsupportedTopLevelConfigFields(raw map[string]interface{}) []string {
	allowed := map[string]bool{
		"version":     true,
		"listeners":   true,
		"providers":   true,
		"routing":     true,
		"recipes":     true,
		"entrypoints": true,
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
