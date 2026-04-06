package dsl

import (
	"bytes"
	"fmt"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// EmitYAML compiles a DSL source string and emits YAML bytes.
func EmitYAML(input string) ([]byte, []error) {
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		return nil, errs
	}
	yamlBytes, err := EmitYAMLFromConfig(cfg)
	if err != nil {
		return nil, []error{err}
	}
	return yamlBytes, nil
}

// EmitYAMLFromConfig marshals a RouterConfig to YAML bytes.
func EmitYAMLFromConfig(cfg *config.RouterConfig) ([]byte, error) {
	if cfg != nil && len(cfg.KnowledgeBases) > 0 {
		canonical := config.CanonicalConfigFromRouterConfig(cfg)
		return yaml.Marshal(canonical)
	}
	return yaml.Marshal(cfg)
}

// EmitUserYAML emits YAML in the user-friendly nested format (signals/providers)
// that matches the config.yaml format used by vllm-serve.
// This is the inverse of normalizeYAML.
func EmitUserYAML(cfg *config.RouterConfig) ([]byte, error) {
	if cfg != nil && len(cfg.KnowledgeBases) > 0 {
		canonical := config.CanonicalConfigFromRouterConfig(cfg)
		return yaml.Marshal(canonical)
	}
	// First marshal to flat YAML, then restructure via map manipulation.
	flatBytes, err := yaml.Marshal(cfg)
	if err != nil {
		return nil, err
	}
	var raw YAMLObject
	if err := yaml.Unmarshal(flatBytes, &raw); err != nil {
		return nil, err
	}

	denormalizeSignals(raw)
	denormalizeProviders(raw)
	pruneZeroValueInfra(raw)

	return yaml.Marshal(raw)
}

// denormalizeSignals groups flat signal keys into a nested "signals" section.
func denormalizeSignals(raw YAMLObject) {
	signalKeyMap := map[string]string{
		"keyword_rules":       "keywords",
		"embedding_rules":     "embeddings",
		"categories":          "domains",
		"fact_check_rules":    "fact_check",
		"user_feedback_rules": "user_feedbacks",
		"reask_rules":         "reasks",
		"preference_rules":    "preferences",
		"language_rules":      "language",
		"context_rules":       "context",
		"structure_rules":     "structure",
		"complexity_rules":    "complexity",
		"modality_rules":      "modality",
		"role_bindings":       "authz",
		"jailbreak":           "jailbreak",
		"pii":                 "pii",
		"kb":                  "kb",
	}

	signals := make(YAMLObject)
	for flatKey, nestedKey := range signalKeyMap {
		if v, ok := raw[flatKey]; ok {
			if !isEmptySlice(v) {
				signals[nestedKey] = v
			}
			delete(raw, flatKey)
		}
	}
	if len(signals) > 0 {
		raw["signals"] = signals
	}
}

type endpointInfo struct {
	name     string
	address  string
	port     int
	weight   YAMLValue
	protocol string
	epType   string
	apiKey   string
}

type modelEntry struct {
	name      string
	endpoints []endpointInfo
	config    YAMLObject
}

// denormalizeProviders groups vllm_endpoints + model_config into a nested "providers" section
// and reconstructs the user-friendly "models" list.
func denormalizeProviders(raw YAMLObject) {
	providers := make(YAMLObject)

	// Reconstruct models from vllm_endpoints + model_config
	endpoints, _ := raw["vllm_endpoints"].(YAMLList)
	modelConfigRaw, _ := raw["model_config"].(YAMLObject)

	if len(endpoints) > 0 {
		models := buildModelsFromEndpoints(endpoints, modelConfigRaw)
		if len(models) > 0 {
			providers["models"] = models
		}
	}
	delete(raw, "vllm_endpoints")
	delete(raw, "model_config")

	// Hoist simple keys into providers
	for _, key := range []string{"default_model", "reasoning_families", "default_reasoning_effort"} {
		if v, ok := raw[key]; ok {
			providers[key] = v
			delete(raw, key)
		}
	}

	if len(providers) > 0 {
		raw["providers"] = providers
	}
}

// buildModelsFromEndpoints reconstructs the nested models list from flat endpoints.
// normalizeYAML creates endpoint names as "{modelName}_{epName}".
// We group by modelName and reconstruct endpoints with address:port → "endpoint" field.
func buildModelsFromEndpoints(endpoints YAMLList, modelConfigRaw YAMLObject) YAMLList {
	modelMap, modelOrder := collectModelEntries(endpoints)
	mergeModelConfig(modelMap, &modelOrder, modelConfigRaw)
	return buildUserModels(modelMap, modelOrder)
}

func collectModelEntries(endpoints YAMLList) (map[string]*modelEntry, []string) {
	modelMap := make(map[string]*modelEntry)
	var modelOrder []string
	for _, ep := range endpoints {
		addEndpointToModelMap(modelMap, &modelOrder, ep)
	}
	return modelMap, modelOrder
}

func addEndpointToModelMap(modelMap map[string]*modelEntry, modelOrder *[]string, ep YAMLValue) {
	epMap, ok := ep.(YAMLObject)
	if !ok {
		return
	}
	fullName, _ := epMap["name"].(string)
	modelName, epName := deriveModelAndEndpointNames(epMap, fullName)
	if modelName == "" {
		return
	}
	me := ensureModelEntry(modelMap, modelOrder, modelName)
	me.endpoints = append(me.endpoints, endpointInfo{
		name:     epName,
		address:  toString(epMap["address"]),
		port:     toInt(epMap["port"]),
		weight:   epMap["weight"],
		protocol: toString(epMap["protocol"]),
		epType:   toString(epMap["type"]),
		apiKey:   toString(epMap["api_key"]),
	})
}

func deriveModelAndEndpointNames(epMap YAMLObject, fullName string) (string, string) {
	modelName := toString(epMap["model"])
	epName := "vllm_endpoint"
	if modelName == "" {
		return splitEndpointName(fullName)
	}
	if strings.HasPrefix(fullName, modelName+"_") {
		epName = fullName[len(modelName)+1:]
	}
	return modelName, epName
}

func ensureModelEntry(modelMap map[string]*modelEntry, modelOrder *[]string, modelName string) *modelEntry {
	if me, ok := modelMap[modelName]; ok {
		return me
	}
	me := &modelEntry{name: modelName}
	modelMap[modelName] = me
	*modelOrder = append(*modelOrder, modelName)
	return me
}

func mergeModelConfig(modelMap map[string]*modelEntry, modelOrder *[]string, modelConfigRaw YAMLObject) {
	for modelName, mcRaw := range modelConfigRaw {
		mc, ok := mcRaw.(YAMLObject)
		if !ok {
			continue
		}
		ensureModelEntry(modelMap, modelOrder, modelName).config = mc
	}
}

func buildUserModels(modelMap map[string]*modelEntry, modelOrder []string) YAMLList {
	models := make(YAMLList, 0, len(modelOrder))
	for _, modelName := range modelOrder {
		models = append(models, buildUserModelEntry(modelMap[modelName]))
	}
	return models
}

func buildUserModelEntry(me *modelEntry) YAMLObject {
	m := YAMLObject{"name": me.name}
	for k, v := range me.config {
		if k == "preferred_endpoints" || isZeroValue(v) {
			continue
		}
		m[k] = v
	}
	if endpoints := buildUserEndpoints(me.endpoints); len(endpoints) > 0 {
		m["endpoints"] = endpoints
	}
	return m
}

func buildUserEndpoints(endpoints []endpointInfo) YAMLList {
	epList := make(YAMLList, 0, len(endpoints))
	for _, ep := range endpoints {
		epOut := YAMLObject{"name": ep.name}
		endpoint := ep.address
		if ep.port != 0 {
			endpoint = fmt.Sprintf("%s:%d", ep.address, ep.port)
		}
		epOut["endpoint"] = endpoint
		if ep.weight != nil && !isZeroValue(ep.weight) {
			epOut["weight"] = ep.weight
		}
		if ep.protocol != "" && ep.protocol != "http" {
			epOut["protocol"] = ep.protocol
		}
		if ep.epType != "" {
			epOut["type"] = ep.epType
		}
		if ep.apiKey != "" {
			epOut["api_key"] = ep.apiKey
		}
		epList = append(epList, epOut)
	}
	return epList
}

// splitEndpointName tries to split "modelName_epName" back into parts.
// Since model names can contain underscores, we try the last "_" segment as epName.
func splitEndpointName(fullName string) (string, string) {
	idx := strings.LastIndex(fullName, "_")
	if idx <= 0 {
		return fullName, ""
	}
	return fullName[:idx], fullName[idx+1:]
}

// pruneZeroValueInfra removes infrastructure config sections that are all zero values.
// These are config sections not representable in DSL (embedding_models, classifiers, etc.)
func pruneZeroValueInfra(raw YAMLObject) {
	infraKeys := []string{
		"embedding_models", "classifier",
		"prompt_guard", "hallucination_mitigation",
		"feedback_detector", "modality_detector",
		"semantic_cache", "memory", "response_api",
		"router_replay", "api", "tools",
		"config_source", "external_models",
		"looper", "model_selection", "vector_store",
		"authz", "ratelimit", "mom_registry",
		"auto_model_name", "include_config_models_in_list",
		"clear_route_cache",
		"image_gen_backends", "provider_profiles",
		"batch_classification",
		"observability",
	}
	for _, key := range infraKeys {
		if v, ok := raw[key]; ok {
			if isZeroValue(v) {
				delete(raw, key)
			}
		}
	}

	// Also remove strategy if empty
	if v, ok := raw["strategy"]; ok {
		if s, ok := v.(string); ok && s == "" {
			delete(raw, "strategy")
		}
	}
}

// isEmptySlice returns true if v is a nil or empty slice.
func isEmptySlice(v YAMLValue) bool {
	if v == nil {
		return true
	}
	if s, ok := v.(YAMLList); ok {
		return len(s) == 0
	}
	return false
}

// isZeroValue returns true for Go zero values after YAML round-trip.
func isZeroValue(v YAMLValue) bool {
	if v == nil {
		return true
	}
	switch val := v.(type) {
	case bool:
		return !val
	case int:
		return val == 0
	case float64:
		return val == 0
	case string:
		return val == ""
	case YAMLList:
		return len(val) == 0
	case YAMLObject:
		if len(val) == 0 {
			return true
		}
		// Check if all values in map are zero
		for _, mv := range val {
			if !isZeroValue(mv) {
				return false
			}
		}
		return true
	}
	return false
}

// toInt converts a YAML scalar to int for port numbers.
func toInt(v YAMLValue) int {
	switch val := v.(type) {
	case int:
		return val
	case float64:
		return int(val)
	case int64:
		return int(val)
	}
	return 0
}

func toString(v YAMLValue) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

// EmitUserYAMLOrdered emits YAML in user-friendly format with a controlled key order
// matching the canonical config.yaml layout.
func EmitUserYAMLOrdered(cfg *config.RouterConfig) ([]byte, error) {
	flatBytes, err := yaml.Marshal(cfg)
	if err != nil {
		return nil, err
	}
	var raw YAMLObject
	if unmarshalErr := yaml.Unmarshal(flatBytes, &raw); unmarshalErr != nil {
		return nil, unmarshalErr
	}

	denormalizeSignals(raw)
	denormalizeProviders(raw)
	pruneZeroValueInfra(raw)

	// Build ordered YAML document
	doc := &yaml.Node{Kind: yaml.DocumentNode}
	mapNode := buildOrderedMap(raw)
	doc.Content = append(doc.Content, mapNode)

	out, err := yaml.Marshal(doc)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// buildOrderedMap creates a yaml.Node mapping with keys in the canonical config.yaml order.
func buildOrderedMap(raw YAMLObject) *yaml.Node {
	// Canonical key order for top-level config
	keyOrder := []string{
		"listeners",
		"signals",
		"decisions",
		"providers",
		"observability",
		"strategy",
	}

	mapNode := &yaml.Node{Kind: yaml.MappingNode}

	// Add keys in order
	added := make(map[string]bool)
	for _, key := range keyOrder {
		if v, ok := raw[key]; ok {
			addKeyValue(mapNode, key, v)
			added[key] = true
		}
	}

	// Add remaining keys alphabetically
	var remaining []string
	for k := range raw {
		if !added[k] {
			remaining = append(remaining, k)
		}
	}
	sort.Strings(remaining)
	for _, key := range remaining {
		addKeyValue(mapNode, key, raw[key])
	}

	return mapNode
}

// addKeyValue adds a key-value pair to a yaml MappingNode.
func addKeyValue(mapNode *yaml.Node, key string, value YAMLValue) {
	keyNode := &yaml.Node{Kind: yaml.ScalarNode, Value: key, Tag: "!!str"}
	valNode := &yaml.Node{}
	valBytes, _ := yaml.Marshal(value)
	_ = yaml.Unmarshal(valBytes, valNode)
	// yaml.Unmarshal wraps in a document node; unwrap it
	if valNode.Kind == yaml.DocumentNode && len(valNode.Content) > 0 {
		valNode = valNode.Content[0]
	}
	mapNode.Content = append(mapNode.Content, keyNode, valNode)
}

// EmitCRD wraps a RouterConfig in a SemanticRouter CRD envelope matching the
// Operator's SemanticRouterSpec structure (vllm.ai/v1alpha1 SemanticRouter).
//
// The mapping is:
//
//	spec.config      ← routing logic (decisions, strategy, reasoning_families,
//	                   complexity_rules, classifier, prompt_guard, semantic_cache, etc.)
//	spec.vllmEndpoints ← model backends converted to K8s-native service references
//
// Signal rules (keyword_rules, embedding_rules, categories, etc.) that the CRD
// does NOT model are preserved in spec.config as extra fields, so the output is
// self-contained and can be used with a ConfigMap-based deployment.
func EmitCRD(cfg *config.RouterConfig, name, namespace string) ([]byte, error) {
	if namespace == "" {
		namespace = "default"
	}

	// Build spec.config from RouterConfig fields
	configSpec := buildCRDConfigSpec(cfg)

	// Build spec.vllmEndpoints from flat vllm_endpoints + model_config
	vllmEndpoints := buildCRDVLLMEndpoints(cfg)

	spec := YAMLObject{
		"config": configSpec,
	}
	if len(vllmEndpoints) > 0 {
		spec["vllmEndpoints"] = vllmEndpoints
	}

	crd := YAMLObject{
		"apiVersion": "vllm.ai/v1alpha1",
		"kind":       "SemanticRouter",
		"metadata": YAMLObject{
			"name":      name,
			"namespace": namespace,
		},
		"spec": spec,
	}

	// Marshal, then prune zero-value leaves for a clean output
	rawBytes, err := yaml.Marshal(crd)
	if err != nil {
		return nil, err
	}
	var raw YAMLObject
	if err := yaml.Unmarshal(rawBytes, &raw); err != nil {
		return nil, err
	}
	pruneZeroValues(raw)

	// Build ordered output: apiVersion, kind, metadata, spec
	doc := &yaml.Node{Kind: yaml.DocumentNode}
	mapNode := &yaml.Node{Kind: yaml.MappingNode}
	for _, key := range []string{"apiVersion", "kind", "metadata", "spec"} {
		if v, ok := raw[key]; ok {
			addKeyValue(mapNode, key, v)
		}
	}
	doc.Content = append(doc.Content, mapNode)
	return yaml.Marshal(doc)
}

// buildCRDConfigSpec constructs the CRD spec.config map from RouterConfig.
// It mirrors the Operator's ConfigSpec structure:
//   - decisions, strategy, complexity_rules, reasoning_families, default_reasoning_effort
//   - embedding_models, classifier, prompt_guard, semantic_cache, tools, observability, api
//   - Signal rules not in ConfigSpec are included as extra keys for completeness
func buildCRDConfigSpec(cfg *config.RouterConfig) YAMLObject {
	// Marshal the full RouterConfig to a flat map first
	flatBytes, _ := yaml.Marshal(cfg)
	var flat YAMLObject
	_ = yaml.Unmarshal(flatBytes, &flat)

	configSpec := make(YAMLObject)

	// --- Fields that belong in ConfigSpec (per semanticrouter_types.go) ---

	// Routing logic
	moveKey(flat, configSpec, "decisions")
	moveKey(flat, configSpec, "strategy")
	moveKey(flat, configSpec, "complexity_rules")
	moveKey(flat, configSpec, "reasoning_families")
	moveKey(flat, configSpec, "default_reasoning_effort")
	moveKey(flat, configSpec, "default_model")

	// Infrastructure configs that ConfigSpec supports
	moveKey(flat, configSpec, "embedding_models")
	moveKey(flat, configSpec, "classifier")
	moveKey(flat, configSpec, "prompt_guard")
	moveKey(flat, configSpec, "semantic_cache")
	moveKey(flat, configSpec, "tools")
	moveKey(flat, configSpec, "api")
	moveKey(flat, configSpec, "observability")

	// --- Signal rules: not in ConfigSpec but essential for routing ---
	// Include them in config so the CR is self-contained
	signalKeys := []string{
		"keyword_rules", "embedding_rules", "categories",
		"fact_check_rules", "user_feedback_rules", "reask_rules", "preference_rules",
		"language_rules", "context_rules", "structure_rules",
		"modality_rules", "role_bindings", "jailbreak", "pii",
		"kb",
	}
	for _, key := range signalKeys {
		moveKey(flat, configSpec, key)
	}

	return configSpec
}

// buildCRDVLLMEndpoints converts flat vllm_endpoints + model_config into the
// CRD's VLLMEndpointSpec format with K8s-native backend references.
func buildCRDVLLMEndpoints(cfg *config.RouterConfig) []YAMLObject {
	if len(cfg.VLLMEndpoints) == 0 {
		return nil
	}

	// Build model → reasoning_family lookup from model_config
	reasoningFamilyLookup := make(map[string]string)
	for modelName, mc := range cfg.ModelConfig {
		if mc.ReasoningFamily != "" {
			reasoningFamilyLookup[modelName] = mc.ReasoningFamily
		}
	}

	var endpoints []YAMLObject
	for _, ep := range cfg.VLLMEndpoints {
		modelName := ep.Model
		if modelName == "" {
			// Try to extract from endpoint name pattern: modelName_epName
			modelName, _ = splitEndpointName(ep.Name)
		}

		entry := YAMLObject{
			"name":  ep.Name,
			"model": modelName,
		}

		// Add reasoning family from model_config if available
		if rf, ok := reasoningFamilyLookup[modelName]; ok {
			entry["reasoningFamily"] = rf
		}

		// Build backend spec: use type=service with the address/port
		backend := YAMLObject{
			"type": "service",
			"service": YAMLObject{
				"name": ep.Address,
				"port": ep.Port,
			},
		}
		entry["backend"] = backend

		if ep.Weight > 0 && ep.Weight != 1 {
			entry["weight"] = ep.Weight
		}

		endpoints = append(endpoints, entry)
	}
	return endpoints
}

// moveKey moves a key from src to dst if it exists and is non-zero.
func moveKey(src, dst YAMLObject, key string) {
	if v, ok := src[key]; ok {
		if !isZeroValue(v) {
			dst[key] = v
		}
		delete(src, key)
	}
}

// EmitHelm emits a Helm values fragment that only carries the DSL-owned routing
// surface under the chart's canonical `config:` key.
func EmitHelm(cfg *config.RouterConfig) ([]byte, error) {
	type helmValuesConfig struct {
		Version string                  `yaml:"version"`
		Routing config.CanonicalRouting `yaml:"routing"`
	}

	values := YAMLObject{
		"config": helmValuesConfig{
			Version: "v0.3",
			Routing: config.CanonicalRoutingFromRouterConfig(cfg),
		},
	}

	doc := &yaml.Node{Kind: yaml.DocumentNode}
	mapNode := &yaml.Node{Kind: yaml.MappingNode}
	addKeyValue(mapNode, "config", values["config"])
	doc.Content = append(doc.Content, mapNode)

	return yaml.Marshal(doc)
}

// MergeRoutingIntoBase takes the DSL-compiled RouterConfig and a base YAML
// document (containing version, listeners, providers), replaces the routing
// section with the compiled one, and emits a complete canonical config YAML.
func MergeRoutingIntoBase(cfg *config.RouterConfig, baseYAML []byte) ([]byte, error) {
	var base YAMLObject
	if err := yaml.Unmarshal(baseYAML, &base); err != nil {
		return nil, fmt.Errorf("failed to parse base YAML: %w", err)
	}

	routingBytes, err := yaml.Marshal(config.CanonicalRoutingFromRouterConfig(cfg))
	if err != nil {
		return nil, fmt.Errorf("failed to marshal routing: %w", err)
	}
	var routing YAMLValue
	if err := yaml.Unmarshal(routingBytes, &routing); err != nil {
		return nil, fmt.Errorf("failed to re-parse routing: %w", err)
	}

	base["routing"] = routing

	doc := &yaml.Node{Kind: yaml.DocumentNode}
	mapNode := &yaml.Node{Kind: yaml.MappingNode}
	canonicalOrder := []string{"version", "listeners", "providers", "routing", "global"}
	added := make(map[string]bool)
	for _, key := range canonicalOrder {
		if v, ok := base[key]; ok {
			addKeyValue(mapNode, key, v)
			added[key] = true
		}
	}
	var remaining []string
	for k := range base {
		if !added[k] {
			remaining = append(remaining, k)
		}
	}
	sort.Strings(remaining)
	for _, key := range remaining {
		addKeyValue(mapNode, key, base[key])
	}
	doc.Content = append(doc.Content, mapNode)
	return marshalYAMLIndent2(doc)
}

// marshalYAMLIndent2 encodes a yaml.Node with 2-space indentation to match
// the project's yamllint configuration.
func marshalYAMLIndent2(node *yaml.Node) ([]byte, error) {
	var buf bytes.Buffer
	enc := yaml.NewEncoder(&buf)
	enc.SetIndent(2)
	if err := enc.Encode(node); err != nil {
		return nil, err
	}
	if err := enc.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// pruneZeroValues recursively removes zero-value entries from a nested map.
func pruneZeroValues(m YAMLObject) {
	for k, v := range m {
		switch val := v.(type) {
		case YAMLObject:
			pruneZeroValues(val)
			if len(val) == 0 {
				delete(m, k)
			}
		case YAMLList:
			if len(val) == 0 {
				delete(m, k)
			}
		default:
			if isZeroValue(v) {
				delete(m, k)
			}
		}
	}
}
