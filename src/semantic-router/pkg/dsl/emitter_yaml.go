package dsl

import (
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
	return yaml.Marshal(cfg)
}

// EmitUserYAML emits YAML in the user-friendly nested format (signals/providers)
// that matches the config.yaml format used by vllm-serve.
// This is the inverse of normalizeYAML.
func EmitUserYAML(cfg *config.RouterConfig) ([]byte, error) {
	// First marshal to flat YAML, then restructure via map manipulation.
	flatBytes, err := yaml.Marshal(cfg)
	if err != nil {
		return nil, err
	}
	var raw map[string]interface{}
	if err := yaml.Unmarshal(flatBytes, &raw); err != nil {
		return nil, err
	}

	denormalizeSignals(raw)
	denormalizeProviders(raw)
	pruneZeroValueInfra(raw)

	return yaml.Marshal(raw)
}

// denormalizeSignals groups flat signal keys into a nested "signals" section.
func denormalizeSignals(raw map[string]interface{}) {
	signalKeyMap := map[string]string{
		"keyword_rules":       "keywords",
		"embedding_rules":     "embeddings",
		"categories":          "domains",
		"fact_check_rules":    "fact_check",
		"user_feedback_rules": "user_feedbacks",
		"preference_rules":    "preferences",
		"language_rules":      "language",
		"context_rules":       "context",
		"complexity_rules":    "complexity",
		"modality_rules":      "modality",
		"role_bindings":       "authz",
		"jailbreak":           "jailbreak",
		"pii":                 "pii",
	}

	signals := make(map[string]interface{})
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

// denormalizeProviders groups vllm_endpoints + model_config into a nested "providers" section
// and reconstructs the user-friendly "models" list.
func denormalizeProviders(raw map[string]interface{}) {
	providers := make(map[string]interface{})

	// Reconstruct models from vllm_endpoints + model_config
	endpoints, _ := raw["vllm_endpoints"].([]interface{})
	modelConfigRaw, _ := raw["model_config"].(map[string]interface{})

	if len(endpoints) > 0 {
		models := buildModelsFromEndpoints(endpoints, modelConfigRaw)
		if len(models) > 0 {
			providers["models"] = models
		}
		delete(raw, "vllm_endpoints")
	}
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
func buildModelsFromEndpoints(endpoints []interface{}, modelConfigRaw map[string]interface{}) []interface{} {
	// Group endpoints by model name (extracted from endpoint name pattern: modelName_epName)
	type endpointInfo struct {
		name     string
		address  string
		port     int
		weight   interface{}
		protocol string
		epType   string
		apiKey   string
	}

	type modelEntry struct {
		name      string
		endpoints []endpointInfo
		config    map[string]interface{}
	}

	modelMap := make(map[string]*modelEntry)
	var modelOrder []string

	for _, ep := range endpoints {
		epMap, ok := ep.(map[string]interface{})
		if !ok {
			continue
		}
		fullName, _ := epMap["name"].(string)
		address, _ := epMap["address"].(string)
		port := toInt(epMap["port"])
		weight := epMap["weight"]
		protocol, _ := epMap["protocol"].(string)
		epType, _ := epMap["type"].(string)
		apiKey, _ := epMap["api_key"].(string)
		model, _ := epMap["model"].(string)

		// Determine model name and original endpoint name.
		// normalizeYAML sets name = modelName + "_" + epName
		// and stores model = modelName.
		modelName := model
		epName := "vllm_endpoint"
		if modelName == "" {
			// Fallback: try to extract from fullName pattern modelName_epName
			modelName, epName = splitEndpointName(fullName)
		} else if strings.HasPrefix(fullName, modelName+"_") {
			epName = fullName[len(modelName)+1:]
		}

		if modelName == "" {
			continue
		}

		me, exists := modelMap[modelName]
		if !exists {
			me = &modelEntry{name: modelName}
			modelMap[modelName] = me
			modelOrder = append(modelOrder, modelName)
		}
		me.endpoints = append(me.endpoints, endpointInfo{
			name:     epName,
			address:  address,
			port:     port,
			weight:   weight,
			protocol: protocol,
			epType:   epType,
			apiKey:   apiKey,
		})
	}

	// Merge model_config data
	for modelName, mcRaw := range modelConfigRaw {
		mc, ok := mcRaw.(map[string]interface{})
		if !ok {
			continue
		}
		me, exists := modelMap[modelName]
		if !exists {
			me = &modelEntry{name: modelName}
			modelMap[modelName] = me
			modelOrder = append(modelOrder, modelName)
		}
		me.config = mc
	}

	// Build output
	var models []interface{}
	for _, modelName := range modelOrder {
		me := modelMap[modelName]
		m := map[string]interface{}{
			"name": me.name,
		}

		// Add model_config fields (reasoning_family, param_size, etc.)
		if me.config != nil {
			for k, v := range me.config {
				if k == "preferred_endpoints" {
					continue // Don't emit this in nested format
				}
				if !isZeroValue(v) {
					m[k] = v
				}
			}
		}

		// Build endpoints list
		if len(me.endpoints) > 0 {
			var epList []interface{}
			for _, ep := range me.endpoints {
				epOut := map[string]interface{}{
					"name": ep.name,
				}
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
			m["endpoints"] = epList
		}

		models = append(models, m)
	}
	return models
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
// These are config sections not representable in DSL (embedding_models, bert_model, etc.)
func pruneZeroValueInfra(raw map[string]interface{}) {
	infraKeys := []string{
		"embedding_models", "bert_model", "classifier",
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
func isEmptySlice(v interface{}) bool {
	if v == nil {
		return true
	}
	if s, ok := v.([]interface{}); ok {
		return len(s) == 0
	}
	return false
}

// isZeroValue returns true for Go zero values after YAML round-trip.
func isZeroValue(v interface{}) bool {
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
	case []interface{}:
		return len(val) == 0
	case map[string]interface{}:
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

// toInt converts interface{} to int for port numbers.
func toInt(v interface{}) int {
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

// EmitUserYAMLOrdered emits YAML in user-friendly format with a controlled key order
// matching the canonical config.yaml layout.
func EmitUserYAMLOrdered(cfg *config.RouterConfig) ([]byte, error) {
	flatBytes, err := yaml.Marshal(cfg)
	if err != nil {
		return nil, err
	}
	var raw map[string]interface{}
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
func buildOrderedMap(raw map[string]interface{}) *yaml.Node {
	// Canonical key order for top-level config
	keyOrder := []string{
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
func addKeyValue(mapNode *yaml.Node, key string, value interface{}) {
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

	spec := map[string]interface{}{
		"config": configSpec,
	}
	if len(vllmEndpoints) > 0 {
		spec["vllmEndpoints"] = vllmEndpoints
	}

	crd := map[string]interface{}{
		"apiVersion": "vllm.ai/v1alpha1",
		"kind":       "SemanticRouter",
		"metadata": map[string]interface{}{
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
	var raw map[string]interface{}
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
//   - bert_model, classifier, prompt_guard, semantic_cache, tools, observability, api
//   - Signal rules not in ConfigSpec are included as extra keys for completeness
func buildCRDConfigSpec(cfg *config.RouterConfig) map[string]interface{} {
	// Marshal the full RouterConfig to a flat map first
	flatBytes, _ := yaml.Marshal(cfg)
	var flat map[string]interface{}
	_ = yaml.Unmarshal(flatBytes, &flat)

	configSpec := make(map[string]interface{})

	// --- Fields that belong in ConfigSpec (per semanticrouter_types.go) ---

	// Routing logic
	moveKey(flat, configSpec, "decisions")
	moveKey(flat, configSpec, "strategy")
	moveKey(flat, configSpec, "complexity_rules")
	moveKey(flat, configSpec, "reasoning_families")
	moveKey(flat, configSpec, "default_reasoning_effort")
	moveKey(flat, configSpec, "default_model")

	// Infrastructure configs that ConfigSpec supports
	moveKey(flat, configSpec, "bert_model")
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
		"fact_check_rules", "user_feedback_rules", "preference_rules",
		"language_rules", "context_rules", "modality_rules",
		"role_bindings", "jailbreak", "pii",
	}
	for _, key := range signalKeys {
		moveKey(flat, configSpec, key)
	}

	return configSpec
}

// buildCRDVLLMEndpoints converts flat vllm_endpoints + model_config into the
// CRD's VLLMEndpointSpec format with K8s-native backend references.
func buildCRDVLLMEndpoints(cfg *config.RouterConfig) []map[string]interface{} {
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

	var endpoints []map[string]interface{}
	for _, ep := range cfg.VLLMEndpoints {
		modelName := ep.Model
		if modelName == "" {
			// Try to extract from endpoint name pattern: modelName_epName
			modelName, _ = splitEndpointName(ep.Name)
		}

		entry := map[string]interface{}{
			"name":  ep.Name,
			"model": modelName,
		}

		// Add reasoning family from model_config if available
		if rf, ok := reasoningFamilyLookup[modelName]; ok {
			entry["reasoningFamily"] = rf
		}

		// Build backend spec: use type=service with the address/port
		backend := map[string]interface{}{
			"type": "service",
			"service": map[string]interface{}{
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
func moveKey(src, dst map[string]interface{}, key string) {
	if v, ok := src[key]; ok {
		if !isZeroValue(v) {
			dst[key] = v
		}
		delete(src, key)
	}
}

// EmitHelm wraps a RouterConfig into a Helm values.yaml structure compatible
// with the semantic-router Helm chart (deploy/helm/semantic-router/).
//
// The chart's ConfigMap template renders `.Values.config` as the routing config
// (config.yaml), so we nest the RouterConfig under a `config:` key and prune
// infrastructure zero values just like EmitUserYAML does for clean output.
func EmitHelm(cfg *config.RouterConfig) ([]byte, error) {
	flatBytes, err := yaml.Marshal(cfg)
	if err != nil {
		return nil, err
	}
	var raw map[string]interface{}
	if err := yaml.Unmarshal(flatBytes, &raw); err != nil {
		return nil, err
	}

	// Remove zero-value infrastructure sections
	pruneZeroValueInfra(raw)

	// Remove strategy if empty
	if v, ok := raw["strategy"]; ok {
		if s, ok := v.(string); ok && s == "" {
			delete(raw, "strategy")
		}
	}

	// Wrap in Helm values structure
	values := map[string]interface{}{
		"config": raw,
	}

	// Build ordered output
	doc := &yaml.Node{Kind: yaml.DocumentNode}
	mapNode := &yaml.Node{Kind: yaml.MappingNode}
	addKeyValue(mapNode, "config", values["config"])
	doc.Content = append(doc.Content, mapNode)

	return yaml.Marshal(doc)
}

// pruneZeroValues recursively removes zero-value entries from a nested map.
func pruneZeroValues(m map[string]interface{}) {
	for k, v := range m {
		switch val := v.(type) {
		case map[string]interface{}:
			pruneZeroValues(val)
			if len(val) == 0 {
				delete(m, k)
			}
		case []interface{}:
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
