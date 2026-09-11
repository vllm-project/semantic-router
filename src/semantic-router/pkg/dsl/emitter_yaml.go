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
	return yaml.Marshal(config.CanonicalConfigFromRouterConfig(cfg))
}

// EmitUserYAML emits YAML in the user-friendly nested format (signals/providers)
// that matches the config.yaml format used by vllm-serve.
// This is the inverse of normalizeYAML.
func EmitUserYAML(cfg *config.RouterConfig) ([]byte, error) {
	return EmitYAMLFromConfig(cfg)
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

// EmitUserYAMLOrdered emits YAML in user-friendly format with a controlled key order
// matching the canonical config.yaml layout.
func EmitUserYAMLOrdered(cfg *config.RouterConfig) ([]byte, error) {
	return EmitYAMLFromConfig(cfg)
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
//	spec.config        ← canonical routing and supported runtime modules
//	spec.vllmEndpoints ← model backends converted to K8s-native service references
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

// buildCRDConfigSpec constructs the operator's canonical config surface. Routing
// stays under config.routing so DSL emission cannot recreate the retired flat
// defaults and reasoning-family registry.
func buildCRDConfigSpec(cfg *config.RouterConfig) map[string]interface{} {
	canonical := config.CanonicalConfigFromRouterConfig(cfg)
	flatBytes, _ := yaml.Marshal(cfg)
	var flat map[string]interface{}
	_ = yaml.Unmarshal(flatBytes, &flat)

	configSpec := make(map[string]interface{})
	routingBytes, _ := yaml.Marshal(canonical.Routing)
	var routing map[string]interface{}
	_ = yaml.Unmarshal(routingBytes, &routing)
	if !isZeroValue(routing) {
		configSpec["routing"] = routing
	}
	if effort := canonical.Providers.Defaults.DefaultReasoningEffort; effort != "" {
		configSpec["reasoning_effort"] = effort
	}

	// Infrastructure configs that ConfigSpec supports
	moveKey(flat, configSpec, "embedding_models")
	moveKey(flat, configSpec, "classifier")
	moveKey(flat, configSpec, "prompt_guard")
	moveKey(flat, configSpec, "semantic_cache")
	moveKey(flat, configSpec, "tools")
	moveKey(flat, configSpec, "api")
	moveKey(flat, configSpec, "observability")
	return configSpec
}

// buildCRDVLLMEndpoints converts flat vllm_endpoints + model_config into the
// CRD's VLLMEndpointSpec format with K8s-native backend references.
func buildCRDVLLMEndpoints(cfg *config.RouterConfig) []map[string]interface{} {
	if len(cfg.VLLMEndpoints) == 0 {
		return nil
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

		if model, ok := cfg.ModelConfig[modelName]; ok {
			if model.Catalog != "" {
				entry["catalog"] = model.Catalog
			}
			if model.ReasoningFamily != "" {
				entry["reasoning"] = map[string]interface{}{"family": model.ReasoningFamily}
			}
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

// EmitHelm emits a Helm values fragment that only carries the DSL-owned routing
// surface under the chart's canonical `config:` key.
func EmitHelm(cfg *config.RouterConfig) ([]byte, error) {
	type helmValuesConfig struct {
		Version string                  `yaml:"version"`
		Routing config.CanonicalRouting `yaml:"routing"`
	}

	values := map[string]interface{}{
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
	var base map[string]interface{}
	if err := yaml.Unmarshal(baseYAML, &base); err != nil {
		return nil, fmt.Errorf("failed to parse base YAML: %w", err)
	}

	canonical := config.CanonicalConfigFromRouterConfig(cfg)
	routingBytes, err := yaml.Marshal(canonical.Routing)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal routing: %w", err)
	}
	var routing interface{}
	if err := yaml.Unmarshal(routingBytes, &routing); err != nil {
		return nil, fmt.Errorf("failed to re-parse routing: %w", err)
	}
	preserveBaseDecisionField(routing, base["routing"], "adaptations")

	base["routing"] = routing
	if len(canonical.Entrypoints) > 0 {
		base["entrypoints"] = canonical.Entrypoints
	} else {
		delete(base, "entrypoints")
	}
	if len(canonical.Recipes) > 0 {
		base["recipes"] = canonical.Recipes
	} else {
		delete(base, "recipes")
	}

	doc := &yaml.Node{Kind: yaml.DocumentNode}
	mapNode := &yaml.Node{Kind: yaml.MappingNode}
	canonicalOrder := []string{"version", "listeners", "providers", "routing", "entrypoints", "recipes", "global"}
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
