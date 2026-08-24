package dsl

import (
	"bytes"
	"fmt"
	"sort"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// EmitYAML compiles a DSL source string and emits one model-free Recipe
// document.
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

// EmitYAMLFromConfig emits the only YAML value owned by the DSL: one
// model-free Recipe document. A complete standalone manifest is produced only
// by explicitly merging this value into one existing Recipe with --base.
func EmitYAMLFromConfig(cfg *config.RouterConfig) ([]byte, error) {
	return EmitRoutingYAMLFromConfig(cfg)
}

// MergeRoutingIntoBase replaces the document of the one Recipe in a complete
// v0.4 manifest. Selecting among multiple Recipes is an explicit control-plane
// operation and is therefore rejected by this narrow CLI helper.
func MergeRoutingIntoBase(cfg *config.RouterConfig, baseYAML []byte) ([]byte, error) {
	var base map[string]interface{}
	if err := yaml.Unmarshal(baseYAML, &base); err != nil {
		return nil, fmt.Errorf("failed to parse base YAML: %w", err)
	}

	canonicalDocument, err := canonicalRecipeDocument(cfg)
	if err != nil {
		return nil, fmt.Errorf("select Recipe document: %w", err)
	}
	documentBytes, err := yaml.Marshal(canonicalDocument)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal Recipe document: %w", err)
	}
	var document interface{}
	if err := yaml.Unmarshal(documentBytes, &document); err != nil {
		return nil, fmt.Errorf("failed to re-parse Recipe document: %w", err)
	}
	if err := replaceSingleRecipeDocument(base, document); err != nil {
		return nil, err
	}

	doc := &yaml.Node{Kind: yaml.DocumentNode}
	mapNode := &yaml.Node{Kind: yaml.MappingNode}
	canonicalOrder := []string{"version", "listeners", "models", "recipes", "entrypoints", "global"}
	added := make(map[string]bool)
	for _, key := range canonicalOrder {
		if value, ok := base[key]; ok {
			addKeyValue(mapNode, key, value)
			added[key] = true
		}
	}
	remaining := make([]string, 0, len(base)-len(added))
	for key := range base {
		if !added[key] {
			remaining = append(remaining, key)
		}
	}
	sort.Strings(remaining)
	for _, key := range remaining {
		addKeyValue(mapNode, key, base[key])
	}
	doc.Content = append(doc.Content, mapNode)
	return marshalYAMLIndent2(doc)
}

func replaceSingleRecipeDocument(base map[string]interface{}, document interface{}) error {
	recipes, ok := base["recipes"].([]interface{})
	if !ok || len(recipes) != 1 {
		return fmt.Errorf("base manifest must contain exactly one Recipe")
	}
	recipe, ok := recipes[0].(map[string]interface{})
	if !ok {
		return fmt.Errorf("base manifest Recipe must be a mapping")
	}
	preserveBaseDecisionField(document, recipe["document"], "adaptations")
	recipe["document"] = document
	return nil
}

func addKeyValue(mapNode *yaml.Node, key string, value interface{}) {
	keyNode := &yaml.Node{Kind: yaml.ScalarNode, Value: key, Tag: "!!str"}
	valueNode := &yaml.Node{}
	valueBytes, _ := yaml.Marshal(value)
	_ = yaml.Unmarshal(valueBytes, valueNode)
	if valueNode.Kind == yaml.DocumentNode && len(valueNode.Content) > 0 {
		valueNode = valueNode.Content[0]
	}
	mapNode.Content = append(mapNode.Content, keyNode, valueNode)
}

func marshalYAMLIndent2(node *yaml.Node) ([]byte, error) {
	var output bytes.Buffer
	encoder := yaml.NewEncoder(&output)
	encoder.SetIndent(2)
	if err := encoder.Encode(node); err != nil {
		return nil, err
	}
	if err := encoder.Close(); err != nil {
		return nil, err
	}
	return output.Bytes(), nil
}
