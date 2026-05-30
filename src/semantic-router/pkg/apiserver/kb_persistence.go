//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"os"
	"reflect"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func knowledgeBaseOverrideYAML(existingData []byte, kbs []config.KnowledgeBaseConfig) ([]byte, error) {
	doc, err := parseYAMLDocument(existingData)
	if err != nil {
		return nil, fmt.Errorf("parse config.yaml: %w", err)
	}
	root, err := documentMappingNode(doc)
	if err != nil {
		return nil, err
	}

	if !shouldPersistKnowledgeBaseOverride(kbs) {
		deleteNestedMappingValue(root, "global", "model_catalog", "kbs")
		return marshalYAMLDocument(doc)
	}

	kbNode, err := yamlNodeFromValue(kbs)
	if err != nil {
		return nil, fmt.Errorf("encode kb override: %w", err)
	}
	setNestedMappingValue(root, kbNode, "global", "model_catalog", "kbs")
	return marshalYAMLDocument(doc)
}

func shouldPersistKnowledgeBaseOverride(kbs []config.KnowledgeBaseConfig) bool {
	defaults := normalizeKnowledgeBaseSlice(config.DefaultCanonicalGlobal().ModelCatalog.KBs)
	current := normalizeKnowledgeBaseSlice(kbs)
	return !reflect.DeepEqual(defaults, current)
}

func normalizeKnowledgeBaseSlice(kbs []config.KnowledgeBaseConfig) []config.KnowledgeBaseConfig {
	normalized := make([]config.KnowledgeBaseConfig, 0, len(kbs))
	for _, kb := range kbs {
		copyKB := kb
		copyKB.Source.Path = cleanKnowledgeBaseSourcePath(copyKB.Source.Path)
		copyKB.Source.Manifest = strings.TrimSpace(copyKB.Source.Manifest)
		copyKB.LabelThresholds = cloneLabelThresholds(copyKB.LabelThresholds)
		copyKB.Groups = cloneKnowledgeBaseGroups(copyKB.Groups)
		copyKB.Metrics = cloneKnowledgeBaseMetrics(copyKB.Metrics)
		normalized = append(normalized, copyKB)
	}
	sortKnowledgeBaseConfigs(normalized)
	return normalized
}

func sortKnowledgeBaseConfigs(kbs []config.KnowledgeBaseConfig) {
	sort.Slice(kbs, func(i, j int) bool { return kbs[i].Name < kbs[j].Name })
}

func validateConfigWithBaseDir(baseDir string, yamlBytes []byte) (*config.RouterConfig, error) {
	tempFile, err := os.CreateTemp(baseDir, ".kb-*.yaml")
	if err != nil {
		return nil, err
	}
	tempPath := tempFile.Name()
	defer func() {
		_ = tempFile.Close()
		_ = os.Remove(tempPath)
	}()
	if _, err := tempFile.Write(yamlBytes); err != nil {
		return nil, err
	}
	if err := tempFile.Close(); err != nil {
		return nil, err
	}
	return config.Parse(tempPath)
}

func persistConfigAndSync(
	s *ClassificationAPIServer,
	paths configPersistencePaths,
	previousData []byte,
	yamlBytes []byte,
	newCfg *config.RouterConfig,
) error {
	if err := writeConfigAtomically(paths.sourcePath, yamlBytes); err != nil {
		return err
	}
	if paths.usesRuntimeOverride() {
		if _, err := runtimeConfigSyncRunner(paths.sourcePath); err != nil {
			_ = writeConfigAtomically(paths.sourcePath, previousData)
			_, _ = runtimeConfigSyncRunner(paths.sourcePath)
			return err
		}
	}
	s.publishConfigMutation(newCfg)
	return nil
}

func yamlNodeFromValue(value any) (*yaml.Node, error) {
	data, err := yaml.Marshal(value)
	if err != nil {
		return nil, err
	}
	doc, err := parseYAMLDocument(data)
	if err != nil {
		return nil, err
	}
	root, err := documentMappingNode(doc)
	if err == nil {
		return root, nil
	}
	if len(doc.Content) > 0 {
		return doc.Content[0], nil
	}
	return nil, fmt.Errorf("failed to decode YAML node")
}

func parseYAMLDocument(data []byte) (*yaml.Node, error) {
	var doc yaml.Node
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return nil, err
	}
	if len(doc.Content) == 0 {
		doc.Content = []*yaml.Node{{Kind: yaml.MappingNode, Tag: "!!map"}}
	}
	return &doc, nil
}

func documentMappingNode(doc *yaml.Node) (*yaml.Node, error) {
	if doc == nil {
		return nil, fmt.Errorf("yaml document is nil")
	}
	if doc.Kind == 0 && len(doc.Content) == 0 {
		doc.Kind = yaml.DocumentNode
		doc.Content = []*yaml.Node{{Kind: yaml.MappingNode, Tag: "!!map"}}
	}
	if doc.Kind != yaml.DocumentNode {
		return nil, fmt.Errorf("expected YAML document node, got kind %d", doc.Kind)
	}
	if len(doc.Content) == 0 {
		doc.Content = []*yaml.Node{{Kind: yaml.MappingNode, Tag: "!!map"}}
	}
	root := doc.Content[0]
	if root.Kind == 0 {
		root.Kind = yaml.MappingNode
		root.Tag = "!!map"
	}
	if root.Kind != yaml.MappingNode {
		return nil, fmt.Errorf("expected YAML root mapping, got kind %d", root.Kind)
	}
	return root, nil
}

func mappingValueNode(mapping *yaml.Node, key string) *yaml.Node {
	if mapping == nil || mapping.Kind != yaml.MappingNode {
		return nil
	}
	for i := 0; i+1 < len(mapping.Content); i += 2 {
		if mapping.Content[i].Value == key {
			return mapping.Content[i+1]
		}
	}
	return nil
}

func setMappingValueNode(mapping *yaml.Node, key string, value *yaml.Node) {
	for i := 0; i+1 < len(mapping.Content); i += 2 {
		if mapping.Content[i].Value == key {
			mapping.Content[i+1] = cloneYAMLNode(value)
			return
		}
	}
	mapping.Content = append(mapping.Content,
		&yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: key},
		cloneYAMLNode(value),
	)
}

func deleteMappingValueNode(mapping *yaml.Node, key string) {
	if mapping == nil || mapping.Kind != yaml.MappingNode {
		return
	}
	for i := 0; i+1 < len(mapping.Content); i += 2 {
		if mapping.Content[i].Value == key {
			mapping.Content = append(mapping.Content[:i], mapping.Content[i+2:]...)
			return
		}
	}
}

func cloneYAMLNode(node *yaml.Node) *yaml.Node {
	if node == nil {
		return nil
	}
	clone := *node
	if len(node.Content) > 0 {
		clone.Content = make([]*yaml.Node, len(node.Content))
		for i, child := range node.Content {
			clone.Content[i] = cloneYAMLNode(child)
		}
	}
	return &clone
}

func ensureNestedMappingNode(root *yaml.Node, keys ...string) *yaml.Node {
	current := root
	for _, key := range keys {
		next := mappingValueNode(current, key)
		if next == nil || next.Kind != yaml.MappingNode {
			next = &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
			setMappingValueNode(current, key, next)
			next = mappingValueNode(current, key)
		}
		current = next
	}
	return current
}

func setNestedMappingValue(root *yaml.Node, value *yaml.Node, keys ...string) {
	if len(keys) == 0 {
		return
	}
	parent := ensureNestedMappingNode(root, keys[:len(keys)-1]...)
	setMappingValueNode(parent, keys[len(keys)-1], value)
}

func deleteNestedMappingValue(root *yaml.Node, keys ...string) {
	if len(keys) == 0 {
		return
	}
	current := root
	for _, key := range keys[:len(keys)-1] {
		current = mappingValueNode(current, key)
		if current == nil || current.Kind != yaml.MappingNode {
			return
		}
	}
	deleteMappingValueNode(current, keys[len(keys)-1])
}

func marshalYAMLDocument(doc *yaml.Node) ([]byte, error) {
	return yaml.Marshal(doc)
}
