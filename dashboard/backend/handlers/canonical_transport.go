package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
	sigsyaml "sigs.k8s.io/yaml"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type routingFragmentDocument struct {
	Routing routerconfig.CanonicalRouting `yaml:"routing"`
}

type setupModeConfig struct {
	Mode bool `yaml:"mode,omitempty"`
}

type setupConfigFile struct {
	routerconfig.CanonicalConfig `yaml:",inline"`
	Setup                        *setupModeConfig `yaml:"setup,omitempty"`
}

func decodeYAMLTaggedBody[T any](reader io.Reader) (T, error) {
	var value T
	data, err := io.ReadAll(reader)
	if err != nil {
		return value, err
	}
	return decodeYAMLTaggedBytes[T](data)
}

func decodeYAMLTaggedBytes[T any](data []byte) (T, error) {
	var value T
	if len(strings.TrimSpace(string(data))) == 0 {
		return value, nil
	}
	if err := yaml.Unmarshal(data, &value); err != nil {
		return value, err
	}
	return value, nil
}

func marshalYAMLTaggedJSON(value any) ([]byte, error) {
	yamlBytes, err := yaml.Marshal(value)
	if err != nil {
		return nil, err
	}
	return sigsyaml.YAMLToJSON(yamlBytes)
}

func marshalYAMLBytes(value any) ([]byte, error) {
	return yaml.Marshal(value)
}

func writeYAMLTaggedJSON(w http.ResponseWriter, value any) error {
	payload, err := marshalYAMLTaggedJSON(value)
	if err != nil {
		return err
	}
	w.Header().Set("Content-Type", "application/json")
	_, err = w.Write(payload)
	return err
}

func rawJSONMessage(value any) (json.RawMessage, error) {
	payload, err := marshalYAMLTaggedJSON(value)
	if err != nil {
		return nil, err
	}
	return json.RawMessage(payload), nil
}

func readCanonicalConfigFile(configPath string) (*routerconfig.CanonicalConfig, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, err
	}
	cfg, err := decodeYAMLTaggedBytes[routerconfig.CanonicalConfig](data)
	if err != nil {
		return nil, err
	}
	return &cfg, nil
}

func readSetupConfigFile(configPath string) (*setupConfigFile, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, err
	}
	cfg, err := decodeYAMLTaggedBytes[setupConfigFile](data)
	if err != nil {
		return nil, err
	}
	return &cfg, nil
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

func mergeMappingNodes(dst, src *yaml.Node) error {
	if dst == nil || src == nil {
		return nil
	}
	if dst.Kind != yaml.MappingNode || src.Kind != yaml.MappingNode {
		return fmt.Errorf("merge requires YAML mapping nodes")
	}
	for i := 0; i+1 < len(src.Content); i += 2 {
		key := src.Content[i].Value
		srcValue := src.Content[i+1]
		dstValue := mappingValueNode(dst, key)
		if dstValue != nil && dstValue.Kind == yaml.MappingNode && srcValue.Kind == yaml.MappingNode {
			if err := mergeMappingNodes(dstValue, srcValue); err != nil {
				return err
			}
			continue
		}
		setMappingValueNode(dst, key, srcValue)
	}
	return nil
}

func marshalYAMLDocument(doc *yaml.Node) ([]byte, error) {
	return yaml.Marshal(doc)
}
