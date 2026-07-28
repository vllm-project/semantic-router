package config

import (
	"reflect"
	"strings"
)

// CanonicalContractSchema is the generated, language-neutral shape of the
// canonical configuration contract.
type CanonicalContractSchema struct {
	SupportedVersions []string                       `json:"supportedVersions"`
	Root              CanonicalSchemaNode            `json:"root"`
	Definitions       map[string]CanonicalSchemaNode `json:"definitions"`
}

// CanonicalSchemaNode describes one node in the canonical configuration tree.
// Map keys are dynamic, while object fields are closed unless Opaque is true.
type CanonicalSchemaNode struct {
	Type   string                         `json:"type"`
	Fields map[string]CanonicalSchemaNode `json:"fields,omitempty"`
	Items  *CanonicalSchemaNode           `json:"items,omitempty"`
	Values *CanonicalSchemaNode           `json:"values,omitempty"`
	Opaque bool                           `json:"opaque,omitempty"`
	Ref    string                         `json:"ref,omitempty"`
}

// BuildCanonicalContractSchema derives the cross-language schema directly
// from the Go canonical config types.
func BuildCanonicalContractSchema() CanonicalContractSchema {
	builder := canonicalSchemaBuilder{
		definitions: make(map[string]CanonicalSchemaNode),
		building:    make(map[string]bool),
	}
	return CanonicalContractSchema{
		SupportedVersions: SupportedCanonicalVersions(),
		Root:              builder.buildNode(reflect.TypeOf(CanonicalConfig{})),
		Definitions:       builder.definitions,
	}
}

type canonicalSchemaBuilder struct {
	definitions map[string]CanonicalSchemaNode
	building    map[string]bool
}

func (b *canonicalSchemaBuilder) buildNode(t reflect.Type) CanonicalSchemaNode {
	t = derefType(t)
	switch t.Kind() {
	case reflect.Struct:
		key := t.PkgPath() + "." + t.Name()
		if t.Name() != "" {
			if _, exists := b.definitions[key]; exists || b.building[key] {
				return CanonicalSchemaNode{Ref: key}
			}
			b.building[key] = true
		}
		fields := b.schemaFields(t)
		node := CanonicalSchemaNode{Type: "object", Fields: fields}
		if isExplicitConfigExtension(t) {
			node.Opaque = true
		}
		if t.Name() != "" {
			b.definitions[key] = node
			delete(b.building, key)
			return CanonicalSchemaNode{Ref: key}
		}
		return node
	case reflect.Slice, reflect.Array:
		items := b.buildNode(t.Elem())
		return CanonicalSchemaNode{Type: "array", Items: &items}
	case reflect.Map:
		values := b.buildNode(t.Elem())
		return CanonicalSchemaNode{Type: "map", Values: &values}
	case reflect.Interface:
		return CanonicalSchemaNode{Type: "any", Opaque: true}
	default:
		return CanonicalSchemaNode{Type: "scalar"}
	}
}

func (b *canonicalSchemaBuilder) schemaFields(t reflect.Type) map[string]CanonicalSchemaNode {
	fields := make(map[string]CanonicalSchemaNode)
	for index := 0; index < t.NumField(); index++ {
		field := t.Field(index)
		if field.PkgPath != "" {
			continue
		}
		tag := field.Tag.Get("yaml")
		if tag == "-" {
			continue
		}
		name, options := splitYAMLTag(tag)
		if strings.Contains(options, "inline") || (name == "" && field.Anonymous) {
			inlineType := derefType(field.Type)
			if inlineType.Kind() == reflect.Struct {
				for inlineName, inlineNode := range b.schemaFields(inlineType) {
					fields[inlineName] = inlineNode
				}
			}
			continue
		}
		if name == "" {
			name = strings.ToLower(field.Name)
		}
		fields[name] = b.buildNode(field.Type)
	}
	return fields
}
