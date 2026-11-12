// Package openapischema derives OpenAPI 3.0 schemas from Go types. The Router
// management API and the Dashboard API share it so a request or response type
// is described the same way on both surfaces.
package openapischema

import (
	"encoding/json"
	"maps"
	"reflect"
	"strings"
	"time"
)

// Schema describes an OpenAPI 3.0 schema object.
type Schema struct {
	Minimum              *int64            `json:"minimum,omitempty"`
	OneOf                []Schema          `json:"oneOf,omitempty"`
	Nullable             bool              `json:"nullable,omitempty"`
	Type                 string            `json:"type,omitempty"`
	Format               string            `json:"format,omitempty"`
	Description          string            `json:"description,omitempty"`
	Enum                 []string          `json:"enum,omitempty"`
	Properties           map[string]Schema `json:"properties,omitempty"`
	Required             []string          `json:"required,omitempty"`
	Items                *Schema           `json:"items,omitempty"`
	AdditionalProperties any               `json:"additionalProperties,omitempty"`
	Ref                  string            `json:"$ref,omitempty"`
}

// Accepts reports whether an object with exactly these property names would
// validate against the schema: every required property present, and, when
// the schema closes additional properties, no name outside those declared.
// It lets a oneOf's branches be tested for exclusivity against the field sets
// a handler actually encodes.
func (s Schema) Accepts(keys []string) bool {
	present := make(map[string]bool, len(keys))
	for _, key := range keys {
		present[key] = true
	}
	for _, required := range s.Required {
		if !present[required] {
			return false
		}
	}
	if closed, ok := s.AdditionalProperties.(bool); ok && !closed {
		for _, key := range keys {
			if _, declared := s.Properties[key]; !declared {
				return false
			}
		}
	}
	return true
}

// Overrides replaces the derived schema of specific types, for types whose
// JSON encoding reflection cannot see.
type Overrides map[reflect.Type]Schema

var (
	jsonRawMessageType = reflect.TypeFor[json.RawMessage]()
	timeType           = reflect.TypeFor[time.Time]()
)

// jsonWireRepresentation is implemented by owners with a custom JSON encoder.
// JSONWire must return the same concrete representation for every value,
// including the zero value, and MarshalJSON must serialize that representation.
// Keeping this structural interface here avoids dependencies from runtime DTOs
// to an API package.
type jsonWireRepresentation interface{ JSONWire() any }

// For derives the schema of T, the type a handler encodes or decodes.
func For[T any](overrides Overrides) Schema {
	return FromType(reflect.TypeFor[T](), overrides)
}

// FromType derives the schema of valueType.
func FromType(valueType reflect.Type, overrides Overrides) Schema {
	return generator{overrides: overrides, visiting: map[reflect.Type]bool{}}.schema(valueType)
}

type generator struct {
	overrides Overrides
	visiting  map[reflect.Type]bool
}

func (g generator) schema(valueType reflect.Type) Schema {
	if valueType.Kind() == reflect.Pointer {
		schema := g.schema(valueType.Elem())
		schema.Nullable = true
		return schema
	}
	if override, ok := g.overrides[valueType]; ok {
		return override
	}
	if valueType == jsonRawMessageType {
		return Schema{}
	}
	if valueType == timeType {
		return Schema{Type: "string", Format: "date-time"}
	}
	if wire, ok := reflect.Zero(valueType).Interface().(jsonWireRepresentation); ok {
		if representation := reflect.TypeOf(wire.JSONWire()); representation != nil && representation != valueType {
			return g.schema(representation)
		}
	}

	switch valueType.Kind() {
	case reflect.Struct:
		return g.object(valueType)
	case reflect.Slice, reflect.Array:
		item := g.schema(valueType.Elem())
		return Schema{Type: "array", Items: &item, Nullable: valueType.Kind() == reflect.Slice}
	case reflect.Map:
		if valueType.Key().Kind() != reflect.String {
			return Schema{Type: "object"}
		}
		if valueType.Elem().Kind() == reflect.Interface {
			return Schema{Type: "object", AdditionalProperties: true, Nullable: true}
		}
		additional := g.schema(valueType.Elem())
		return Schema{Type: "object", AdditionalProperties: &additional, Nullable: true}
	case reflect.Interface:
		return Schema{}
	case reflect.Bool:
		return Schema{Type: "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return Schema{Type: "integer"}
	case reflect.Float32, reflect.Float64:
		return Schema{Type: "number"}
	case reflect.String:
		return Schema{Type: "string"}
	default:
		return Schema{}
	}
}

func (g generator) object(valueType reflect.Type) Schema {
	if g.visiting[valueType] {
		return Schema{Type: "object"}
	}
	g.visiting[valueType] = true
	defer delete(g.visiting, valueType)

	schema := Schema{Type: "object", Properties: map[string]Schema{}}
	required := map[string]bool{}
	for index := 0; index < valueType.NumField(); index++ {
		field := valueType.Field(index)
		if !field.IsExported() && !field.Anonymous {
			continue
		}
		name, optional, skip := jsonField(field)
		if skip {
			continue
		}
		fieldSchema := g.schema(field.Type)
		if field.Anonymous && field.Tag.Get("json") == "" && fieldSchema.Type == "object" {
			maps.Copy(schema.Properties, fieldSchema.Properties)
			if field.Type.Kind() != reflect.Pointer {
				for _, childName := range fieldSchema.Required {
					if !required[childName] {
						schema.Required = append(schema.Required, childName)
					}
					required[childName] = true
				}
			}
			continue
		}
		schema.Properties[name] = fieldSchema
		if !optional && !required[name] {
			schema.Required = append(schema.Required, name)
		}
		required[name] = !optional
	}
	// A custom wire representation can shadow embedded compatibility fields.
	// Its outer JSON tag controls optionality, and each required name occurs once.
	filtered := schema.Required[:0]
	for _, name := range schema.Required {
		if required[name] {
			filtered = append(filtered, name)
		}
	}
	schema.Required = filtered
	if len(schema.Properties) == 0 {
		schema.Properties = nil
	}
	return schema
}

func jsonField(field reflect.StructField) (name string, optional, skip bool) {
	name = field.Name
	parts := strings.Split(field.Tag.Get("json"), ",")
	if parts[0] == "-" {
		return "", false, true
	}
	if parts[0] != "" {
		name = parts[0]
	}
	for _, option := range parts[1:] {
		if option == "omitempty" || option == "omitzero" {
			optional = true
		}
	}
	return name, optional, false
}
