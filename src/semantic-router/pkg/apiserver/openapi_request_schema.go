//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"reflect"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var (
	jsonRawMessageType = reflect.TypeOf(json.RawMessage{})
	timeType           = reflect.TypeOf(time.Time{})
)

// jsonWireRepresentation is implemented by owners with a custom JSON encoder.
// JSONWire must return the same concrete representation for every value,
// including the zero value, and MarshalJSON must serialize that representation.
// Keeping this structural interface here avoids dependencies from runtime DTOs
// to the management API or route-specific schema overrides.
type jsonWireRepresentation interface{ JSONWire() any }

// openAPIRequestSchemaFor derives request fields from the same Go type decoded
// by a route handler. Request structs remain the field-level source of truth;
// the route catalog only associates that type with its operation.
func openAPIRequestSchemaFor[T any]() *OpenAPISchema {
	typeOfT := reflect.TypeFor[T]()
	schema := openAPISchemaFromType(typeOfT, make(map[reflect.Type]bool))
	return &schema
}

func openAPISchemaFromType(valueType reflect.Type, visiting map[reflect.Type]bool) OpenAPISchema {
	if valueType.Kind() == reflect.Pointer {
		schema := openAPISchemaFromType(valueType.Elem(), visiting)
		schema.Nullable = true
		return schema
	}
	if valueType == reflect.TypeOf(config.OutputTokenDefault{}) {
		minimum := int64(1)
		return OpenAPISchema{OneOf: []OpenAPISchema{
			{Type: "integer", Minimum: &minimum},
			{Type: "string", Enum: []string{"auto"}},
		}}
	}
	if valueType == jsonRawMessageType {
		return OpenAPISchema{}
	}
	if valueType == timeType {
		return OpenAPISchema{Type: "string", Format: "date-time"}
	}
	if wire, ok := reflect.Zero(valueType).Interface().(jsonWireRepresentation); ok {
		if representation := reflect.TypeOf(wire.JSONWire()); representation != nil && representation != valueType {
			return openAPISchemaFromType(representation, visiting)
		}
	}

	switch valueType.Kind() {
	case reflect.Struct:
		return openAPIObjectSchema(valueType, visiting)
	case reflect.Slice, reflect.Array:
		item := openAPISchemaFromType(valueType.Elem(), visiting)
		return OpenAPISchema{Type: "array", Items: &item, Nullable: valueType.Kind() == reflect.Slice}
	case reflect.Map:
		if valueType.Key().Kind() != reflect.String {
			return OpenAPISchema{Type: "object"}
		}
		if valueType.Elem().Kind() == reflect.Interface {
			return OpenAPISchema{Type: "object", AdditionalProperties: true, Nullable: true}
		}
		additional := openAPISchemaFromType(valueType.Elem(), visiting)
		return OpenAPISchema{Type: "object", AdditionalProperties: &additional, Nullable: true}
	case reflect.Interface:
		return OpenAPISchema{}
	case reflect.Bool:
		return OpenAPISchema{Type: "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return OpenAPISchema{Type: "integer"}
	case reflect.Float32, reflect.Float64:
		return OpenAPISchema{Type: "number"}
	case reflect.String:
		return OpenAPISchema{Type: "string"}
	default:
		return OpenAPISchema{}
	}
}

func openAPIObjectSchema(valueType reflect.Type, visiting map[reflect.Type]bool) OpenAPISchema {
	if visiting[valueType] {
		return OpenAPISchema{Type: "object"}
	}
	visiting[valueType] = true
	defer delete(visiting, valueType)

	schema := OpenAPISchema{
		Type:       "object",
		Properties: make(map[string]OpenAPISchema),
	}
	required := make(map[string]bool)
	for index := 0; index < valueType.NumField(); index++ {
		field := valueType.Field(index)
		if !field.IsExported() && !field.Anonymous {
			continue
		}
		name, optional, skip := openAPIJSONField(field)
		if skip {
			continue
		}
		fieldSchema := openAPISchemaFromType(field.Type, visiting)
		if field.Anonymous && field.Tag.Get("json") == "" && fieldSchema.Type == "object" {
			for childName, childSchema := range fieldSchema.Properties {
				schema.Properties[childName] = childSchema
			}
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

func openAPIJSONField(field reflect.StructField) (name string, optional, skip bool) {
	name = field.Name
	tag := field.Tag.Get("json")
	parts := strings.Split(tag, ",")
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
