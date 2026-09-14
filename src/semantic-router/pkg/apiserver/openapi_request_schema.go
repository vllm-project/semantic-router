//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"reflect"
	"strings"
	"time"
)

var (
	jsonRawMessageType = reflect.TypeOf(json.RawMessage{})
	timeType           = reflect.TypeOf(time.Time{})
)

// openAPIRequestSchemaFor derives request fields from the same Go type decoded
// by a route handler. Request structs remain the field-level source of truth;
// the route catalog only associates that type with its operation.
func openAPIRequestSchemaFor[T any]() *OpenAPISchema {
	typeOfT := reflect.TypeFor[T]()
	schema := openAPISchemaFromType(typeOfT, make(map[reflect.Type]bool))
	return &schema
}

func openAPISchemaFromType(valueType reflect.Type, visiting map[reflect.Type]bool) OpenAPISchema {
	for valueType.Kind() == reflect.Pointer {
		valueType = valueType.Elem()
	}
	if valueType == jsonRawMessageType {
		return OpenAPISchema{}
	}
	if valueType == timeType {
		return OpenAPISchema{Type: "string", Format: "date-time"}
	}

	switch valueType.Kind() {
	case reflect.Struct:
		return openAPIObjectSchema(valueType, visiting)
	case reflect.Slice, reflect.Array:
		item := openAPISchemaFromType(valueType.Elem(), visiting)
		return OpenAPISchema{Type: "array", Items: &item}
	case reflect.Map:
		if valueType.Key().Kind() != reflect.String {
			return OpenAPISchema{Type: "object"}
		}
		if valueType.Elem().Kind() == reflect.Interface {
			return OpenAPISchema{Type: "object", AdditionalProperties: true}
		}
		additional := openAPISchemaFromType(valueType.Elem(), visiting)
		return OpenAPISchema{Type: "object", AdditionalProperties: &additional}
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
	for index := 0; index < valueType.NumField(); index++ {
		field := valueType.Field(index)
		if !field.IsExported() {
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
			schema.Required = append(schema.Required, fieldSchema.Required...)
			continue
		}
		schema.Properties[name] = fieldSchema
		if !optional {
			schema.Required = append(schema.Required, name)
		}
	}
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
