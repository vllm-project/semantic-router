package apicontract

import (
	"encoding/json"
	"maps"
	"reflect"
	"strings"
	"time"
)

var (
	jsonRawMessageType = reflect.TypeFor[json.RawMessage]()
	timeType           = reflect.TypeFor[time.Time]()
)

// SchemaFor derives a schema from the Go type a handler encodes or decodes, so
// the handler's type stays the field-level source of truth.
func SchemaFor[T any]() Schema {
	return schemaFromType(reflect.TypeFor[T](), map[reflect.Type]bool{})
}

func schemaFromType(valueType reflect.Type, visiting map[reflect.Type]bool) Schema {
	if valueType.Kind() == reflect.Pointer {
		schema := schemaFromType(valueType.Elem(), visiting)
		schema.Nullable = true
		return schema
	}
	switch valueType {
	case jsonRawMessageType:
		return Schema{}
	case timeType:
		return Schema{Type: "string", Format: "date-time"}
	}

	switch valueType.Kind() {
	case reflect.Struct:
		return objectSchema(valueType, visiting)
	case reflect.Slice, reflect.Array:
		item := schemaFromType(valueType.Elem(), visiting)
		return Schema{Type: "array", Items: &item, Nullable: valueType.Kind() == reflect.Slice}
	case reflect.Map:
		if valueType.Key().Kind() != reflect.String || valueType.Elem().Kind() == reflect.Interface {
			return Schema{Type: "object", AdditionalProperties: true, Nullable: true}
		}
		additional := schemaFromType(valueType.Elem(), visiting)
		return Schema{Type: "object", AdditionalProperties: &additional, Nullable: true}
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

func objectSchema(valueType reflect.Type, visiting map[reflect.Type]bool) Schema {
	if visiting[valueType] {
		return Schema{Type: "object"}
	}
	visiting[valueType] = true
	defer delete(visiting, valueType)

	schema := Schema{Type: "object", Properties: map[string]Schema{}}
	for index := 0; index < valueType.NumField(); index++ {
		field := valueType.Field(index)
		if !field.IsExported() && !field.Anonymous {
			continue
		}
		name, optional, skip := jsonField(field)
		if skip {
			continue
		}
		fieldSchema := schemaFromType(field.Type, visiting)
		if field.Anonymous && field.Tag.Get("json") == "" && fieldSchema.Type == "object" {
			maps.Copy(schema.Properties, fieldSchema.Properties)
			if field.Type.Kind() != reflect.Pointer {
				schema.Required = append(schema.Required, fieldSchema.Required...)
			}
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
