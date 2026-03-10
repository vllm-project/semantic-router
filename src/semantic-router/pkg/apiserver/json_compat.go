//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"reflect"
	"strings"
)

func jsonCompatibleValue(value interface{}) interface{} {
	return jsonCompatibleReflectValue(reflect.ValueOf(value))
}

func jsonCompatibleReflectValue(value reflect.Value) interface{} {
	if !value.IsValid() {
		return nil
	}

	switch value.Kind() {
	case reflect.Interface, reflect.Pointer:
		if value.IsNil() {
			return nil
		}
		return jsonCompatibleReflectValue(value.Elem())
	case reflect.Struct:
		return jsonCompatibleStruct(value)
	case reflect.Map:
		return jsonCompatibleMap(value)
	case reflect.Slice, reflect.Array:
		if value.Kind() == reflect.Slice && value.Type().Elem().Kind() == reflect.Uint8 {
			if value.IsNil() {
				return nil
			}
			buf := make([]byte, value.Len())
			reflect.Copy(reflect.ValueOf(buf), value)
			return buf
		}

		items := make([]interface{}, value.Len())
		for i := range items {
			items[i] = jsonCompatibleReflectValue(value.Index(i))
		}
		return items
	default:
		return value.Interface()
	}
}

func jsonCompatibleStruct(value reflect.Value) map[string]interface{} {
	structType := value.Type()
	normalized := make(map[string]interface{})

	for i := 0; i < value.NumField(); i++ {
		fieldType := structType.Field(i)
		if fieldType.PkgPath != "" {
			continue
		}

		fieldValue := value.Field(i)
		name, omitEmpty, skip := jsonFieldMetadata(fieldType)
		if skip {
			continue
		}

		if fieldType.Anonymous && name == "" {
			embedded := jsonCompatibleReflectValue(fieldValue)
			embeddedMap, ok := embedded.(map[string]interface{})
			if !ok {
				continue
			}
			for key, nestedValue := range embeddedMap {
				normalized[key] = nestedValue
			}
			continue
		}

		if name == "" {
			name = fieldType.Name
		}

		normalizedValue := jsonCompatibleReflectValue(fieldValue)
		if omitEmpty && isJSONEmptyValue(fieldValue) {
			continue
		}
		normalized[name] = normalizedValue
	}

	return normalized
}

func jsonCompatibleMap(value reflect.Value) map[string]interface{} {
	if value.IsNil() {
		return nil
	}

	normalized := make(map[string]interface{}, value.Len())
	iter := value.MapRange()
	for iter.Next() {
		key := fmt.Sprint(iter.Key().Interface())
		normalized[key] = jsonCompatibleReflectValue(iter.Value())
	}
	return normalized
}

func jsonFieldMetadata(field reflect.StructField) (name string, omitEmpty bool, skip bool) {
	tag := field.Tag.Get("json")
	if tag == "-" {
		return "", false, true
	}

	if tag == "" {
		return "", false, false
	}

	parts := strings.Split(tag, ",")
	name = parts[0]
	for _, option := range parts[1:] {
		if option == "omitempty" {
			omitEmpty = true
		}
	}

	return name, omitEmpty, false
}

func isJSONEmptyValue(value reflect.Value) bool {
	if !value.IsValid() {
		return true
	}

	switch value.Kind() {
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		return value.Len() == 0
	case reflect.Bool:
		return !value.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return value.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return value.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return value.Float() == 0
	case reflect.Interface, reflect.Pointer:
		return value.IsNil()
	case reflect.Struct:
		return value.IsZero()
	default:
		return false
	}
}
