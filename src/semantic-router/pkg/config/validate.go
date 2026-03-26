package config

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// WarnUnknownFields logs warnings for YAML keys in raw that don't match any
// struct tag on targetType. Only field NAMES are validated, not values.
// Called once at startup after config parsing succeeds.
func WarnUnknownFields(raw map[string]interface{}, targetType reflect.Type) {
	for _, w := range collectUnknownFields(raw, targetType) {
		logging.Warnf("%s", w)
	}
}

// collectUnknownFields returns warning messages without logging them.
func collectUnknownFields(raw map[string]interface{}, targetType reflect.Type) []string {
	var warnings []string
	collectUnknownFieldsRecursive(raw, targetType, "", &warnings)
	return warnings
}

func collectUnknownFieldsRecursive(raw map[string]interface{}, t reflect.Type, path string, out *[]string) {
	t = derefType(t)
	if t.Kind() != reflect.Struct {
		return
	}

	known := collectKnownFields(t)
	if len(known) == 0 {
		return // opaque/schemaless struct (e.g., StructuredPayload) — skip
	}
	for key := range raw {
		entry, ok := known[key]
		if !ok {
			*out = append(*out, formatUnknownField(key, path, known))
			continue
		}
		recurseIntoValue(raw[key], entry.fieldType, joinPath(path, key), out)
	}
}

func formatUnknownField(key, path string, known map[string]fieldEntry) string {
	fullPath := joinPath(path, key)
	if suggestion := closestField(key, known); suggestion != "" {
		return fmt.Sprintf("[config] Unknown field %q in %s — did you mean %q?", key, displayPath(fullPath), suggestion)
	}
	return fmt.Sprintf("[config] Unknown field %q in %s", key, displayPath(fullPath))
}

// closestField returns the nearest known field name if edit distance ≤ 3.
// Reuses the levenshtein function already defined in domain_contract.go.
func closestField(unknown string, known map[string]fieldEntry) string {
	best := ""
	bestDist := 4
	for k := range known {
		if d := levenshtein(unknown, k); d < bestDist {
			bestDist = d
			best = k
		}
	}
	return best
}

func recurseIntoValue(value interface{}, fieldType reflect.Type, path string, out *[]string) {
	if fieldType == nil {
		return
	}
	ft := derefType(fieldType)

	switch ft.Kind() {
	case reflect.Struct:
		m := nestedStringMap(value)
		if len(m) > 0 {
			collectUnknownFieldsRecursive(m, ft, path, out)
		}
	case reflect.Slice:
		recurseIntoSlice(value, ft, path, out)
	case reflect.Map:
		recurseIntoMap(value, ft, path, out)
	}
}

func recurseIntoSlice(value interface{}, ft reflect.Type, path string, out *[]string) {
	elemType := derefType(ft.Elem())
	if elemType.Kind() != reflect.Struct {
		return
	}
	items, ok := value.([]interface{})
	if !ok {
		return
	}
	for _, item := range items {
		itemMap := nestedStringMap(item)
		if len(itemMap) > 0 {
			collectUnknownFieldsRecursive(itemMap, elemType, path, out)
		}
	}
}

func recurseIntoMap(value interface{}, ft reflect.Type, path string, out *[]string) {
	if ft.Elem().Kind() != reflect.Struct {
		return
	}
	valMap := nestedStringMap(value)
	for mapKey, mapVal := range valMap {
		subMap := nestedStringMap(mapVal)
		if len(subMap) > 0 {
			collectUnknownFieldsRecursive(subMap, ft.Elem(), joinPath(path, mapKey), out)
		}
	}
}

type fieldEntry struct {
	fieldType reflect.Type
}

// collectKnownFields builds a map of yaml-tag → field info for a struct type,
// handling inline fields by promoting their children to the current level.
func collectKnownFields(t reflect.Type) map[string]fieldEntry {
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	known := make(map[string]fieldEntry)
	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		tag := field.Tag.Get("yaml")
		if tag == "-" {
			continue
		}
		name, opts := splitYAMLTag(tag)

		if strings.Contains(opts, "inline") || (name == "" && field.Anonymous) {
			// Promote inline/embedded fields to the current level.
			ft := derefType(field.Type)
			if ft.Kind() == reflect.Struct {
				for k, v := range collectKnownFields(ft) {
					known[k] = v
				}
			}
			continue
		}
		if name == "" {
			name = strings.ToLower(field.Name)
		}
		known[name] = fieldEntry{fieldType: field.Type}
	}
	return known
}

func splitYAMLTag(tag string) (string, string) {
	parts := strings.SplitN(tag, ",", 2)
	if len(parts) > 1 {
		return parts[0], parts[1]
	}
	return parts[0], ""
}

func derefType(t reflect.Type) reflect.Type {
	for t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	return t
}

func joinPath(parent, child string) string {
	if parent == "" {
		return child
	}
	return parent + "." + child
}

func displayPath(fullPath string) string {
	if i := strings.LastIndex(fullPath, "."); i >= 0 {
		return fullPath[:i]
	}
	return "top level"
}
