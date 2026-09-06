package config

import (
	"fmt"
	"reflect"
	"sort"
	"strings"
)

func validateKnownFields(raw map[string]interface{}, targetType reflect.Type) error {
	unknown := collectUnknownFields(raw, targetType)
	if len(unknown) == 0 {
		return nil
	}
	sort.Strings(unknown)
	return fmt.Errorf("config contains unknown fields: %s", strings.Join(unknown, "; "))
}

type unknownFieldRecord struct {
	Field   string
	Message string
}

// CollectUnknownFieldDiagnostics returns schema-unknown YAML keys as
// field-addressable diagnostics. The loader treats these as fatal; Evaluate
// surfaces the same findings on the v1 validate contract.
func CollectUnknownFieldDiagnostics(raw map[string]interface{}) []Diagnostic {
	records := collectUnknownFieldRecords(raw, reflect.TypeOf(CanonicalConfig{}))
	diagnostics := make([]Diagnostic, 0, len(records))
	for _, record := range records {
		diagnostics = append(diagnostics, Diagnostic{
			Code:     DiagnosticUnknownField,
			Severity: SeverityError,
			Resource: resourceFromField(record.Field),
			Stage:    StageParse,
			Field:    record.Field,
			Message:  record.Message,
		})
	}
	return diagnostics
}

// collectUnknownFields returns path-aware diagnostics for keys not represented
// by the canonical Go contract. Opaque StructuredPayload values are excluded;
// their discriminator-specific validators own those nested fields.
func collectUnknownFields(raw map[string]interface{}, targetType reflect.Type) []string {
	records := collectUnknownFieldRecords(raw, targetType)
	diagnostics := make([]string, 0, len(records))
	for _, record := range records {
		diagnostics = append(diagnostics, record.Message)
	}
	return diagnostics
}

func collectUnknownFieldRecords(raw map[string]interface{}, targetType reflect.Type) []unknownFieldRecord {
	var records []unknownFieldRecord
	collectUnknownFieldsRecursive(raw, targetType, "", &records)
	return records
}

func collectUnknownFieldsRecursive(raw map[string]interface{}, t reflect.Type, path string, out *[]unknownFieldRecord) {
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
			*out = append(*out, unknownFieldRecord{
				Field:   joinPath(path, key),
				Message: formatUnknownField(key, path, known),
			})
			continue
		}
		recurseIntoValue(raw[key], entry.fieldType, joinPath(path, key), out)
	}
}

func formatUnknownField(key, path string, known map[string]fieldEntry) string {
	fullPath := joinPath(path, key)
	if suggestion := closestField(key, known); suggestion != "" {
		return fmt.Sprintf("unknown field %q in %s; did you mean %q?", key, displayPath(fullPath), suggestion)
	}
	return fmt.Sprintf("unknown field %q in %s", key, displayPath(fullPath))
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

func recurseIntoValue(value interface{}, fieldType reflect.Type, path string, out *[]unknownFieldRecord) {
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

func recurseIntoSlice(value interface{}, ft reflect.Type, path string, out *[]unknownFieldRecord) {
	elemType := derefType(ft.Elem())
	if elemType.Kind() != reflect.Struct {
		return
	}
	items, ok := value.([]interface{})
	if !ok {
		return
	}
	for index, item := range items {
		itemMap := nestedStringMap(item)
		if len(itemMap) > 0 {
			collectUnknownFieldsRecursive(itemMap, elemType, fmt.Sprintf("%s[%d]", path, index), out)
		}
	}
}

func recurseIntoMap(value interface{}, ft reflect.Type, path string, out *[]unknownFieldRecord) {
	elemType := derefType(ft.Elem())
	if elemType.Kind() != reflect.Struct {
		return
	}
	valMap := nestedStringMap(value)
	for mapKey, mapVal := range valMap {
		subMap := nestedStringMap(mapVal)
		if len(subMap) > 0 {
			collectUnknownFieldsRecursive(subMap, elemType, joinPath(path, mapKey), out)
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
