package config

import (
	"fmt"
	"reflect"
	"sort"
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

// RejectUnknownFields rejects keys outside the typed contract. Only explicitly
// named extension types such as StructuredPayload are intentionally skipped.
func RejectUnknownFields(raw map[string]interface{}, targetType reflect.Type) error {
	return rejectUnknownFields(raw, targetType, "", "yaml")
}

// RejectUnknownConfigValue validates a YAML-shaped map or list against a typed
// producer contract. Path is prepended to every diagnostic.
func RejectUnknownConfigValue(raw interface{}, targetType reflect.Type, path string) error {
	var diagnostics []unknownFieldDiagnostic
	targetType = derefType(targetType)
	switch targetType.Kind() {
	case reflect.Struct:
		collectUnknownFieldsRecursive(
			nestedStringMap(raw),
			targetType,
			path,
			"yaml",
			&diagnostics,
		)
	case reflect.Slice:
		recurseIntoSlice(raw, targetType, path, "yaml", &diagnostics)
	case reflect.Map:
		recurseIntoMap(raw, targetType, path, "yaml", &diagnostics)
	}
	return rejectUnknownFieldDiagnostics(diagnostics)
}

func rejectUnknownJSONFields(raw map[string]interface{}, targetType reflect.Type, path string) error {
	return rejectUnknownFields(raw, targetType, path, "json")
}

func rejectUnknownFields(raw map[string]interface{}, targetType reflect.Type, path, tagName string) error {
	diagnostics := collectUnknownFieldDiagnosticsWithTag(raw, targetType, path, tagName)
	return rejectUnknownFieldDiagnostics(diagnostics)
}

func rejectUnknownFieldDiagnostics(diagnostics []unknownFieldDiagnostic) error {
	if len(diagnostics) == 0 {
		return nil
	}
	fields := make([]string, 0, len(diagnostics))
	for _, diagnostic := range diagnostics {
		field := diagnostic.path
		if diagnostic.suggestion != "" {
			field += fmt.Sprintf(" (did you mean %q?)", diagnostic.suggestion)
		}
		fields = append(fields, field)
	}
	return fmt.Errorf("unsupported config fields: %s", strings.Join(fields, ", "))
}

// collectUnknownFields returns warning messages without logging them.
func collectUnknownFields(raw map[string]interface{}, targetType reflect.Type) []string {
	diagnostics := collectUnknownFieldDiagnostics(raw, targetType)
	warnings := make([]string, 0, len(diagnostics))
	for _, diagnostic := range diagnostics {
		warnings = append(warnings, formatUnknownField(diagnostic))
	}
	return warnings
}

type unknownFieldDiagnostic struct {
	path       string
	suggestion string
}

func collectUnknownFieldDiagnostics(raw map[string]interface{}, targetType reflect.Type) []unknownFieldDiagnostic {
	return collectUnknownFieldDiagnosticsWithTag(raw, targetType, "", "yaml")
}

func collectUnknownFieldDiagnosticsWithTag(
	raw map[string]interface{},
	targetType reflect.Type,
	path string,
	tagName string,
) []unknownFieldDiagnostic {
	var diagnostics []unknownFieldDiagnostic
	collectUnknownFieldsRecursive(raw, targetType, path, tagName, &diagnostics)
	return diagnostics
}

func collectUnknownFieldsRecursive(
	raw map[string]interface{},
	t reflect.Type,
	path string,
	tagName string,
	out *[]unknownFieldDiagnostic,
) {
	t = derefType(t)
	if t.Kind() != reflect.Struct {
		return
	}
	if isExplicitConfigExtension(t) {
		return
	}

	known := collectKnownTaggedFields(t, tagName)
	keys := make([]string, 0, len(raw))
	for key := range raw {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		entry, ok := known[key]
		if !ok {
			*out = append(*out, unknownFieldDiagnostic{
				path:       joinPath(path, key),
				suggestion: closestField(key, known),
			})
			continue
		}
		recurseIntoValue(raw[key], entry.fieldType, joinPath(path, key), tagName, out)
	}
}

func formatUnknownField(diagnostic unknownFieldDiagnostic) string {
	key := diagnostic.path
	if index := strings.LastIndex(key, "."); index >= 0 {
		key = key[index+1:]
	}
	if diagnostic.suggestion != "" {
		return fmt.Sprintf(
			"[config] Unknown field %q in %s — did you mean %q?",
			key,
			displayPath(diagnostic.path),
			diagnostic.suggestion,
		)
	}
	return fmt.Sprintf("[config] Unknown field %q in %s", key, displayPath(diagnostic.path))
}

// closestField returns the nearest known field name if edit distance ≤ 3.
// Reuses the levenshtein function already defined in domain_contract.go.
func closestField(unknown string, known map[string]fieldEntry) string {
	best := ""
	bestDist := 4
	keys := make([]string, 0, len(known))
	for key := range known {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if d := levenshtein(unknown, k); d < bestDist {
			bestDist = d
			best = k
		}
	}
	return best
}

func recurseIntoValue(
	value interface{},
	fieldType reflect.Type,
	path string,
	tagName string,
	out *[]unknownFieldDiagnostic,
) {
	if fieldType == nil {
		return
	}
	ft := derefType(fieldType)

	switch ft.Kind() {
	case reflect.Struct:
		m := nestedStringMap(value)
		if len(m) > 0 {
			collectUnknownFieldsRecursive(m, ft, path, tagName, out)
		}
	case reflect.Slice:
		recurseIntoSlice(value, ft, path, tagName, out)
	case reflect.Map:
		recurseIntoMap(value, ft, path, tagName, out)
	}
}

func recurseIntoSlice(
	value interface{},
	ft reflect.Type,
	path string,
	tagName string,
	out *[]unknownFieldDiagnostic,
) {
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
			collectUnknownFieldsRecursive(itemMap, elemType, fmt.Sprintf("%s[%d]", path, index), tagName, out)
		}
	}
}

func recurseIntoMap(
	value interface{},
	ft reflect.Type,
	path string,
	tagName string,
	out *[]unknownFieldDiagnostic,
) {
	elemType := derefType(ft.Elem())
	if elemType.Kind() != reflect.Struct {
		return
	}
	valMap := nestedStringMap(value)
	keys := make([]string, 0, len(valMap))
	for mapKey := range valMap {
		keys = append(keys, mapKey)
	}
	sort.Strings(keys)
	for _, mapKey := range keys {
		mapVal := valMap[mapKey]
		subMap := nestedStringMap(mapVal)
		if len(subMap) > 0 {
			collectUnknownFieldsRecursive(subMap, elemType, joinPath(path, mapKey), tagName, out)
		}
	}
}

type fieldEntry struct {
	fieldType reflect.Type
}

// collectKnownFields builds a map of config-tag → field info for a struct type,
// handling inline fields by promoting their children to the current level.
func collectKnownFields(t reflect.Type) map[string]fieldEntry {
	return collectKnownTaggedFields(t, "yaml")
}

func collectKnownTaggedFields(t reflect.Type, tagName string) map[string]fieldEntry {
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	known := make(map[string]fieldEntry)
	for i := 0; i < t.NumField(); i++ {
		collectKnownTaggedField(known, t.Field(i), tagName)
	}
	return known
}

func collectKnownTaggedField(known map[string]fieldEntry, field reflect.StructField, tagName string) {
	tag, hasTag := field.Tag.Lookup(tagName)
	if !hasTag && tagName == "json" {
		tag = field.Tag.Get("yaml")
	}
	if tag == "-" {
		return
	}
	name, opts := splitYAMLTag(tag)
	if strings.Contains(opts, "inline") || (name == "" && field.Anonymous) {
		collectInlineTaggedFields(known, field.Type, tagName)
		return
	}
	if name == "" {
		name = strings.ToLower(field.Name)
	}
	known[name] = fieldEntry{fieldType: field.Type}
}

func collectInlineTaggedFields(known map[string]fieldEntry, fieldType reflect.Type, tagName string) {
	fieldType = derefType(fieldType)
	if fieldType.Kind() != reflect.Struct {
		return
	}
	for name, entry := range collectKnownTaggedFields(fieldType, tagName) {
		known[name] = entry
	}
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

func isExplicitConfigExtension(t reflect.Type) bool {
	return derefType(t) == reflect.TypeOf(StructuredPayload{})
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
