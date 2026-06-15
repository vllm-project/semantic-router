package config

import (
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"

	yamlv3 "gopkg.in/yaml.v3"
)

type testingT interface {
	Helper()
	Fatalf(format string, args ...interface{})
}

func readReferenceConfigYAML(t testingT) []byte {
	t.Helper()
	root := referenceConfigRepoRoot(t)
	path := filepath.Join(root, "config", "config.yaml")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read %s: %v", path, err)
	}
	return data
}

func loadReferenceConfigRaw(t testingT) map[string]interface{} {
	t.Helper()
	var root map[string]interface{}
	if err := yamlv3.Unmarshal(readReferenceConfigYAML(t), &root); err != nil {
		t.Fatalf("failed to unmarshal config/config.yaml into raw map: %v", err)
	}
	return root
}

func referenceConfigRepoRoot(t testingT) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatalf("failed to resolve reference config test path")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "../../../../"))
}

func assertPluginConfigCoverage(t testingT, plugins []map[string]interface{}, typ reflect.Type, pluginType string) {
	t.Helper()
	configs := collectChildMapsFromSlice(t, plugins, "configuration", "plugins("+pluginType+")")
	assertSliceUnionCoversStructFields(t, configs, typ, "routing.decisions[].plugins[type="+pluginType+"].configuration")
}

func collectRuleCoverage(t testingT, rule map[string]interface{}, operators map[string]bool, signalTypes map[string]bool) {
	t.Helper()
	if signalType, ok := rule["type"].(string); ok && signalType != "" {
		signalTypes[signalType] = true
	}
	if operator, ok := rule["operator"].(string); ok && operator != "" {
		operators[operator] = true
	}
	conditions, ok := rule["conditions"]
	if !ok {
		return
	}
	for _, rawCondition := range mustSliceValue(t, conditions, "rule.conditions") {
		collectRuleCoverage(t, mustMapValue(t, rawCondition, "rule.conditions[]"), operators, signalTypes)
	}
}

func mapByStringField(t testingT, items []map[string]interface{}, field string, label string) map[string]map[string]interface{} {
	t.Helper()
	result := make(map[string]map[string]interface{}, len(items))
	for _, item := range items {
		result[mustStringAt(t, item, field)] = item
	}
	return result
}

func collectNestedSliceItems(t testingT, items interface{}, key string, path string) []interface{} {
	t.Helper()
	var result []interface{}
	for _, rawItem := range normalizeSliceItems(t, items, path) {
		item := mustMapValue(t, rawItem, path)
		rawNested, ok := item[key]
		if !ok || rawNested == nil {
			continue
		}
		result = append(result, mustSliceValue(t, rawNested, path+"."+key)...)
	}
	if len(result) == 0 {
		t.Fatalf("%s does not contain any nested %s entries", path, key)
	}
	return result
}

func collectChildMapsFromSlice(t testingT, items interface{}, key string, path string) []map[string]interface{} {
	t.Helper()
	result := make([]map[string]interface{}, 0, len(normalizeSliceItems(t, items, path)))
	for _, rawItem := range normalizeSliceItems(t, items, path) {
		item := mustMapValue(t, rawItem, path)
		rawChild, ok := item[key]
		if !ok || rawChild == nil {
			continue
		}
		result = append(result, mustMapValue(t, rawChild, path+"."+key))
	}
	if len(result) == 0 {
		t.Fatalf("%s does not contain any %s maps", path, key)
	}
	return result
}

func normalizeSliceItems(t testingT, items interface{}, path string) []interface{} {
	t.Helper()
	switch typed := items.(type) {
	case []interface{}:
		return typed
	case []map[string]interface{}:
		result := make([]interface{}, 0, len(typed))
		for _, item := range typed {
			result = append(result, item)
		}
		return result
	default:
		t.Fatalf("%s is not a slice: %T", path, items)
		return nil
	}
}

func mapValuesToSlice(t testingT, value map[string]interface{}, path string) []interface{} {
	t.Helper()
	keys := make([]string, 0, len(value))
	for key := range value {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	result := make([]interface{}, 0, len(keys))
	for _, key := range keys {
		result = append(result, value[key])
	}
	if len(result) == 0 {
		t.Fatalf("%s is empty", path)
	}
	return result
}

func assertMapCoversStructFields(t testingT, node map[string]interface{}, typ reflect.Type, path string, skip ...string) {
	t.Helper()
	for _, field := range requiredYAMLFields(t, typ, skip...) {
		if _, ok := node[field]; !ok {
			t.Fatalf("%s is missing required reference-config key %q", path, field)
		}
	}
}

func assertSliceUnionCoversStructFields(t testingT, items interface{}, typ reflect.Type, path string, skip ...string) {
	t.Helper()
	rawItems := normalizeSliceItems(t, items, path)
	if len(rawItems) == 0 {
		t.Fatalf("%s is empty", path)
	}
	union := unionMapKeysFromInterfaces(t, rawItems, path)
	for _, field := range requiredYAMLFields(t, typ, skip...) {
		if !union[field] {
			t.Fatalf("%s does not cover reference-config field %q anywhere in the slice", path, field)
		}
	}
}

func unionMapKeysFromInterfaces(t testingT, items []interface{}, path string) map[string]bool {
	t.Helper()
	union := make(map[string]bool)
	for _, rawItem := range items {
		for key := range mustMapValue(t, rawItem, path) {
			union[key] = true
		}
	}
	return union
}

func requiredYAMLFields(t testingT, typ reflect.Type, skip ...string) []string {
	t.Helper()
	typ = indirectType(typ)
	if typ.Kind() != reflect.Struct {
		t.Fatalf("requiredYAMLFields expects a struct type, got %s", typ.Kind())
	}

	skipSet := make(map[string]bool, len(skip))
	for _, name := range skip {
		skipSet[name] = true
	}

	fields := map[string]bool{}
	collectYAMLFields(t, typ, fields, skipSet)

	names := make([]string, 0, len(fields))
	for name := range fields {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func collectYAMLFields(t testingT, typ reflect.Type, fields map[string]bool, skipSet map[string]bool) {
	t.Helper()
	for index := 0; index < typ.NumField(); index++ {
		field := typ.Field(index)
		if field.PkgPath != "" {
			continue
		}

		tag := field.Tag.Get("yaml")
		if tag == "-" {
			continue
		}

		name, inline := parseYAMLTag(field, tag)
		if inline {
			collectYAMLFields(t, indirectType(field.Type), fields, skipSet)
			continue
		}
		if name == "" {
			t.Fatalf("public field %s on %s must declare an explicit yaml tag for reference-config enforcement", field.Name, typ.Name())
		}
		if skipSet[name] {
			continue
		}
		fields[name] = true
	}
}

func requireMapKeys(t testingT, node map[string]interface{}, path string, keys ...string) {
	t.Helper()
	for _, key := range keys {
		if _, ok := node[key]; !ok {
			t.Fatalf("%s is missing required key %q", path, key)
		}
	}
}

func parseYAMLTag(_ reflect.StructField, tag string) (string, bool) {
	if tag == "" {
		return "", false
	}
	parts := bytesSplitComma(tag)
	name := parts[0]
	for _, part := range parts[1:] {
		if part == "inline" {
			return name, true
		}
	}
	return name, false
}

func bytesSplitComma(tag string) []string {
	parts := make([]string, 0, 2)
	start := 0
	for index := 0; index < len(tag); index++ {
		if tag[index] != ',' {
			continue
		}
		parts = append(parts, tag[start:index])
		start = index + 1
	}
	return append(parts, tag[start:])
}

func indirectType(typ reflect.Type) reflect.Type {
	for typ.Kind() == reflect.Pointer {
		typ = typ.Elem()
	}
	return typ
}

func mustMapAt(t testingT, root map[string]interface{}, path ...string) map[string]interface{} {
	t.Helper()
	current := interface{}(root)
	fullPath := ""
	for index, part := range path {
		if index > 0 {
			fullPath += "."
		}
		fullPath += part
		mapped := mustMapValue(t, current, fullPath)
		next, ok := mapped[part]
		if !ok {
			t.Fatalf("%s is missing key %q", fullPath, part)
		}
		current = next
	}
	return mustMapValue(t, current, fullPath)
}

func mustSliceAt(t testingT, root map[string]interface{}, path ...string) []interface{} {
	t.Helper()
	current := interface{}(root)
	fullPath := ""
	for index, part := range path {
		if index > 0 {
			fullPath += "."
		}
		fullPath += part
		mapped := mustMapValue(t, current, fullPath)
		next, ok := mapped[part]
		if !ok {
			t.Fatalf("%s is missing key %q", fullPath, part)
		}
		current = next
	}
	return mustSliceValue(t, current, fullPath)
}

func mustStringAt(t testingT, node map[string]interface{}, key string) string {
	t.Helper()
	value, ok := node[key]
	if !ok {
		t.Fatalf("missing key %q", key)
	}
	text, ok := value.(string)
	if !ok || text == "" {
		t.Fatalf("key %q must be a non-empty string, got %T", key, value)
	}
	return text
}

func mustMapValue(t testingT, value interface{}, path string) map[string]interface{} {
	t.Helper()
	mapped, ok := value.(map[string]interface{})
	if !ok {
		t.Fatalf("%s is not a map[string]interface{}: %T", path, value)
	}
	return mapped
}

func mustSliceValue(t testingT, value interface{}, path string) []interface{} {
	t.Helper()
	items, ok := value.([]interface{})
	if !ok {
		t.Fatalf("%s is not a []interface{}: %T", path, value)
	}
	return items
}
