package dsl

import (
	"fmt"
	"reflect"
	"sort"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func knownPluginConfigKeys(p *config.DecisionPlugin) map[string]struct{} {
	fields := pluginConfigToFields(p)
	if len(fields) == 0 {
		return nil
	}
	keys := make(map[string]struct{}, len(fields))
	for key := range fields {
		keys[key] = struct{}{}
	}
	return keys
}

func filterPluginConfigMap(raw map[string]interface{}, omit map[string]struct{}) map[string]interface{} {
	if len(raw) == 0 {
		return nil
	}
	filtered := make(map[string]interface{}, len(raw))
	for key, value := range raw {
		if _, skip := omit[key]; skip {
			continue
		}
		filtered[key] = value
	}
	return filtered
}

func decodePluginConfig[T any](p *config.DecisionPlugin) (*T, bool) {
	if p == nil || p.Configuration == nil {
		return nil, false
	}
	var result T
	if err := p.Configuration.DecodeInto(&result); err != nil {
		return nil, false
	}
	return &result, true
}

func writePluginConfigMap(sb *strings.Builder, raw map[string]interface{}, indent string) {
	keys := make([]string, 0, len(raw))
	for key := range raw {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		fmt.Fprintf(sb, "%s%s: %s\n", indent, key, formatPluginConfigValue(raw[key]))
	}
}

func normalizePluginConfigMap(raw *config.StructuredPayload) (map[string]interface{}, bool) {
	if raw == nil {
		return nil, false
	}
	typed, err := raw.AsStringMap()
	if err != nil {
		return nil, false
	}
	normalized := make(map[string]interface{}, len(typed))
	for key, value := range typed {
		normalized[key] = normalizePluginConfigValue(value)
	}
	return normalized, true
}

func normalizePluginConfigValue(raw interface{}) interface{} {
	switch typed := raw.(type) {
	case map[string]interface{}:
		normalized := make(map[string]interface{}, len(typed))
		for key, value := range typed {
			normalized[key] = normalizePluginConfigValue(value)
		}
		return normalized
	case map[interface{}]interface{}:
		normalized := make(map[string]interface{}, len(typed))
		for key, value := range typed {
			normalized[fmt.Sprintf("%v", key)] = normalizePluginConfigValue(value)
		}
		return normalized
	case []interface{}:
		normalized := make([]interface{}, len(typed))
		for index, value := range typed {
			normalized[index] = normalizePluginConfigValue(value)
		}
		return normalized
	default:
		return normalizePluginConfigReflectValue(raw)
	}
}

func normalizePluginConfigReflectValue(raw interface{}) interface{} {
	if raw == nil {
		return nil
	}
	value := reflect.ValueOf(raw)
	switch value.Kind() {
	case reflect.Array, reflect.Slice:
		normalized := make([]interface{}, value.Len())
		for index := 0; index < value.Len(); index++ {
			normalized[index] = normalizePluginConfigValue(value.Index(index).Interface())
		}
		return normalized
	case reflect.Map:
		normalized := make(map[string]interface{}, value.Len())
		iter := value.MapRange()
		for iter.Next() {
			normalized[fmt.Sprintf("%v", iter.Key().Interface())] = normalizePluginConfigValue(iter.Value().Interface())
		}
		return normalized
	default:
		return raw
	}
}

func formatPluginConfigValue(raw interface{}) string {
	normalized := normalizePluginConfigValue(raw)
	if scalar, ok := formatPluginConfigScalar(normalized); ok {
		return scalar
	}

	switch typed := normalized.(type) {
	case []interface{}:
		return formatPluginConfigList(typed)
	case map[string]interface{}:
		return formatPluginConfigMap(typed)
	case nil:
		return "null"
	default:
		return fmt.Sprintf("%v", normalized)
	}
}

func formatPluginConfigScalar(raw interface{}) (string, bool) {
	switch typed := raw.(type) {
	case string:
		return fmt.Sprintf("%q", typed), true
	case bool:
		return strconv.FormatBool(typed), true
	case int:
		return strconv.Itoa(typed), true
	case int64:
		return strconv.FormatInt(typed, 10), true
	case float64:
		return strconv.FormatFloat(typed, 'f', -1, 64), true
	case float32:
		return strconv.FormatFloat(float64(typed), 'f', -1, 64), true
	default:
		return "", false
	}
}

func formatPluginConfigList(items []interface{}) string {
	parts := make([]string, len(items))
	for i, item := range items {
		parts[i] = formatPluginConfigValue(item)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func formatPluginConfigMap(values map[string]interface{}) string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	parts := make([]string, len(keys))
	for i, key := range keys {
		parts[i] = fmt.Sprintf("%s: %s", key, formatPluginConfigValue(values[key]))
	}
	return "{ " + strings.Join(parts, ", ") + " }"
}
