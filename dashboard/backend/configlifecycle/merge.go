package configlifecycle

import (
	"fmt"
	"net"
	"strings"

	"gopkg.in/yaml.v3"
)

// DeepMerge recursively merges src into dst and mutates dst in place.
func DeepMerge(dst, src map[string]interface{}) map[string]interface{} {
	for key, srcVal := range src {
		if dstVal, exists := dst[key]; exists {
			if dstMap, ok := dstVal.(map[string]interface{}); ok {
				if srcMap, ok := srcVal.(map[string]interface{}); ok {
					dst[key] = DeepMerge(dstMap, srcMap)
					continue
				}
			}
			if dstMap, ok := ToStringKeyMap(dstVal); ok {
				if srcMap, ok := ToStringKeyMap(srcVal); ok {
					dst[key] = DeepMerge(dstMap, srcMap)
					continue
				}
			}
		}
		dst[key] = srcVal
	}
	return dst
}

// ToStringKeyMap converts YAML-style generic maps into map[string]interface{}.
func ToStringKeyMap(v interface{}) (map[string]interface{}, bool) {
	switch m := v.(type) {
	case map[string]interface{}:
		return m, true
	case map[interface{}]interface{}:
		result := make(map[string]interface{}, len(m))
		for k, val := range m {
			result[fmt.Sprintf("%v", k)] = val
		}
		return result, true
	default:
		return nil, false
	}
}

// CanonicalizeYAMLForDiff normalizes YAML to reduce order-only diff noise.
func CanonicalizeYAMLForDiff(raw []byte) string {
	text := string(raw)
	if strings.TrimSpace(text) == "" || strings.Contains(text, "# No existing config") {
		return text
	}

	var parsed interface{}
	if err := yaml.Unmarshal(raw, &parsed); err != nil {
		return text
	}

	canonical, err := yaml.Marshal(normalizeYAMLValue(parsed))
	if err != nil {
		return text
	}
	return string(canonical)
}

func normalizeYAMLValue(v interface{}) interface{} {
	switch value := v.(type) {
	case map[string]interface{}:
		normalized := make(map[string]interface{}, len(value))
		for key, item := range value {
			normalized[key] = normalizeYAMLValue(item)
		}
		return normalized
	case map[interface{}]interface{}:
		normalized := make(map[string]interface{}, len(value))
		for key, item := range value {
			normalized[fmt.Sprintf("%v", key)] = normalizeYAMLValue(item)
		}
		return normalized
	case []interface{}:
		normalized := make([]interface{}, len(value))
		for i, item := range value {
			normalized[i] = normalizeYAMLValue(item)
		}
		return normalized
	default:
		return value
	}
}

// ValidateEndpointAddress validates endpoint host/address syntax.
func ValidateEndpointAddress(address string) error {
	if address == "" {
		return fmt.Errorf("address cannot be empty")
	}
	if strings.HasPrefix(address, "http://") || strings.HasPrefix(address, "https://") {
		return fmt.Errorf("protocol prefix not allowed in address (use 'port' field for port number)")
	}
	if strings.Contains(address, "/") {
		return fmt.Errorf("paths not allowed in address field")
	}
	if strings.Contains(address, ":") && net.ParseIP(address) == nil {
		parts := strings.Split(address, ":")
		if len(parts) == 2 && len(parts[1]) > 0 && len(parts[1]) <= 5 {
			return fmt.Errorf("port not allowed in address field (use 'port' field instead)")
		}
	}

	if ip := net.ParseIP(address); ip != nil {
		return nil
	}
	if len(address) > 253 {
		return fmt.Errorf("DNS name too long (max 253 characters)")
	}
	for _, char := range address {
		if (char < 'a' || char > 'z') &&
			(char < 'A' || char > 'Z') &&
			(char < '0' || char > '9') &&
			char != '.' && char != '-' {
			return fmt.Errorf("invalid character in DNS name: %c", char)
		}
	}
	if strings.HasPrefix(address, ".") || strings.HasSuffix(address, ".") || strings.Contains(address, "..") {
		return fmt.Errorf("invalid DNS name format")
	}
	return nil
}
