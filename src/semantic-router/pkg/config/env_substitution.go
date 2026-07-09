package config

import (
	"os"
	"strings"
)

// expandEnvSubstitutionsInMap walks a parsed YAML tree and expands environment
// variable references in string scalars. Supported forms mirror common compose
// interpolation:
//   - ${VAR} and $VAR
//   - ${VAR:-default} when VAR is unset or empty
//   - ${VAR-default} when VAR is unset
//   - $$ for a literal $
func expandEnvSubstitutionsInMap(raw map[string]interface{}) {
	for key, value := range raw {
		raw[key] = expandEnvSubstitutionsInValue(value)
	}
}

func expandEnvSubstitutionsInValue(value interface{}) interface{} {
	switch typed := value.(type) {
	case string:
		return expandEnvString(typed)
	case map[string]interface{}:
		expandEnvSubstitutionsInMap(typed)
		return typed
	case map[interface{}]interface{}:
		converted := nestedStringMap(typed)
		expandEnvSubstitutionsInMap(converted)
		return converted
	case []interface{}:
		for i, item := range typed {
			typed[i] = expandEnvSubstitutionsInValue(item)
		}
		return typed
	default:
		return value
	}
}

func expandEnvString(value string) string {
	if value == "" || !strings.Contains(value, "$") {
		return value
	}

	const dollarPlaceholder = "\x00DOLLAR\x00"
	escaped := strings.ReplaceAll(value, "$$", dollarPlaceholder)

	var builder strings.Builder
	builder.Grow(len(escaped))

	for i := 0; i < len(escaped); {
		if escaped[i] != '$' {
			builder.WriteByte(escaped[i])
			i++
			continue
		}
		if i+1 >= len(escaped) {
			builder.WriteByte('$')
			break
		}

		switch escaped[i+1] {
		case '{':
			closeIdx := strings.IndexByte(escaped[i+2:], '}')
			if closeIdx < 0 {
				builder.WriteByte('$')
				i++
				continue
			}
			closeIdx += i + 2
			builder.WriteString(resolveBracedEnvReference(escaped[i+2 : closeIdx]))
			i = closeIdx + 1
		case '$':
			builder.WriteByte('$')
			i += 2
		default:
			j := i + 1
			for j < len(escaped) && isEnvNameByte(escaped[j]) {
				j++
			}
			if j == i+1 {
				builder.WriteByte('$')
				i++
				continue
			}
			builder.WriteString(os.Getenv(escaped[i+1 : j]))
			i = j
		}
	}

	return strings.ReplaceAll(builder.String(), dollarPlaceholder, "$")
}

func resolveBracedEnvReference(inner string) string {
	if inner == "" {
		return ""
	}
	if idx := strings.Index(inner, ":-"); idx > 0 {
		name := inner[:idx]
		defaultValue := inner[idx+2:]
		if value, ok := os.LookupEnv(name); ok && value != "" {
			return value
		}
		return defaultValue
	}
	if idx := strings.Index(inner, "-"); idx > 0 {
		name := inner[:idx]
		defaultValue := inner[idx+1:]
		if value, ok := os.LookupEnv(name); ok {
			return value
		}
		return defaultValue
	}
	return os.Getenv(inner)
}

func isEnvNameByte(ch byte) bool {
	return (ch >= 'A' && ch <= 'Z') ||
		(ch >= 'a' && ch <= 'z') ||
		(ch >= '0' && ch <= '9') ||
		ch == '_'
}
