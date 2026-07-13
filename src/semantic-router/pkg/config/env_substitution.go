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
		if key == "request_template" {
			if template, ok := value.(string); ok {
				raw[key] = expandEnvStringPreservingRAGPlaceholders(template)
				continue
			}
		}
		raw[key] = expandEnvSubstitutionsInValue(value)
	}
}

var ragRequestTemplatePlaceholders = []string{
	"${user_content}",
	"${top_k}",
	"${threshold}",
}

// expandEnvStringPreservingRAGPlaceholders keeps the external RAG template
// language distinct from config environment interpolation. Other environment
// references in the same template continue to expand normally.
func expandEnvStringPreservingRAGPlaceholders(value string) string {
	const escapedDollarMarker = "\x00RAG_REQUEST_ESCAPED_DOLLAR\x00"
	protected := strings.ReplaceAll(value, "$$", escapedDollarMarker)
	markers := []string{
		"\x00RAG_REQUEST_PLACEHOLDER_0\x00",
		"\x00RAG_REQUEST_PLACEHOLDER_1\x00",
		"\x00RAG_REQUEST_PLACEHOLDER_2\x00",
	}
	for i, placeholder := range ragRequestTemplatePlaceholders {
		protected = strings.ReplaceAll(protected, placeholder, markers[i])
	}

	expanded := expandEnvString(protected)
	for i, marker := range markers {
		expanded = strings.ReplaceAll(expanded, marker, ragRequestTemplatePlaceholders[i])
	}
	return strings.ReplaceAll(expanded, escapedDollarMarker, "$")
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
		expanded, next := expandEnvDollarToken(escaped, i)
		builder.WriteString(expanded)
		i = next
	}

	return strings.ReplaceAll(builder.String(), dollarPlaceholder, "$")
}

func expandEnvDollarToken(escaped string, start int) (string, int) {
	if start+1 >= len(escaped) {
		return "$", start + 1
	}

	switch escaped[start+1] {
	case '{':
		return expandBracedEnvToken(escaped, start)
	case '$':
		return "$", start + 2
	default:
		return expandUnbracedEnvToken(escaped, start)
	}
}

func expandBracedEnvToken(escaped string, start int) (string, int) {
	closeIdx := strings.IndexByte(escaped[start+2:], '}')
	if closeIdx < 0 {
		return "$", start + 1
	}
	closeIdx += start + 2
	return resolveBracedEnvReference(escaped[start+2 : closeIdx]), closeIdx + 1
}

func expandUnbracedEnvToken(escaped string, start int) (string, int) {
	end := start + 1
	for end < len(escaped) && isEnvNameByte(escaped[end]) {
		end++
	}
	if end == start+1 {
		return "$", start + 1
	}
	return os.Getenv(escaped[start+1 : end]), end
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
