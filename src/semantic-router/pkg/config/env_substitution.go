package config

import (
	"os"
	"strings"
)

// envExpander expands environment variable references in the string scalars
// of a parsed YAML tree. Supported forms mirror common compose interpolation:
//   - ${VAR} and $VAR
//   - ${VAR:-default} when VAR is unset or empty
//   - ${VAR-default} when VAR is unset
//   - $$ for a literal $
type envExpander struct {
	lookup func(string) (string, bool)
	// keepUnset writes a reference without a default as is, instead of empty,
	// when lookup does not find its variable.
	keepUnset bool
	// unset names variables written empty even when keepUnset is set.
	unset map[string]bool
	// kept maps each reference written as is to its variable.
	kept map[string]string
}

// processEnv resolves references from this process, as the Router does when it loads a config.
var processEnv = envExpander{lookup: os.LookupEnv}

func noEnv(string) (string, bool) { return "", false }

func (e envExpander) expandMap(raw map[string]interface{}) {
	for key, value := range raw {
		raw[key] = e.expandValue(value)
	}
}

func (e envExpander) expandValue(value interface{}) interface{} {
	switch typed := value.(type) {
	case string:
		return e.expandString(typed)
	case map[string]interface{}:
		e.expandMap(typed)
		return typed
	case map[interface{}]interface{}:
		converted := nestedStringMap(typed)
		e.expandMap(converted)
		return converted
	case []interface{}:
		for i, item := range typed {
			typed[i] = e.expandValue(item)
		}
		return typed
	default:
		return value
	}
}

func (e envExpander) expandString(value string) string {
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
		expanded, next := e.expandDollarToken(escaped, i)
		builder.WriteString(expanded)
		i = next
	}

	return strings.ReplaceAll(builder.String(), dollarPlaceholder, "$")
}

func (e envExpander) expandDollarToken(escaped string, start int) (string, int) {
	if start+1 >= len(escaped) {
		return "$", start + 1
	}

	switch escaped[start+1] {
	case '{':
		return e.expandBracedToken(escaped, start)
	case '$':
		return "$", start + 2
	default:
		return e.expandUnbracedToken(escaped, start)
	}
}

func (e envExpander) expandBracedToken(escaped string, start int) (string, int) {
	closeIdx := strings.IndexByte(escaped[start+2:], '}')
	if closeIdx < 0 {
		return "$", start + 1
	}
	closeIdx += start + 2
	return e.resolveBraced(escaped[start+2 : closeIdx]), closeIdx + 1
}

func (e envExpander) expandUnbracedToken(escaped string, start int) (string, int) {
	end := start + 1
	for end < len(escaped) && isEnvNameByte(escaped[end]) {
		end++
	}
	if end == start+1 {
		return "$", start + 1
	}
	return e.resolve(escaped[start+1:end], escaped[start:end]), end
}

func (e envExpander) resolveBraced(inner string) string {
	if inner == "" {
		return ""
	}
	if idx := strings.Index(inner, ":-"); idx > 0 {
		name := inner[:idx]
		defaultValue := inner[idx+2:]
		if value, ok := e.lookup(name); ok && value != "" {
			return value
		}
		return defaultValue
	}
	if idx := strings.Index(inner, "-"); idx > 0 {
		name := inner[:idx]
		defaultValue := inner[idx+1:]
		if value, ok := e.lookup(name); ok {
			return value
		}
		return defaultValue
	}
	return e.resolve(inner, "${"+inner+"}")
}

// resolve returns the value of a reference without a default.
func (e envExpander) resolve(name, reference string) string {
	if value, ok := e.lookup(name); ok {
		return value
	}
	if !e.keepUnset || e.unset[name] {
		return ""
	}
	if e.kept != nil {
		e.kept[reference] = name
	}
	return reference
}

// unsetQuoted makes each variable whose kept reference appears in message read
// as empty from now on, and reports whether there was one.
func (e envExpander) unsetQuoted(message string) bool {
	found := false
	for reference, name := range e.kept {
		if !e.unset[name] && quotesReference(message, reference) {
			e.unset[name] = true
			found = true
		}
	}
	return found
}

// quotesReference reports whether message contains reference other than as
// the start of a longer $NAME.
func quotesReference(message, reference string) bool {
	for {
		index := strings.Index(message, reference)
		if index < 0 {
			return false
		}
		message = message[index+len(reference):]
		if strings.HasSuffix(reference, "}") || message == "" || !isEnvNameByte(message[0]) {
			return true
		}
	}
}

func isEnvNameByte(ch byte) bool {
	return (ch >= 'A' && ch <= 'Z') ||
		(ch >= 'a' && ch <= 'z') ||
		(ch >= '0' && ch <= '9') ||
		ch == '_'
}
