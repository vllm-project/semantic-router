package config

import (
	"regexp"
	"strings"
)

// RedactedConfigValue replaces plaintext secret fields in dry-run output.
const RedactedConfigValue = "[REDACTED]"

var environmentReferencePattern = regexp.MustCompile(
	`^\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)$`,
)

// RedactSensitiveConfigValue replaces known secret keys with [REDACTED].
// Environment-variable names (*_env) and pure ${VAR}/$VAR references stay visible.
func RedactSensitiveConfigValue(value any) any {
	switch typed := value.(type) {
	case map[string]interface{}:
		out := make(map[string]interface{}, len(typed))
		for key, nested := range typed {
			if isSensitiveConfigKey(key) {
				if isPureEnvironmentReference(nested) {
					out[key] = nested
				} else {
					out[key] = RedactedConfigValue
				}
				continue
			}
			out[key] = RedactSensitiveConfigValue(nested)
		}
		return out
	case []interface{}:
		out := make([]interface{}, len(typed))
		for i, nested := range typed {
			out[i] = RedactSensitiveConfigValue(nested)
		}
		return out
	default:
		return value
	}
}

func isPureEnvironmentReference(value any) bool {
	text, ok := value.(string)
	return ok && environmentReferencePattern.MatchString(strings.TrimSpace(text))
}

func isSensitiveConfigKey(key string) bool {
	normalized := strings.ToLower(strings.TrimSpace(key))
	compact := strings.NewReplacer("_", "", "-", "", " ", "").Replace(normalized)
	// Env var names (api_key_env) and presence flags stay visible.
	if strings.HasSuffix(compact, "env") || strings.HasSuffix(compact, "envset") {
		return false
	}
	switch compact {
	case "apikey", "xapikey", "accesskey", "password", "authpassword",
		"clientsecret", "privatekey", "authorization", "proxyauthorization",
		"credential", "secret", "token":
		return true
	}
	for _, suffix := range []string{
		"apikey",
		"accesskey",
		"password",
		"clientsecret",
		"privatekey",
		"authorization",
		"credential",
		"token",
	} {
		if strings.HasSuffix(compact, suffix) {
			return true
		}
	}
	return false
}
