package config

import "strings"

// RedactedConfigValue replaces plaintext secret fields in dry-run output.
const RedactedConfigValue = "[REDACTED]"

// RedactSensitiveConfigValue replaces known secret keys with [REDACTED].
// Environment-variable names (*_env) stay visible.
func RedactSensitiveConfigValue(value any) any {
	switch typed := value.(type) {
	case map[string]interface{}:
		out := make(map[string]interface{}, len(typed))
		for key, nested := range typed {
			if isSensitiveConfigKey(key) {
				out[key] = RedactedConfigValue
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

func isSensitiveConfigKey(key string) bool {
	normalized := strings.ToLower(strings.TrimSpace(key))
	compact := strings.ReplaceAll(normalized, "_", "")
	if strings.HasSuffix(compact, "env") || strings.HasSuffix(compact, "envset") {
		return false
	}
	switch compact {
	case "apikey", "accesskey", "password", "authpassword",
		"clientsecret", "privatekey", "secret":
		return true
	}
	for _, suffix := range []string{
		"apikey",
		"accesskey",
		"password",
		"clientsecret",
		"privatekey",
	} {
		if strings.HasSuffix(compact, suffix) {
			return true
		}
	}
	return false
}
