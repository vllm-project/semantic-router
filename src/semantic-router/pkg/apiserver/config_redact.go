//go:build !windows && cgo

package apiserver

import (
	"context"
	"net/http"
	"regexp"
	"strings"
)

const redactedConfigValue = "[REDACTED]"

// sensitiveAssignmentPattern matches YAML/JSON-style secret assignments that
// sometimes appear inside parse/validation error strings.
var sensitiveAssignmentPattern = regexp.MustCompile(
	`(?i)((?:["']?(?:api[_-]?key|access[_-]?key|password|auth[_-]?password|client[_-]?secret|private[_-]?key|secret)["']?)\s*[:=]\s*)(?:"[^"]*"|'[^']*'|[^\s,}\]]+)`,
)

// scrubSecretsInErrorMessage removes plaintext credential assignments from
// management API error messages so parse/deploy failures cannot leak secrets.
func scrubSecretsInErrorMessage(message string) string {
	if message == "" {
		return message
	}
	return sensitiveAssignmentPattern.ReplaceAllString(message, "${1}"+redactedConfigValue)
}

// canViewSecrets reports whether the request principal may see plaintext
// secrets in config dumps. Requires secret_view (admin via "*" by default).
// Missing principal (direct handler calls) defaults to false.
func (s *ClassificationAPIServer) canViewSecrets(r *http.Request) bool {
	if r == nil {
		return false
	}
	principal, ok := managementPrincipalFromContext(r.Context())
	if !ok {
		return false
	}
	return principal.hasPermission(PermSecretView, s.managementAuthPolicy().Roles)
}

func managementPrincipalFromContext(ctx context.Context) (managementPrincipal, bool) {
	if ctx == nil {
		return managementPrincipal{}, false
	}
	principal, ok := ctx.Value(managementPrincipalContextKey).(managementPrincipal)
	return principal, ok
}

// maybeRedactConfigView leaves value unchanged when the caller has secret_view;
// otherwise recursively redacts known secret fields.
func (s *ClassificationAPIServer) maybeRedactConfigView(r *http.Request, value interface{}) interface{} {
	if s.canViewSecrets(r) {
		return value
	}
	return redactSensitiveConfigValue(value)
}

func redactSensitiveConfigValue(value interface{}) interface{} {
	switch typed := value.(type) {
	case map[string]interface{}:
		out := make(map[string]interface{}, len(typed))
		for key, nested := range typed {
			if isSensitiveConfigKey(key) {
				out[key] = redactedConfigValue
				continue
			}
			out[key] = redactSensitiveConfigValue(nested)
		}
		return out
	case []interface{}:
		out := make([]interface{}, len(typed))
		for i, nested := range typed {
			out[i] = redactSensitiveConfigValue(nested)
		}
		return out
	default:
		return value
	}
}

func isSensitiveConfigKey(key string) bool {
	normalized := strings.ToLower(strings.TrimSpace(key))
	compact := strings.ReplaceAll(normalized, "_", "")
	// Env var names (api_key_env) and presence flags stay visible.
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
