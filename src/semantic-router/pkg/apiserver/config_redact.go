//go:build !windows

package apiserver

import (
	"context"
	"net/http"
	"regexp"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const redactedConfigValue = "[REDACTED]"

// sensitiveAssignmentPattern matches YAML/JSON-style secret assignments that
// sometimes appear inside parse/validation error strings.
var sensitiveAssignmentPattern = regexp.MustCompile(
	`(?i)((?:["']?(?:api[_-]?key|x[_-]?api[_-]?key|access[_-]?(?:key|token)|auth[_-]?(?:password|token)|refresh[_-]?token|client[_-]?secret|private[_-]?key|proxy[_-]?authorization|authorization|password|credential|secret|token)["']?)\s*[:=]\s*)(?:"[^"]*"|'[^']*'|(?:Bearer|Basic)\s+[^\s,}\]]+|[^\s,}\]]+)`,
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
				if isPureEnvironmentReference(nested) {
					out[key] = nested
				} else {
					out[key] = redactedConfigValue
				}
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

// isPureEnvironmentReference reports a value that is exactly one $NAME or
// ${NAME} the loader reads from the environment.
func isPureEnvironmentReference(value interface{}) bool {
	text, ok := value.(string)
	if !ok {
		return false
	}
	text = strings.TrimSpace(text)
	names := config.EnvironmentReferenceNames(text)
	return len(names) == 1 && (text == "$"+names[0] || text == "${"+names[0]+"}")
}

func isSensitiveConfigKey(key string) bool {
	normalized := strings.ToLower(strings.TrimSpace(key))
	compact := strings.NewReplacer("_", "", "-", "", " ", "").Replace(normalized)
	// Env var names (api_key_env) and presence flags stay visible.
	if strings.HasSuffix(compact, "env") || strings.HasSuffix(compact, "envset") {
		return false
	}
	if isSensitiveConfigName(compact) {
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
	// A plural names the same secret only when its singular is one of the names
	// above: listeners[].api_keys, tokens, credentials. It is not a suffix rule,
	// which would read a count such as max_tokens as a token.
	if strings.HasSuffix(compact, "s") && isSensitiveConfigName(strings.TrimSuffix(compact, "s")) {
		return true
	}
	return false
}

// isSensitiveConfigName reports a key that is itself a secret, without the
// prefix rules a longer name may carry.
func isSensitiveConfigName(compact string) bool {
	switch compact {
	case "apikey", "xapikey", "accesskey", "password", "authpassword",
		"clientsecret", "privatekey", "authorization", "proxyauthorization",
		"credential", "secret", "token":
		return true
	}
	return false
}
