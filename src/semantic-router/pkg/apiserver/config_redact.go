//go:build !windows && cgo

package apiserver

import (
	"context"
	"net/http"
	"regexp"

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
	return config.RedactSensitiveConfigValue(value)
}
