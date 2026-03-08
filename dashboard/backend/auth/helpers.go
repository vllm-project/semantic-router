package auth

import (
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

func bootstrapUserKey(id string) string {
	return "bootstrap:" + strings.TrimSpace(id)
}

func proxyUserKey(subject string) string {
	return "proxy:" + strings.TrimSpace(subject)
}

func requestIsHTTPS(r *http.Request) bool {
	if r == nil {
		return false
	}
	if r.TLS != nil {
		return true
	}
	return strings.EqualFold(r.Header.Get("X-Forwarded-Proto"), "https")
}

func sessionIsUsable(session console.Session, now time.Time) bool {
	if session.Status != console.SessionStatusActive {
		return false
	}
	if session.RevokedAt != nil {
		return false
	}
	if session.ExpiresAt != nil && !session.ExpiresAt.After(now) {
		return false
	}
	return true
}

func parseRoles(raw string) []console.ConsoleRole {
	parts := strings.FieldsFunc(raw, func(r rune) bool {
		return r == ',' || r == ';' || r == ' ' || r == '\t' || r == '\n'
	})
	seen := map[console.ConsoleRole]struct{}{}
	roles := make([]console.ConsoleRole, 0, len(parts))
	for _, part := range parts {
		role := normalizeRole(part)
		if role == "" {
			continue
		}
		if _, ok := seen[role]; ok {
			continue
		}
		seen[role] = struct{}{}
		roles = append(roles, role)
	}
	return roles
}

func parseRolesMetadata(raw interface{}) []console.ConsoleRole {
	switch typed := raw.(type) {
	case []string:
		return parseRoles(strings.Join(typed, ","))
	case []interface{}:
		values := make([]string, 0, len(typed))
		for _, value := range typed {
			if text, ok := value.(string); ok {
				values = append(values, text)
			}
		}
		return parseRoles(strings.Join(values, ","))
	case string:
		return parseRoles(typed)
	default:
		return nil
	}
}

func normalizeRole(raw string) console.ConsoleRole {
	switch console.ConsoleRole(strings.ToLower(strings.TrimSpace(raw))) {
	case console.ConsoleRoleViewer:
		return console.ConsoleRoleViewer
	case console.ConsoleRoleEditor:
		return console.ConsoleRoleEditor
	case console.ConsoleRoleOperator:
		return console.ConsoleRoleOperator
	case console.ConsoleRoleAdmin:
		return console.ConsoleRoleAdmin
	default:
		return ""
	}
}

func roleStrings(roles []console.ConsoleRole) []string {
	values := make([]string, 0, len(roles))
	for _, role := range roles {
		values = append(values, string(role))
	}
	return values
}

func mergeMetadata(base map[string]interface{}, overlays ...map[string]interface{}) map[string]interface{} {
	merged := map[string]interface{}{}
	for key, value := range base {
		merged[key] = value
	}
	for _, overlay := range overlays {
		for key, value := range overlay {
			merged[key] = value
		}
	}
	return merged
}

func existingMetadata(user *console.User) map[string]interface{} {
	if user == nil {
		return nil
	}
	return user.Metadata
}

func existingCreatedAt(user *console.User) time.Time {
	if user == nil {
		return time.Time{}
	}
	return user.CreatedAt
}

func expiresAtPtr(value time.Time) *time.Time {
	return &value
}

func coalesceString(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}
