package auth

import (
	"context"
	"log"
	"net/http"
	"strings"
	"time"
)

type contextKey string

const authContextKey contextKey = "dashboardAuthContext"

const authSessionCookieName = "vsr_session"

// AuthContext contains authenticated user metadata.
type AuthContext struct {
	UserID string
	Email  string
	Role   string
	Perms  map[string]bool
}

func AuthenticateRequest(service *Service) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if !requiresAuthentication(r.URL.Path) {
				next.ServeHTTP(w, r)
				return
			}
			token := extractAccessToken(r)
			if token == "" {
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
				return
			}

			claims, err := service.ParseToken(token)
			if err != nil {
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
				return
			}

			user, perms, err := service.ResolveSessionUser(r.Context(), claims)
			if err != nil {
				log.Printf("permission load failed for user %s: %v", claims.UserID, err)
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
				return
			}

			required := RequiredPermission(r.Method, r.URL.Path)
			if required != "" && !perms[required] {
				http.Error(w, "Forbidden", http.StatusForbidden)
				return
			}

			ctx := context.WithValue(r.Context(), authContextKey, AuthContext{
				UserID: user.ID,
				Email:  user.Email,
				Role:   user.Role,
				Perms:  perms,
			})
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

func RequiredPermission(method, path string) string {
	path = strings.TrimSpace(strings.ToLower(path))
	for _, resolver := range []func(string, string) (string, bool){
		adminPermission,
		settingsPermission,
		routerPermission,
		knowledgePermission,
		toolsPermission,
		observabilityPermission,
		featurePermission,
	} {
		if permission, ok := resolver(method, path); ok {
			return permission
		}
	}

	if strings.HasPrefix(path, "/api/") {
		return PermConfigRead
	}

	return ""
}

func adminPermission(method, path string) (string, bool) {
	switch {
	case strings.HasPrefix(path, "/api/admin/users/password"):
		return PermUsersManage, true
	case strings.HasPrefix(path, "/api/admin/audit-logs"), strings.HasPrefix(path, "/api/admin/permissions"):
		return PermUsersManage, true
	case path == "/api/admin/users" || strings.HasPrefix(path, "/api/admin/users/"):
		if method == http.MethodGet {
			return PermUsersView, true
		}
		return PermUsersManage, true
	case strings.HasPrefix(path, "/api/admin/"):
		return PermUsersManage, true
	default:
		return "", false
	}
}

func settingsPermission(method, path string) (string, bool) {
	switch {
	case strings.HasPrefix(path, "/api/settings"):
		if method == http.MethodPut || method == http.MethodPost {
			return PermConfigWrite, true
		}
		return PermConfigRead, true
	case strings.HasPrefix(path, "/api/setup/validate"),
		strings.HasPrefix(path, "/api/setup/activate"),
		strings.HasPrefix(path, "/api/setup/import-remote"):
		return PermConfigWrite, true
	default:
		return "", false
	}
}

func routerPermission(method, path string) (string, bool) {
	switch {
	case strings.HasPrefix(path, "/api/router/config/"):
		if method == http.MethodGet {
			return PermConfigRead, true
		}
		return PermConfigWrite, true
	case strings.HasPrefix(path, "/api/router/"):
		return PermConfigRead, true
	default:
		return "", false
	}
}

func knowledgePermission(_ string, path string) (string, bool) {
	switch {
	case strings.HasPrefix(path, "/embedded/wizmap/"), path == "/embedded/wizmap":
		return PermConfigRead, true
	default:
		return "", false
	}
}

func toolsPermission(_ string, path string) (string, bool) {
	switch {
	case strings.HasPrefix(path, "/api/mcp/tools/execute"):
		return PermToolsUse, true
	case path == "/api/mcp/tools":
		return PermMcpRead, true
	case path == "/api/mcp/servers":
		return PermMcpRead, true
	case strings.HasPrefix(path, "/api/mcp/servers/") && strings.HasSuffix(path, "/status"):
		return PermMcpRead, true
	case strings.HasPrefix(path, "/api/tools"):
		return PermToolsUse, true
	case strings.HasPrefix(path, "/api/mcp/"):
		return PermMcpManage, true
	default:
		return "", false
	}
}

func observabilityPermission(_ string, path string) (string, bool) {
	switch {
	case strings.HasPrefix(path, "/api/status"), strings.HasPrefix(path, "/api/logs"):
		return PermLogsRead, true
	case strings.HasPrefix(path, "/embedded/grafana/"), strings.HasPrefix(path, "/embedded/jaeger"):
		return PermLogsRead, true
	case strings.HasPrefix(path, "/api/topology"):
		return PermTopologyRead, true
	default:
		return "", false
	}
}

func featurePermission(method, path string) (string, bool) {
	switch {
	case strings.HasPrefix(path, "/api/evaluation"):
		if method == http.MethodPost || method == http.MethodDelete {
			return PermEvalWrite, true
		}
		return PermEvalRead, true
	case strings.HasPrefix(path, "/api/openclaw/"), strings.HasPrefix(path, "/embedded/openclaw/"):
		return openclawPermission(method, path)
	case strings.HasPrefix(path, "/api/ml-pipeline/"):
		return PermMlPipeline, true
	default:
		return "", false
	}
}

func openclawPermission(method, path string) (string, bool) {
	switch {
	case strings.HasPrefix(path, "/embedded/openclaw/"):
		return PermOpenClawRead, true
	case strings.HasPrefix(path, "/api/openclaw/mcp"):
		return PermMcpManage, true
	case hasAnyPrefix(path,
		"/api/openclaw/provision",
		"/api/openclaw/start",
		"/api/openclaw/stop",
		"/api/openclaw/containers/",
		"/api/openclaw/next-port",
	):
		return PermOpenClaw, true
	case strings.HasPrefix(path, "/api/openclaw/rooms/") && (strings.HasSuffix(path, "/messages") || strings.HasSuffix(path, "/stream") || strings.HasSuffix(path, "/ws")):
		return PermOpenClawRead, true
	case hasAnyPrefix(path,
		"/api/openclaw/status",
		"/api/openclaw/skills",
		"/api/openclaw/token",
	):
		return PermOpenClawRead, true
	case hasAnyPrefix(path,
		"/api/openclaw/teams",
		"/api/openclaw/workers",
		"/api/openclaw/rooms",
	):
		return openclawMethodPermission(method), true
	default:
		return openclawMethodPermission(method), true
	}
}

func hasAnyPrefix(path string, prefixes ...string) bool {
	for _, prefix := range prefixes {
		if strings.HasPrefix(path, prefix) {
			return true
		}
	}
	return false
}

func openclawMethodPermission(method string) string {
	if method == http.MethodGet {
		return PermOpenClawRead
	}
	return PermOpenClaw
}

func AuthFromContext(r *http.Request) (AuthContext, bool) {
	ctxVal := r.Context().Value(authContextKey)
	ac, ok := ctxVal.(AuthContext)
	return ac, ok
}

func WithAuthContext(ctx context.Context, ac AuthContext) context.Context {
	return context.WithValue(ctx, authContextKey, ac)
}

func Require(permission string, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ac, ok := AuthFromContext(r)
		if !ok {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
		if permission != "" && !ac.Perms[permission] {
			http.Error(w, "Forbidden", http.StatusForbidden)
			return
		}
		next(w, r)
	}
}

func AuditMiddleware(store *Store, action, resource string, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		rw := &auditResponseWriter{ResponseWriter: w}
		next(rw, r)
		ac, ok := AuthFromContext(r)
		uid := ""
		if ok {
			uid = ac.UserID
		}
		_ = store.AddAuditLog(r.Context(), AuditLog{
			UserID:     uid,
			Action:     action,
			Resource:   resource,
			Method:     r.Method,
			Path:       r.URL.Path,
			IP:         r.RemoteAddr,
			UserAgent:  r.UserAgent(),
			StatusCode: rw.statusCodeOr200(),
			CreatedAt:  time.Now().Unix(),
		})
	}
}

func extractBearer(raw string) string {
	if raw == "" {
		return ""
	}
	parts := strings.SplitN(raw, " ", 2)
	if len(parts) != 2 {
		return ""
	}
	if !strings.EqualFold(parts[0], "bearer") {
		return ""
	}
	return parts[1]
}

func extractAccessToken(r *http.Request) string {
	if token := extractBearer(r.Header.Get("Authorization")); token != "" {
		return token
	}

	if cookie, err := r.Cookie(authSessionCookieName); err == nil {
		if token := strings.TrimSpace(cookie.Value); token != "" {
			return token
		}
	}

	return strings.TrimSpace(r.URL.Query().Get("authToken"))
}

func requiresAuthentication(path string) bool {
	path = strings.TrimSpace(strings.ToLower(path))

	switch {
	case strings.HasPrefix(path, "/api/auth/login"):
		return false
	case strings.HasPrefix(path, "/api/auth/bootstrap/"):
		return false
	case strings.HasPrefix(path, "/api/auth/me"):
		return true
	case strings.HasPrefix(path, "/api/setup/state"):
		return false
	case strings.HasPrefix(path, "/api/"):
		return true
	case strings.HasPrefix(path, "/embedded/"):
		return true
	default:
		return false
	}
}

type auditResponseWriter struct {
	http.ResponseWriter
	status int
}

func (w *auditResponseWriter) WriteHeader(status int) {
	w.status = status
	w.ResponseWriter.WriteHeader(status)
}

func (w *auditResponseWriter) statusCodeOr200() int {
	if w.status == 0 {
		return http.StatusOK
	}
	return w.status
}
