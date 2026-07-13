package auth

import (
	"context"
	"log"
	"net/http"
	"strings"
	"time"
	"unicode"

	"github.com/vllm-project/semantic-router/dashboard/backend/browsersecurity"
)

type contextKey string

const (
	authContextKey contextKey = "dashboardAuthContext"

	authSessionCookieName = "vsr_session"
	maxAccessTokenBytes   = 8192
)

// AuthContext contains authenticated user metadata.
type AuthContext struct {
	UserID           string
	SessionID        string
	Email            string
	Role             string
	Perms            map[string]bool
	CredentialSource CredentialSource
}

type CredentialSource string

const (
	CredentialSourceBearer CredentialSource = "bearer"
	CredentialSourceCookie CredentialSource = "cookie"
	CredentialSourceQuery  CredentialSource = "query"
)

func AuthenticateRequest(service *Service) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if !requiresAuthentication(r.URL.Path) {
				next.ServeHTTP(w, r)
				return
			}
			protectedWriter := newProtectedResponseWriter(w)
			// Enforce before authentication failures and after a header-only
			// handler returns. The writer also enforces at every commit boundary.
			protectedWriter.enforceCachePolicy()
			defer protectedWriter.enforceCachePolicy()
			if IsWebSocketUpgradeRequest(r) && !ValidWebSocketOrigin(r) {
				http.Error(protectedWriter, "Forbidden", http.StatusForbidden)
				return
			}
			token, credentialSource := extractAccessTokenWithSource(r)
			if token == "" {
				http.Error(protectedWriter, "Unauthorized", http.StatusUnauthorized)
				return
			}
			if !validUnsafeRequestOrigin(r, credentialSource) {
				http.Error(protectedWriter, "Forbidden", http.StatusForbidden)
				return
			}

			claims, err := service.ParseToken(token)
			if err != nil {
				http.Error(protectedWriter, "Unauthorized", http.StatusUnauthorized)
				return
			}

			user, perms, err := service.ResolveSessionUser(r.Context(), claims)
			if err != nil {
				log.Printf("permission load failed for user %s: %v", claims.UserID, err)
				http.Error(protectedWriter, "Unauthorized", http.StatusUnauthorized)
				return
			}

			required := RequiredPermission(r.Method, r.URL.Path)
			if required != "" && !perms[required] {
				http.Error(protectedWriter, "Forbidden", http.StatusForbidden)
				return
			}

			ctx := context.WithValue(r.Context(), authContextKey, AuthContext{
				UserID:           user.ID,
				SessionID:        claims.ID,
				Email:            user.Email,
				Role:             user.Role,
				Perms:            perms,
				CredentialSource: credentialSource,
			})
			next.ServeHTTP(protectedWriter, r.WithContext(ctx))
		})
	}
}

func validUnsafeRequestOrigin(r *http.Request, credentialSource CredentialSource) bool {
	switch r.Method {
	case http.MethodGet, http.MethodHead, http.MethodOptions:
		return true
	}

	if len(r.Header.Values("Origin")) > 0 {
		return browsersecurity.ValidOrigin(r)
	}
	fetchSiteValues := r.Header.Values("Sec-Fetch-Site")
	if len(fetchSiteValues) > 1 {
		return false
	}
	if len(fetchSiteValues) == 1 {
		return strings.EqualFold(strings.TrimSpace(fetchSiteValues[0]), "same-origin")
	}

	// Non-browser API clients can continue using bearer authentication without
	// browser Origin metadata. Cookie/query credentials require same-origin
	// browser evidence on every unsafe method.
	return credentialSource == CredentialSourceBearer
}

// ServiceUnavailableGuard returns middleware that fails closed when the auth
// service could not be initialized. It rejects every request to a route that
// normally requires authentication with 503 Service Unavailable, while still
// allowing public routes (login/bootstrap endpoints, setup state, embedded
// assets, and the static frontend) through so the dashboard can render and
// surface the "authentication service is not configured" state.
//
// This is the deny-by-default counterpart to AuthenticateRequest: it shares
// the same requiresAuthentication policy so the set of protected routes cannot
// drift between the two paths.
func ServiceUnavailableGuard() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if requiresAuthentication(r.URL.Path) {
				setProtectedResponseCachePolicy(w)
				http.Error(w, "Authentication service is not configured", http.StatusServiceUnavailable)
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

// setProtectedResponseCachePolicy prevents browser and intermediary caches
// from retaining cookie-authenticated control-plane data or authentication
// failures. Public static assets keep their handler-specific cache policy.
func setProtectedResponseCachePolicy(w http.ResponseWriter) {
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("Pragma", "no-cache")
}

func RequiredPermission(method, path string) string {
	path = strings.TrimSpace(strings.ToLower(path))
	if path == "/api/auth/password" || path == "/api/auth/password/" {
		// Every authenticated active user may change their own password. The
		// handler derives the target user from AuthContext and never from JSON.
		return ""
	}
	for _, resolver := range []func(string, string) (string, bool){
		adminPermission,
		settingsPermission,
		routerPermission,
		knowledgePermission,
		toolsPermission,
		observabilityPermission,
		fleetSimPermission,
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
	case path == "/api/router/config/deploy",
		path == "/api/router/config/deploy/preview",
		path == "/api/router/config/rollback":
		return PermConfigDeploy, true
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

func toolsPermission(method string, path string) (string, bool) {
	switch {
	case strings.HasPrefix(path, "/api/mcp/tools/execute"):
		return PermToolsUse, true
	case path == "/api/mcp/tools":
		return readOrManagePermission(method, PermMcpRead, PermMcpManage), true
	case path == "/api/mcp/servers":
		return readOrManagePermission(method, PermMcpRead, PermMcpManage), true
	case strings.HasPrefix(path, "/api/mcp/servers/") && strings.HasSuffix(path, "/status"):
		return readOrManagePermission(method, PermMcpRead, PermMcpManage), true
	case strings.HasPrefix(path, "/api/tools"):
		return PermToolsUse, true
	case strings.HasPrefix(path, "/api/mcp/"):
		return PermMcpManage, true
	default:
		return "", false
	}
}

func readOrManagePermission(method, readPermission, managePermission string) string {
	if method == http.MethodGet || method == http.MethodHead || method == http.MethodOptions {
		return readPermission
	}
	return managePermission
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

func fleetSimPermission(method, path string) (string, bool) {
	if !strings.HasPrefix(path, "/api/fleet-sim/") && path != "/api/fleet-sim" {
		return "", false
	}
	return readOrManagePermission(method, PermConfigRead, PermConfigWrite), true
}

func featurePermission(method, path string) (string, bool) {
	switch {
	case strings.HasPrefix(path, "/api/evaluation"):
		if path == "/api/evaluation/run" || strings.HasPrefix(path, "/api/evaluation/cancel/") {
			return PermEvalRun, true
		}
		if method == http.MethodPost || method == http.MethodDelete {
			return PermEvalWrite, true
		}
		return PermEvalRead, true
	case strings.HasPrefix(path, "/api/openclaw/"), strings.HasPrefix(path, "/embedded/openclaw/"):
		return openclawPermission(method, path)
	case strings.HasPrefix(path, "/api/ml-pipeline/"):
		return PermMlPipeline, true
	case strings.HasPrefix(path, "/api/security/"):
		return securityPermission(method), true
	default:
		return "", false
	}
}

func securityPermission(method string) string {
	if method == http.MethodGet {
		return PermConfigRead
	}
	return PermSecurityManage
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
		err := store.AddAuditLog(r.Context(), AuditLog{
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
		reportAuditPersistenceError(err)
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
	return normalizeAccessToken(parts[1])
}

func extractAccessToken(r *http.Request) string {
	token, _ := extractAccessTokenWithSource(r)
	return token
}

func extractAccessTokenWithSource(r *http.Request) (string, CredentialSource) {
	if token := extractBearer(r.Header.Get("Authorization")); token != "" {
		return token, CredentialSourceBearer
	}

	if cookie, err := r.Cookie(authSessionCookieName); err == nil {
		if token := normalizeAccessToken(cookie.Value); token != "" {
			return token, CredentialSourceCookie
		}
	}

	if token := normalizeAccessToken(r.URL.Query().Get("authToken")); token != "" {
		return token, CredentialSourceQuery
	}
	return "", ""
}

func normalizeAccessToken(raw string) string {
	token := strings.TrimSpace(raw)
	if token == "" || len(token) > maxAccessTokenBytes {
		return ""
	}
	for _, r := range token {
		if r == ';' || unicode.IsControl(r) || unicode.IsSpace(r) {
			return ""
		}
	}
	return token
}

func requiresAuthentication(path string) bool {
	path = strings.TrimSpace(strings.ToLower(path))

	switch {
	case path == "/api/auth/login" || path == "/api/auth/login/":
		return false
	case path == "/api/auth/logout" || path == "/api/auth/logout/":
		return false
	case path == "/api/auth/bootstrap/can-register":
		return false
	case path == "/api/auth/bootstrap/can-register/":
		return false
	case path == "/api/auth/bootstrap/register":
		return false
	case path == "/api/auth/bootstrap/register/":
		return false
	case path == "/api/setup/state":
		return false
	case strings.HasPrefix(path, "/embedded/wizmap/assets/"):
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
