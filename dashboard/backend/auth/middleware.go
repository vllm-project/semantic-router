package auth

import (
	"bufio"
	"context"
	"log"
	"net"
	"net/http"
	"strings"
	"time"
	"unicode"

	"github.com/vllm-project/semantic-router/dashboard/backend/observability"
	"github.com/vllm-project/semantic-router/dashboard/backend/routercontract"
)

type contextKey string

const (
	authContextKey contextKey = "dashboardAuthContext"

	authSessionCookieName = "vsr_session"
	maxAccessTokenBytes   = 8192
)

// AuthContext contains authenticated user metadata.
type AuthContext struct {
	UserID    string
	SessionID string
	Email     string
	Role      string
	Perms     map[string]bool
}

func AuthenticateRequest(service *Service, resolvers ...RoutePolicyResolver) func(http.Handler) http.Handler {
	if len(resolvers) > 0 && resolvers[0] != nil {
		return authenticateWithRoutePolicy(service, resolvers[0])
	}
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if !requiresAuthentication(r.URL.Path) {
				next.ServeHTTP(w, r)
				return
			}
			token, tokenSource := extractAccessTokenWithSource(r)
			if token == "" {
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
				return
			}

			claims, err := service.ParseToken(token)
			if err != nil {
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
				return
			}

			// Repairs sessions minted before this shipped, which would otherwise 403 on
			// every write. Safe methods too, so a page load fixes it. Sets a response
			// cookie only; it must not authorise this request.
			if tokenSource == tokenSourceCookie {
				if existing, cookieErr := r.Cookie(csrfCookieName); cookieErr != nil ||
					!csrfTokenValid(service.jwtSecret, claims.ID, existing.Value) {
					setCSRFCookie(w, r, claims.ID, service.jwtSecret, service.ttlDuration)
				}
			}

			// The browser attaches the session cookie to any request aimed here, including
			// one a hostile page caused, so a cookie-authenticated write must prove it
			// originated here. Bearer is exempt: a browser never sets it. See #2465.
			if tokenSource != tokenSourceHeader && requiresCSRFCheck(r.Method) {
				if !originAllowed(r, service.allowedOrigins) {
					http.Error(w, "Forbidden: request origin is not permitted", http.StatusForbidden)
					return
				}
				if !csrfTokenValid(service.jwtSecret, claims.ID, r.Header.Get(csrfHeaderName)) &&
					!embeddedGrafanaQueryAllowed(r, service.allowedOrigins) {
					http.Error(w, "Forbidden: missing or invalid CSRF token", http.StatusForbidden)
					return
				}
			}

			user, perms, err := service.ResolveSessionUser(r.Context(), claims)
			if err != nil {
				log.Printf("permission load failed for user %s: %v", claims.UserID, err)
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
				return
			}

			if !routerGatewayRequestAllowed(r.Method, r.URL.Path) {
				http.Error(w, "Router management route is not exposed by the Dashboard", http.StatusForbidden)
				return
			}

			requiredPermissions := RequiredPermissions(r.Method, r.URL.Path)
			if len(requiredPermissions) == 0 && isProtectedNamespace(r.URL.Path) {
				http.Error(w, "Forbidden", http.StatusForbidden)
				return
			}
			for _, required := range requiredPermissions {
				if !perms[required] {
					http.Error(w, "Forbidden", http.StatusForbidden)
					return
				}
			}

			ctx := context.WithValue(r.Context(), authContextKey, AuthContext{
				UserID:    user.ID,
				SessionID: claims.ID,
				Email:     user.Email,
				Role:      user.Role,
				Perms:     perms,
			})
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
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
func ServiceUnavailableGuard(resolvers ...RoutePolicyResolver) func(http.Handler) http.Handler {
	if len(resolvers) > 0 && resolvers[0] != nil {
		return unavailableWithRoutePolicy(resolvers[0])
	}
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if requiresAuthentication(r.URL.Path) {
				http.Error(w, "Authentication service is not configured", http.StatusServiceUnavailable)
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

func requiredPermission(method, path string) string {
	if !strings.HasPrefix(path, "/api/router/") {
		path = strings.TrimSpace(strings.ToLower(path))
	}
	for _, resolver := range []func(string, string) (string, bool){
		adminPermission,
		settingsPermission,
		routerPermission,
		knowledgePermission,
		toolsPermission,
		observabilityPermission,
		recipePermission,
		featurePermission,
	} {
		if permission, ok := resolver(method, path); ok {
			return permission
		}
	}

	return ""
}

// RequiredPermissions returns every permission needed by a request. Most
// routes require one permission; sr-bench run creation and recovery persist a
// manifest and immediately launch work, so both need write and run permissions.
func RequiredPermissions(method, path string) []string {
	if policy, ok := routercontract.LookupManagement(method, path); ok {
		return policy.Permissions
	}
	if !strings.HasPrefix(path, "/api/router/") {
		path = strings.TrimSpace(strings.ToLower(path))
	}
	if method == http.MethodPost && (path == "/api/sr-bench/v1/runs" || isSRBenchRunAction(path, "recover")) {
		return []string{PermEvalWrite, PermEvalRun}
	}
	primary := requiredPermission(method, path)
	if primary == "" {
		return nil
	}
	return []string{primary}
}

func recipePermission(_ string, path string) (string, bool) {
	path = strings.TrimRight(path, "/")
	if matchesRoute(path, "/api/recipe/import") {
		return PermConfigWrite, true
	}
	if matchesAnyRoute(path, "/api/recipe/activate", "/api/recipe/deactivate") {
		return PermConfigDeploy, true
	}
	if matchesRoute(path, "/api/recipe/packages") {
		return PermConfigRead, true
	}
	if path == "/api/recipe" || matchesRoute(path, "/api/recipe/probes") {
		if strings.HasSuffix(path, "/validate") {
			return PermTopologyRead, true
		}
		return PermConfigRead, true
	}
	return "", false
}

func matchesAnyRoute(path string, bases ...string) bool {
	for _, base := range bases {
		if matchesRoute(path, base) {
			return true
		}
	}
	return false
}

func matchesRoute(path, base string) bool {
	return path == base || strings.HasPrefix(path, base+"/")
}

func adminPermission(method, path string) (string, bool) {
	switch {
	case path == "/api/auth/me" || path == "/api/auth/me/":
		return PermSessionRead, true
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
	if policy, ok := routercontract.LookupManagement(method, path); ok {
		return policy.Permissions[0], true
	}
	switch {
	case path == "/api/models/catalog":
		return PermConfigRead, true
	case path == "/api/models/discover":
		return PermConfigWrite, true
	case path == "/api/models/verify":
		return PermEvalRun, true
	case path == "/api/router/config/deploy", path == "/api/router/config/deploy/preview", path == "/api/router/config/rollback":
		return PermConfigDeploy, true
	case strings.HasPrefix(path, "/api/router/config/"):
		if method == http.MethodGet {
			return PermConfigRead, true
		}
		return PermConfigWrite, true
	case path == "/api/router/v1/chat/completions" && method == http.MethodPost:
		return PermInferenceRun, true
	case strings.HasPrefix(path, "/api/router/"):
		return "", true
	default:
		return "", false
	}
}

// Dashboard-owned config handlers and inference dispatch have their own policy.
// Every management gateway request must exist in the shared exact allowlist.
func routerGatewayRequestAllowed(method, path string) bool {
	if !strings.HasPrefix(path, "/api/router/") || strings.HasPrefix(path, "/api/router/config/") {
		return true
	}
	if path == "/api/router/v1/chat/completions" && (method == http.MethodPost || method == http.MethodOptions) {
		return true
	}
	_, ok := routercontract.LookupManagement(method, path)
	return ok
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
	case strings.HasPrefix(path, "/api/status"):
		return PermTopologyRead, true
	case strings.HasPrefix(path, "/api/logs"):
		return PermLogsRead, true
	case observability.IsGrafanaQueryPath(path), observability.IsJaegerAPIPath(path):
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
	case path == "/api/workflows/health":
		return PermConfigRead, true
	case path == "/api/sr-bench/v1" || strings.HasPrefix(path, "/api/sr-bench/v1/"):
		if IsSRBenchComparisonRequest(method, path) {
			return PermEvalRead, true
		}
		if isSRBenchRunAction(path, "cancel") {
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
	default:
		return "", false
	}
}

// IsSRBenchComparisonRequest identifies the body-based read of saved results.
// It does not exempt the request from normal POST authentication or CSRF checks.
func IsSRBenchComparisonRequest(method, path string) bool {
	return method == http.MethodPost && path == "/api/sr-bench/v1/comparisons"
}

func isSRBenchRunAction(path, action string) bool {
	path = strings.TrimRight(path, "/")
	rest := strings.TrimPrefix(path, "/api/sr-bench/v1/runs/")
	if rest == path {
		return false
	}
	parts := strings.Split(rest, "/")
	return len(parts) == 2 && parts[0] != "" && parts[1] == action
}

func openclawPermission(method, path string) (string, bool) {
	switch {
	case strings.HasPrefix(path, "/embedded/openclaw/"):
		return PermOpenClaw, true
	case strings.HasPrefix(path, "/api/openclaw/mcp"):
		return PermMcpManage, true
	case hasAnyPrefix(path,
		"/api/openclaw/provision",
		"/api/openclaw/start",
		"/api/openclaw/stop",
		"/api/openclaw/containers/",
		"/api/openclaw/next-port",
		"/api/openclaw/token",
	):
		return PermOpenClaw, true
	case strings.HasPrefix(path, "/api/openclaw/rooms/") &&
		(strings.HasSuffix(path, "/ws") || (method == http.MethodPost && strings.HasSuffix(path, "/messages"))):
		return PermOpenClaw, true
	case strings.HasPrefix(path, "/api/openclaw/rooms/") && (strings.HasSuffix(path, "/messages") || strings.HasSuffix(path, "/stream") || strings.HasSuffix(path, "/ws")):
		return PermOpenClawRead, true
	case hasAnyPrefix(path,
		"/api/openclaw/status",
		"/api/openclaw/skills",
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
	return normalizeAccessToken(parts[1])
}

// Which transport supplied the credential, so the CSRF check can apply to cookies only.
type accessTokenSource int

const (
	tokenSourceNone accessTokenSource = iota
	tokenSourceHeader
	tokenSourceCookie
)

// The query-string transport is deliberately absent: a credential in a URL is written down
// by proxy access logs, browser history and the Referer header. Browser transports use the
// HttpOnly vsr_session cookie the browser attaches for them; non-browser clients use
// Authorization: Bearer. See #2465.
func extractAccessTokenWithSource(r *http.Request) (string, accessTokenSource) {
	if token := extractBearer(r.Header.Get("Authorization")); token != "" {
		return token, tokenSourceHeader
	}

	if cookie, err := r.Cookie(authSessionCookieName); err == nil {
		if token := normalizeAccessToken(cookie.Value); token != "" {
			return token, tokenSourceCookie
		}
	}

	return "", tokenSourceNone
}

func extractAccessToken(r *http.Request) string {
	token, _ := extractAccessTokenWithSource(r)
	return token
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
	case strings.HasPrefix(path, "/api/auth/login"):
		return false
	case strings.HasPrefix(path, "/api/auth/logout"):
		return false
	case strings.HasPrefix(path, "/api/auth/bootstrap/"):
		return false
	case strings.HasPrefix(path, "/api/auth/invitations/"):
		return false
	case strings.HasPrefix(path, "/api/auth/me"):
		return true
	case strings.HasPrefix(path, "/api/setup/state"):
		return false
	case path == "/api/status" || path == "/api/status/":
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

func (w *auditResponseWriter) Flush() {
	if flusher, ok := w.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func (w *auditResponseWriter) Unwrap() http.ResponseWriter {
	return w.ResponseWriter
}

func (w *auditResponseWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	conn, buffered, err := http.NewResponseController(w.ResponseWriter).Hijack()
	if err == nil {
		w.status = http.StatusSwitchingProtocols
	}
	return conn, buffered, err
}

func (w *auditResponseWriter) statusCodeOr200() int {
	if w.status == 0 {
		return http.StatusOK
	}
	return w.status
}
