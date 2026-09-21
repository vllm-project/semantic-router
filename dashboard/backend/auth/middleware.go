package auth

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
	"unicode"
)

type contextKey string

const (
	authContextKey contextKey = "dashboardAuthContext"
	routePolicyKey contextKey = "dashboardRoutePolicy"
	revalidatorKey contextKey = "dashboardPermissionRevalidator"

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

type permissionRevalidator func(context.Context) error

var errPermissionDenied = errors.New("permission denied")

// AuthenticateRequest authorizes every request against the route registry.
// A request in a protected namespace with no registered contract is denied;
// a registered route is served only with the session and permissions its
// contract declares. Mutation routes are bounded, re-authorized after the body
// is read, and given a revalidator the handler calls before its side effect.
func AuthenticateRequest(service *Service, resolver RoutePolicyResolver) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			policy, lookup := resolver.LookupRoutePolicy(r.Method, r.URL.Path)
			switch lookup {
			case RouteMethodNotAllowed:
				http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
				return
			case RouteNotFound:
				if isProtectedNamespace(r.URL.Path) {
					http.Error(w, "Forbidden", http.StatusForbidden)
					return
				}
				next.ServeHTTP(w, r)
				return
			}
			if policy.Public {
				next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), routePolicyKey, policy)))
				return
			}

			claims, ok := authenticateSession(w, r, service)
			if !ok {
				return
			}
			user, perms, err := authorizeClaims(r.Context(), service, claims, policy)
			if err != nil {
				writeAuthorizationError(w, claims.UserID, err)
				return
			}
			if policy.MaxBodyBytes > 0 && r.Body != nil {
				if !boundRequestBody(w, r, policy) {
					return
				}
				if policy.Revalidate {
					// The body may have arrived slowly. Re-resolve the actor
					// before the handler sees the complete request.
					if user, perms, err = authorizeClaims(r.Context(), service, claims, policy); err != nil {
						writeAuthorizationError(w, claims.UserID, err)
						return
					}
				}
			}

			ctx := context.WithValue(r.Context(), authContextKey, AuthContext{
				UserID:    user.ID,
				SessionID: claims.ID,
				Email:     user.Email,
				Role:      user.Role,
				Perms:     perms,
			})
			ctx = context.WithValue(ctx, routePolicyKey, policy)
			if policy.Revalidate {
				ctx = context.WithValue(ctx, revalidatorKey, permissionRevalidator(func(checkCtx context.Context) error {
					_, _, checkErr := authorizeClaims(checkCtx, service, claims, policy)
					return checkErr
				}))
			}
			request := r.WithContext(ctx)
			if policy.AuditMode == AuditRequired {
				serveWithRouteAudit(service, policy, w, request, next)
				return
			}
			next.ServeHTTP(w, request)
		})
	}
}

// authenticateSession parses the credential and applies the CSRF policy for
// cookie transports. It writes the response on failure.
func authenticateSession(w http.ResponseWriter, r *http.Request, service *Service) (*TokenClaims, bool) {
	token, tokenSource := extractAccessTokenWithSource(r)
	if token == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return nil, false
	}
	claims, err := service.ParseToken(token)
	if err != nil {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return nil, false
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
			return nil, false
		}
		if !csrfTokenValid(service.jwtSecret, claims.ID, r.Header.Get(csrfHeaderName)) &&
			!embeddedGrafanaQueryAllowed(r, service.allowedOrigins) {
			http.Error(w, "Forbidden: missing or invalid CSRF token", http.StatusForbidden)
			return nil, false
		}
	}
	return claims, true
}

func authorizeClaims(
	ctx context.Context,
	service *Service,
	claims *TokenClaims,
	policy RoutePolicy,
) (*User, map[string]bool, error) {
	user, perms, err := service.ResolveSessionUser(ctx, claims)
	if err != nil {
		return nil, nil, err
	}
	if policy.SessionOnly {
		return user, perms, nil
	}
	for _, permission := range policy.Permissions {
		if !perms[permission] {
			return nil, nil, fmt.Errorf("%w: permission %q is required", errPermissionDenied, permission)
		}
	}
	return user, perms, nil
}

func writeAuthorizationError(w http.ResponseWriter, userID string, err error) {
	if errors.Is(err, errPermissionDenied) {
		http.Error(w, "Forbidden", http.StatusForbidden)
		return
	}
	log.Printf("permission load failed for user %s: %v", userID, err)
	http.Error(w, "Unauthorized", http.StatusUnauthorized)
}

// boundRequestBody enforces the contract's body limit. Revalidated mutations
// read the whole body here so the actor can be re-resolved once it has
// arrived; other bounded routes keep streaming through a limited reader.
func boundRequestBody(w http.ResponseWriter, r *http.Request, policy RoutePolicy) bool {
	limited := http.MaxBytesReader(w, r.Body, policy.MaxBodyBytes)
	if !policy.Revalidate {
		r.Body = limited
		return true
	}
	body, err := io.ReadAll(limited)
	if err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			http.Error(w, "Request body too large", http.StatusRequestEntityTooLarge)
		} else {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
		}
		return false
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	return true
}

func serveWithRouteAudit(
	service *Service,
	policy RoutePolicy,
	w http.ResponseWriter,
	r *http.Request,
	next http.Handler,
) {
	rw := &auditResponseWriter{ResponseWriter: w}
	next.ServeHTTP(rw, r)
	ac, _ := AuthFromContext(r)
	_ = service.store.AddAuditLog(context.WithoutCancel(r.Context()), AuditLog{
		UserID:     ac.UserID,
		Action:     policy.AuditAction,
		Resource:   string(policy.ResourceOwner),
		Method:     r.Method,
		Path:       r.URL.Path,
		IP:         r.RemoteAddr,
		UserAgent:  r.UserAgent(),
		StatusCode: rw.statusCodeOr200(),
		CreatedAt:  time.Now().Unix(),
	})
}

// ServiceUnavailableGuard fails closed when the auth service could not be
// initialized. Every registered protected route, and every unregistered path
// in a protected namespace, answers 503 Service Unavailable; public routes and
// the static frontend stay reachable so the Dashboard can surface the
// "authentication service is not configured" state.
//
// It consults the same registry as AuthenticateRequest so healthy and degraded
// startup cannot drift apart.
func ServiceUnavailableGuard(resolver RoutePolicyResolver) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			policy, lookup := resolver.LookupRoutePolicy(r.Method, r.URL.Path)
			switch {
			case lookup == RouteMethodNotAllowed:
				http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			case lookup == RouteFound && policy.Public:
				next.ServeHTTP(w, r)
			case lookup == RouteFound || isProtectedNamespace(r.URL.Path):
				http.Error(w, "Authentication service is not configured", http.StatusServiceUnavailable)
			default:
				next.ServeHTTP(w, r)
			}
		})
	}
}

// IsSRBenchComparisonRequest identifies the body-based read of saved results.
// It does not exempt the request from normal POST authentication or CSRF checks.
func IsSRBenchComparisonRequest(method, path string) bool {
	return method == http.MethodPost && path == "/api/sr-bench/v1/comparisons"
}

// RoutePolicyFromContext returns the contract policy that admitted the request.
func RoutePolicyFromContext(r *http.Request) (RoutePolicy, bool) {
	policy, ok := r.Context().Value(routePolicyKey).(RoutePolicy)
	return policy, ok
}

// RevalidateRequest re-resolves the live session and its current permissions.
// Handlers call it immediately before a privileged side effect so a session
// revoked or demoted while the request was in flight cannot commit.
func RevalidateRequest(r *http.Request) error {
	revalidate, ok := r.Context().Value(revalidatorKey).(permissionRevalidator)
	if !ok {
		return errors.New("live permission revalidation is unavailable")
	}
	return revalidate(r.Context())
}

// WithPermissionRevalidator installs a revalidator, for tests and for callers
// that dispatch outside AuthenticateRequest.
func WithPermissionRevalidator(ctx context.Context, check func(context.Context) error) context.Context {
	return context.WithValue(ctx, revalidatorKey, permissionRevalidator(check))
}

// RejectRevokedMutation writes 403 and reports true when the live session no
// longer authorizes the request. Requests admitted without a revalidator, such
// as direct handler tests, are not rejected.
func RejectRevokedMutation(w http.ResponseWriter, r *http.Request) bool {
	if r == nil {
		return false
	}
	if _, ok := r.Context().Value(revalidatorKey).(permissionRevalidator); !ok {
		return false
	}
	if err := RevalidateRequest(r); err != nil {
		http.Error(w, "Forbidden", http.StatusForbidden)
		return true
	}
	return false
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

type auditResponseWriter struct {
	http.ResponseWriter
	status int
}

func (w *auditResponseWriter) WriteHeader(status int) {
	if w.status == 0 {
		w.status = status
	}
	w.ResponseWriter.WriteHeader(status)
}

func (w *auditResponseWriter) Write(payload []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	return w.ResponseWriter.Write(payload)
}

func (w *auditResponseWriter) Flush() {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	_ = http.NewResponseController(w.ResponseWriter).Flush()
}

func (w *auditResponseWriter) Unwrap() http.ResponseWriter {
	return w.ResponseWriter
}

func (w *auditResponseWriter) statusCodeOr200() int {
	if w.status == 0 {
		return http.StatusOK
	}
	return w.status
}
