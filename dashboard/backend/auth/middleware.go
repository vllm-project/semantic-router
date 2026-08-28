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
	UserID string
	Email  string
	Role   string
	Perms  map[string]bool
}

type permissionRevalidator func(context.Context) error

var errPermissionDenied = errors.New("permission denied")

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
			if policy.MaxAuthAge > 0 {
				if claims.IssuedAt == nil {
					http.Error(w, "Forbidden", http.StatusForbidden)
					return
				}
				age := time.Since(claims.IssuedAt.Time)
				if age < 0 || age > policy.MaxAuthAge {
					http.Error(w, "Forbidden", http.StatusForbidden)
					return
				}
			}

			user, perms, err := authorizeClaims(r.Context(), service, claims, policy.Permission)
			if err != nil {
				log.Printf("permission load failed for user %s: %v", claims.UserID, err)
				if errors.Is(err, errPermissionDenied) {
					http.Error(w, "Forbidden", http.StatusForbidden)
				} else {
					http.Error(w, "Unauthorized", http.StatusUnauthorized)
				}
				return
			}

			if policy.MaxBodyBytes > 0 && r.Body != nil {
				body, readErr := readBoundedRequestBody(w, r, policy.MaxBodyBytes)
				if readErr != nil {
					return
				}
				r.Body = io.NopCloser(bytes.NewReader(body))
			}
			if policy.Revalidate && policy.MaxBodyBytes > 0 {
				user, perms, err = authorizeClaims(r.Context(), service, claims, policy.Permission)
				if err != nil {
					http.Error(w, "Forbidden", http.StatusForbidden)
					return
				}
			}

			ctx := context.WithValue(r.Context(), authContextKey, AuthContext{
				UserID: user.ID,
				Email:  user.Email,
				Role:   user.Role,
				Perms:  perms,
			})
			ctx = context.WithValue(ctx, routePolicyKey, policy)
			ctx = context.WithValue(ctx, revalidatorKey, permissionRevalidator(func(checkCtx context.Context) error {
				_, _, checkErr := authorizeClaims(checkCtx, service, claims, policy.Permission)
				return checkErr
			}))

			request := r.WithContext(ctx)
			if policy.AuditMode == AuditRequired {
				serveWithRouteAudit(service, policy, w, request, next)
				return
			}
			next.ServeHTTP(w, request)
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
// This is the deny-by-default counterpart to AuthenticateRequest and uses the
// same route policy registry so healthy and degraded startup cannot drift.
func ServiceUnavailableGuard(resolver RoutePolicyResolver) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			policy, lookup := resolver.LookupRoutePolicy(r.Method, r.URL.Path)
			if lookup == RouteMethodNotAllowed {
				http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
				return
			}
			if lookup == RouteFound && policy.Public {
				next.ServeHTTP(w, r)
				return
			}
			if lookup == RouteFound || isProtectedNamespace(r.URL.Path) {
				http.Error(w, "Authentication service is not configured", http.StatusServiceUnavailable)
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

func authorizeClaims(
	ctx context.Context,
	service *Service,
	claims *TokenClaims,
	permission string,
) (*User, map[string]bool, error) {
	user, perms, err := service.ResolveSessionUser(ctx, claims)
	if err != nil {
		return nil, nil, err
	}
	if !perms[permission] {
		return nil, nil, fmt.Errorf("%w: permission %q is required", errPermissionDenied, permission)
	}
	return user, perms, nil
}

func readBoundedRequestBody(w http.ResponseWriter, r *http.Request, maxBytes int64) ([]byte, error) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBytes))
	if err == nil {
		return body, nil
	}
	var maxErr *http.MaxBytesError
	if errors.As(err, &maxErr) {
		http.Error(w, "Request body too large", http.StatusRequestEntityTooLarge)
		return nil, err
	}
	http.Error(w, "Invalid request body", http.StatusBadRequest)
	return nil, err
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
	_ = service.AddAuditLog(context.WithoutCancel(r.Context()), AuditLog{
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

func RoutePolicyFromContext(r *http.Request) (RoutePolicy, bool) {
	policy, ok := r.Context().Value(routePolicyKey).(RoutePolicy)
	return policy, ok
}

func RevalidateRequest(r *http.Request) error {
	revalidate, ok := r.Context().Value(revalidatorKey).(permissionRevalidator)
	if !ok {
		return errors.New("live permission revalidation is unavailable")
	}
	return revalidate(r.Context())
}

func WithPermissionRevalidator(ctx context.Context, check func(context.Context) error) context.Context {
	return context.WithValue(ctx, revalidatorKey, permissionRevalidator(check))
}

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

func extractAccessToken(r *http.Request) string {
	if token := extractBearer(r.Header.Get("Authorization")); token != "" {
		return token
	}

	if cookie, err := r.Cookie(authSessionCookieName); err == nil {
		if token := normalizeAccessToken(cookie.Value); token != "" {
			return token
		}
	}

	return normalizeAccessToken(r.URL.Query().Get("authToken"))
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
	if w.status != 0 {
		return
	}
	w.status = status
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
