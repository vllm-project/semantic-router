package auth

import (
	"bytes"
	"context"
	"errors"
	"io"
	"log"
	"net/http"
	"time"
)

const (
	routePolicyKey contextKey = "dashboardRoutePolicy"
	revalidatorKey contextKey = "dashboardPermissionRevalidator"
)

type permissionRevalidator func(context.Context) error

// ErrPermissionDenied distinguishes a revoked permission from an invalid
// session when a handler rechecks authorization before a side effect.
var ErrPermissionDenied = errors.New("permission denied")

func authenticateWithRoutePolicy(service *Service, resolver RoutePolicyResolver) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			policy, lookup := resolver.LookupRoutePolicy(r.Method, r.URL.EscapedPath())
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
			if policy.MaxAuthAge > 0 {
				if claims.IssuedAt == nil || time.Since(claims.IssuedAt.Time) > policy.MaxAuthAge || time.Since(claims.IssuedAt.Time) < 0 {
					http.Error(w, "Forbidden", http.StatusForbidden)
					return
				}
			}

			if tokenSource == tokenSourceCookie {
				if existing, cookieErr := r.Cookie(csrfCookieName); cookieErr != nil ||
					!csrfTokenValid(service.jwtSecret, claims.ID, existing.Value) {
					setCSRFCookie(w, r, claims.ID, service.jwtSecret, service.ttlDuration)
				}
			}
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

			user, perms, err := authorizeRouteClaims(r.Context(), service, claims, policy)
			if err != nil {
				writeRouteAuthError(w, err)
				return
			}
			if policy.MaxBodyBytes > 0 && r.Body != nil {
				if r.ContentLength > policy.MaxBodyBytes {
					http.Error(w, "Request body too large", http.StatusRequestEntityTooLarge)
					return
				}
				limited := http.MaxBytesReader(w, r.Body, policy.MaxBodyBytes)
				if policy.StreamBody {
					r.Body = limited
				} else {
					body, readErr := io.ReadAll(limited)
					if readErr != nil {
						var maxErr *http.MaxBytesError
						if errors.As(readErr, &maxErr) {
							http.Error(w, "Request body too large", http.StatusRequestEntityTooLarge)
						} else {
							http.Error(w, "Invalid request body", http.StatusBadRequest)
						}
						return
					}
					r.Body = io.NopCloser(bytes.NewReader(body))
				}
			}
			if policy.Revalidate && policy.MaxBodyBytes > 0 && !policy.StreamBody {
				user, perms, err = authorizeRouteClaims(r.Context(), service, claims, policy)
				if err != nil {
					writeRouteAuthError(w, err)
					return
				}
			}

			ctx := context.WithValue(r.Context(), authContextKey, AuthContext{
				UserID: user.ID, SessionID: claims.ID, Email: user.Email, Role: user.Role, Perms: perms,
			})
			ctx = context.WithValue(ctx, routePolicyKey, policy)
			ctx = context.WithValue(ctx, revalidatorKey, permissionRevalidator(func(checkCtx context.Context) error {
				_, _, checkErr := authorizeRouteClaims(checkCtx, service, claims, policy)
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

func unavailableWithRoutePolicy(resolver RoutePolicyResolver) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			policy, lookup := resolver.LookupRoutePolicy(r.Method, r.URL.EscapedPath())
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

func authorizeRouteClaims(ctx context.Context, service *Service, claims *TokenClaims, policy RoutePolicy) (*User, map[string]bool, error) {
	user, perms, err := service.ResolveSessionUser(ctx, claims)
	if err != nil {
		log.Printf("permission load failed for user %s: %v", claims.UserID, err)
		return nil, nil, err
	}
	for _, required := range append([]string{policy.Permission}, policy.AdditionalPermissions...) {
		if !perms[required] {
			return nil, nil, ErrPermissionDenied
		}
	}
	return user, perms, nil
}

func writeRouteAuthError(w http.ResponseWriter, err error) {
	if errors.Is(err, ErrPermissionDenied) {
		http.Error(w, "Forbidden", http.StatusForbidden)
		return
	}
	http.Error(w, "Unauthorized", http.StatusUnauthorized)
}

func serveWithRouteAudit(service *Service, policy RoutePolicy, w http.ResponseWriter, r *http.Request, next http.Handler) {
	rw := &auditResponseWriter{ResponseWriter: w}
	next.ServeHTTP(rw, r)
	ac, _ := AuthFromContext(r)
	_ = service.AddAuditLog(context.WithoutCancel(r.Context()), AuditLog{
		UserID: ac.UserID, Action: policy.AuditAction, Resource: string(policy.ResourceOwner),
		Method: r.Method, Path: r.URL.Path, IP: r.RemoteAddr, UserAgent: r.UserAgent(),
		StatusCode: rw.statusCodeOr200(), CreatedAt: time.Now().Unix(),
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

// RevalidateContextIfPresent checks live permission at a service-side write
// boundary. Direct service callers may have no Dashboard request policy.
func RevalidateContextIfPresent(ctx context.Context) error {
	revalidate, ok := ctx.Value(revalidatorKey).(permissionRevalidator)
	if !ok {
		return nil
	}
	return revalidate(ctx)
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
