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
			if isPublicPath(r.URL.Path) {
				next.ServeHTTP(w, r)
				return
			}
			token := extractBearer(r.Header.Get("Authorization"))
			if token == "" {
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
				return
			}

			claims, err := service.ParseToken(token)
			if err != nil {
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
				return
			}

			perms, err := service.store.GetEffectivePermissions(r.Context(), claims.Role, claims.UserID)
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

			ctx := context.WithValue(r.Context(), authContextKey, AuthContext{UserID: claims.UserID, Email: claims.Email, Role: claims.Role, Perms: perms})
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

func RequiredPermission(method, path string) string {
	path = strings.TrimSpace(strings.ToLower(path))
	switch {
	case strings.HasPrefix(path, "/api/admin/"):
		return PermUsersManage
	case strings.HasPrefix(path, "/api/settings"):
		if method == http.MethodGet {
			return PermConfigRead
		}
		if method == http.MethodPut || method == http.MethodPost {
			return PermConfigWrite
		}
		return PermConfigRead
	case strings.HasPrefix(path, "/api/router/config/"):
		if method == http.MethodGet {
			return PermConfigRead
		}
		return PermConfigWrite
	case strings.HasPrefix(path, "/api/tools"):
		return PermToolsUse
	case strings.HasPrefix(path, "/api/status"):
		return PermLogsRead
	case strings.HasPrefix(path, "/api/logs"):
		return PermLogsRead
	case strings.HasPrefix(path, "/api/topology"):
		return PermTopologyRead
	case strings.HasPrefix(path, "/api/evaluation"):
		switch method {
		case http.MethodPost:
			return PermEvalWrite
		case http.MethodDelete:
			return PermEvalWrite
		default:
			return PermEvalRead
		}
	case strings.HasPrefix(path, "/api/openclaw/") || strings.HasPrefix(path, "/embedded/openclaw/"):
		return PermOpenClaw
	case strings.HasPrefix(path, "/api/ml-pipeline/"):
		return PermMlPipeline
	case strings.HasPrefix(path, "/api/"):
		return PermUsersView
	default:
		return ""
	}
}

func AuthFromContext(r *http.Request) (AuthContext, bool) {
	ctxVal := r.Context().Value(authContextKey)
	ac, ok := ctxVal.(AuthContext)
	return ac, ok
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

func isPublicPath(path string) bool {
	switch {
	case strings.HasPrefix(path, "/api/auth/login"):
		return true
	case path == "/api/auth/login/":
		return true
	case strings.HasPrefix(path, "/api/setup/state"):
		return true
	case path == "/healthz":
		return true
	case strings.HasPrefix(path, "/static/"):
		return true
	case strings.HasPrefix(path, "/public/"):
		return true
	case strings.HasPrefix(path, "/avatar/"):
		return true
	case strings.HasPrefix(path, "/assets/") || strings.HasSuffix(path, ".js") || strings.HasSuffix(path, ".css") || strings.HasSuffix(path, ".png") || strings.HasSuffix(path, ".svg") || strings.HasSuffix(path, ".ico"):
		return true
	case path == "/login" || path == "/":
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
