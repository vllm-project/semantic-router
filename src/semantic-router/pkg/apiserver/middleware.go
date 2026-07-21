//go:build !windows && cgo

package apiserver

import (
	"context"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
)

const managementRequestIDHeader = "X-Request-Id"

type requestContextKey string

const (
	managementPrincipalContextKey requestContextKey = "management_principal"
	managementRequestIDContextKey requestContextKey = "management_request_id"
)

func (s *ClassificationAPIServer) wrapRouteHandler(route apiRoute, handler http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		requestID := requestIDForManagementRequest(r)
		w.Header().Set(managementRequestIDHeader, requestID)

		policy := s.managementAuthPolicy()
		principal, statusCode, code := policy.authorize(route, r)
		if statusCode != 0 {
			s.writeManagementError(w, statusCode, code, managementErrorMessage(code), requestID)
			return
		}

		if route.RequestBody.LimitBytes > 0 && r.Body != nil && route.RequestBody.Kind != "" {
			r.Body = http.MaxBytesReader(w, r.Body, route.RequestBody.LimitBytes)
		}

		ctx := withManagementRequestContext(r.Context(), requestID, principal)
		handler(w, r.WithContext(ctx))
	}
}

func requestIDForManagementRequest(r *http.Request) string {
	if r != nil {
		if existing := strings.TrimSpace(r.Header.Get(managementRequestIDHeader)); existing != "" {
			return existing
		}
	}
	return uuid.NewString()
}

func withManagementRequestContext(ctx context.Context, requestID string, principal managementPrincipal) context.Context {
	ctx = withManagementPrincipal(ctx, principal)
	return context.WithValue(ctx, managementRequestIDContextKey, requestID)
}

func withManagementPrincipal(ctx context.Context, principal managementPrincipal) context.Context {
	return context.WithValue(ctx, managementPrincipalContextKey, principal)
}

func (s *ClassificationAPIServer) writeManagementError(
	w http.ResponseWriter,
	statusCode int,
	code string,
	message string,
	requestID string,
) {
	s.writeJSONResponse(w, statusCode, map[string]interface{}{
		"error": map[string]interface{}{
			"code":       code,
			"message":    message,
			"request_id": requestID,
			"timestamp":  time.Now().UTC().Format(time.RFC3339),
		},
	})
}

func managementErrorMessage(code string) string {
	switch code {
	case "UNAUTHORIZED":
		return "management API authentication required"
	case "FORBIDDEN":
		return "management API permission denied"
	case "MANAGEMENT_AUTH_NOT_CONFIGURED":
		return "management API bearer auth is enabled but no tokens are configured"
	case "INVALID_AUTH_MODE":
		return "management API auth mode is invalid"
	default:
		return "management API request rejected"
	}
}
