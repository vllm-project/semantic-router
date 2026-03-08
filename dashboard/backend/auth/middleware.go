package auth

import (
	"encoding/json"
	"errors"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

// RequireRole resolves the request session and enforces the minimum console role.
func (s *Service) RequireRole(role console.ConsoleRole, next http.Handler) http.Handler {
	if s == nil || next == nil {
		return next
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodOptions {
			next.ServeHTTP(w, r)
			return
		}

		session, err := s.ResolveSession(w, r)
		if err != nil {
			writeAuthError(w, err)
			return
		}
		if !session.Allows(role) {
			writeAuthError(w, &Error{
				StatusCode: http.StatusForbidden,
				Code:       "insufficient_role",
				Message:    "Your session does not have permission to perform this action.",
			})
			return
		}

		next.ServeHTTP(w, r.WithContext(WithRequestSession(r.Context(), session)))
	})
}

func writeAuthError(w http.ResponseWriter, err error) {
	var authErr *Error
	if !errors.As(err, &authErr) || authErr == nil {
		authErr = &Error{
			StatusCode: http.StatusInternalServerError,
			Code:       "auth_error",
			Message:    "Dashboard auth request failed.",
		}
	}
	if authErr.StatusCode == 0 {
		authErr.StatusCode = http.StatusInternalServerError
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(authErr.StatusCode)
	_ = json.NewEncoder(w).Encode(map[string]string{
		"error":   authErr.Code,
		"message": authErr.Message,
	})
}
