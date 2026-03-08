package handlers

import (
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"time"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

type AuthSessionUserResponse struct {
	ID          string `json:"id"`
	Email       string `json:"email,omitempty"`
	DisplayName string `json:"displayName,omitempty"`
}

type AuthSessionMetaResponse struct {
	ID        string     `json:"id"`
	ExpiresAt *time.Time `json:"expiresAt,omitempty"`
}

type AuthSessionResponse struct {
	Authenticated bool                       `json:"authenticated"`
	AuthMode      string                     `json:"authMode"`
	User          AuthSessionUserResponse    `json:"user"`
	Session       AuthSessionMetaResponse    `json:"session"`
	Roles         []string                   `json:"roles"`
	EffectiveRole string                     `json:"effectiveRole"`
	Capabilities  dashboardauth.Capabilities `json:"capabilities"`
}

func AuthSessionHandler(service *dashboardauth.Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		session, err := service.ResolveSession(w, r)
		if err != nil {
			writeAuthErrorResponse(w, err)
			return
		}

		response := AuthSessionResponse{
			Authenticated: session.Authenticated,
			AuthMode:      string(session.AuthMode),
			User: AuthSessionUserResponse{
				ID:          session.User.ID,
				Email:       session.User.Email,
				DisplayName: session.User.DisplayName,
			},
			Session: AuthSessionMetaResponse{
				ID:        session.Session.ID,
				ExpiresAt: session.Session.ExpiresAt,
			},
			Roles:         roleStrings(session.Roles),
			EffectiveRole: string(session.EffectiveRole),
			Capabilities:  session.Capabilities,
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			log.Printf("Error encoding auth session response: %v", err)
		}
	}
}

func AuthLogoutHandler(service *dashboardauth.Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		sessionID, err := service.RevokeRequestSession(w, r)
		if err != nil {
			writeAuthErrorResponse(w, err)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]string{
			"status":    "success",
			"sessionId": sessionID,
		}); err != nil {
			log.Printf("Error encoding auth logout response: %v", err)
		}
	}
}

func requestActorID(r *http.Request) string {
	session, ok := dashboardauth.SessionFromRequest(r)
	if !ok {
		return ""
	}
	return session.User.ID
}

func writeAuthErrorResponse(w http.ResponseWriter, err error) {
	var dashboardauthErr *dashboardauth.Error
	if !errors.As(err, &dashboardauthErr) || dashboardauthErr == nil {
		dashboardauthErr = &dashboardauth.Error{
			StatusCode: http.StatusInternalServerError,
			Code:       "auth_error",
			Message:    "Dashboard auth request failed.",
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(dashboardauthErr.StatusCode)
	if encodeErr := json.NewEncoder(w).Encode(map[string]string{
		"error":   dashboardauthErr.Code,
		"message": dashboardauthErr.Message,
	}); encodeErr != nil {
		log.Printf("Error encoding auth error response: %v", encodeErr)
	}
}

func roleStrings(roles []console.ConsoleRole) []string {
	values := make([]string, 0, len(roles))
	for _, role := range roles {
		values = append(values, string(role))
	}
	return values
}
