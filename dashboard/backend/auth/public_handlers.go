package auth

import (
	"encoding/json"
	"net/http"
	"strings"
)

func bootstrapCanRegisterHandler(svc *Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		canRegister, err := svc.CanBootstrap(r.Context())
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		respondJSON(w, BootstrapStatusResponse{CanRegister: canRegister})
	}
}

func bootstrapRegisterHandler(svc *Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req BootstrapRegistrationRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid body", http.StatusBadRequest)
			return
		}
		if strings.TrimSpace(req.Email) == "" || strings.TrimSpace(req.Password) == "" {
			http.Error(w, "email and password are required", http.StatusBadRequest)
			return
		}

		allowed, err := svc.CanBootstrap(r.Context())
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if !allowed {
			http.Error(w, "bootstrap is disabled", http.StatusConflict)
			return
		}

		hash, err := svc.HashPassword(req.Password)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		user, err := svc.store.CreateUser(r.Context(), req.Email, req.Name, hash, RoleAdmin, "active")
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		token, err := svc.issueTokenForContext(r.Context(), user)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		perms, err := svc.store.GetEffectivePermissions(r.Context(), user.Role, user.ID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		setAuthSessionCookie(w, r, token, svc.ttlDuration)
		writeAudit(r, svc, "user.bootstrap", "/api/auth/bootstrap/register", "")
		respondJSON(w, LoginResponse{Token: token, User: cloneSessionUser(user, perms)})
	}
}

func loginHandler(svc *Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req LoginRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid body", http.StatusBadRequest)
			return
		}

		token, user, err := svc.Login(r.Context(), strings.TrimSpace(req.Email), req.Password)
		if err != nil {
			http.Error(w, err.Error(), http.StatusUnauthorized)
			return
		}

		perms, err := svc.store.GetEffectivePermissions(r.Context(), user.Role, user.ID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		setAuthSessionCookie(w, r, token, svc.ttlDuration)
		respondJSON(w, LoginResponse{Token: token, User: cloneSessionUser(user, perms)})
	}
}

func meHandler(svc *Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		ac, ok := AuthFromContext(r)
		if !ok {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		user, err := svc.GetByID(r.Context(), ac.UserID)
		if err != nil || user == nil {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		respondJSON(w, map[string]any{"user": cloneSessionUser(user, ac.Perms)})
	}
}
