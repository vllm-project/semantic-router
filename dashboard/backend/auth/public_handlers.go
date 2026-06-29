package auth

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
)

func bootstrapCanRegisterHandler(svc *Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// When neither explicit open bootstrap nor dashboard setup mode is active,
		// always report false. This avoids leaking to unauthenticated callers
		// whether a non-bootstrap instance is freshly deployed and claimable.
		if !svc.OpenBootstrapEnabled() {
			respondJSON(w, BootstrapStatusResponse{CanRegister: false})
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

// hashBootstrapPassword is an indirection over Service.HashPassword so tests
// can observe that no bcrypt work happens once the bootstrap window is closed.
var hashBootstrapPassword = func(svc *Service, password string) (string, error) {
	return svc.HashPassword(password)
}

func bootstrapRegisterHandler(svc *Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		if !svc.OpenBootstrapEnabled() {
			http.Error(w, "open bootstrap is disabled; provision an admin via DASHBOARD_ADMIN_* or set DASHBOARD_ALLOW_OPEN_BOOTSTRAP=true to enable web-form bootstrap", http.StatusForbidden)
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

		// Fast-reject before the expensive bcrypt hash: once an admin exists this
		// request is certain to fail, and hashing first lets unauthenticated
		// callers burn a full cost-12 bcrypt round per request for as long as the
		// bootstrap endpoint stays enabled. This check is a cheap pre-filter
		// only; BootstrapRegister re-checks under its lock, which remains the
		// correctness gate for concurrent requests.
		canBootstrap, err := svc.CanBootstrap(r.Context())
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if !canBootstrap {
			http.Error(w, "bootstrap is disabled", http.StatusConflict)
			return
		}

		hash, err := hashBootstrapPassword(svc, req.Password)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		// Atomic check-and-create: only one admin can be created via this path even
		// under concurrent requests (see Service.BootstrapRegister).
		user, err := svc.BootstrapRegister(r.Context(), req.Email, req.Name, hash)
		if errors.Is(err, ErrBootstrapClosed) {
			http.Error(w, "bootstrap is disabled", http.StatusConflict)
			return
		}
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
