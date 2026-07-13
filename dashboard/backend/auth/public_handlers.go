package auth

import (
	"errors"
	"net/http"
	"strings"
)

func bootstrapCanRegisterHandler(svc *Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		setAuthNoStoreHeaders(w)
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
			http.Error(w, "authentication service unavailable", http.StatusInternalServerError)
			return
		}

		respondJSON(w, BootstrapStatusResponse{CanRegister: canRegister})
	}
}

// hashBootstrapPassword is an indirection over Service.HashPassword so tests
// can observe that no bcrypt work happens once the bootstrap window is closed.
var hashBootstrapPassword = func(svc *Service, email, password string) (string, error) {
	return svc.HashPasswordForUser(email, password)
}

func bootstrapRegisterHandler(svc *Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		setAuthNoStoreHeaders(w)
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		cookieOnly, modeErr := cookieOnlyAuthResponse(r)
		if modeErr != nil {
			http.Error(w, modeErr.Error(), http.StatusBadRequest)
			return
		}

		if !svc.OpenBootstrapEnabled() {
			http.Error(w, "open bootstrap is disabled; provision an admin via DASHBOARD_ADMIN_* or set DASHBOARD_ALLOW_OPEN_BOOTSTRAP=true to enable web-form bootstrap", http.StatusForbidden)
			return
		}

		var req BootstrapRegistrationRequest
		if err := decodeAuthJSON(w, r, &req); err != nil {
			writeAuthDecodeError(w, err)
			return
		}
		req.Email = strings.TrimSpace(req.Email)
		if req.Email == "" || req.Password == "" {
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
			http.Error(w, "registration unavailable", http.StatusInternalServerError)
			return
		}
		if !canBootstrap {
			http.Error(w, "bootstrap is disabled", http.StatusConflict)
			return
		}

		attempt, retryAfter := svc.limiter.Reserve(req.Email, loginRequestSource(r))
		if attempt == nil {
			writeLoginRateLimit(w, &LoginRateLimitError{RetryAfter: retryAfter})
			return
		}
		defer attempt.Cancel()
		hash, err := hashBootstrapPassword(svc, req.Email, req.Password)
		if err != nil {
			if writePasswordPolicyError(w, err) {
				return
			}
			http.Error(w, "password hashing failed", http.StatusInternalServerError)
			return
		}

		// Atomic check-and-create: only one admin can be created via this path even
		// under concurrent requests (see Service.BootstrapRegister).
		user, err := svc.BootstrapRegister(r.Context(), req.Email, req.Name, hash)
		if errors.Is(err, ErrBootstrapClosed) {
			attempt.Finish()
			http.Error(w, "bootstrap is disabled", http.StatusConflict)
			return
		}
		if err != nil {
			attempt.Finish()
			http.Error(w, "registration failed", http.StatusInternalServerError)
			return
		}
		attempt.Succeed()

		token, err := svc.issueTokenForContext(r.Context(), user)
		if err != nil {
			http.Error(w, "authentication service unavailable", http.StatusInternalServerError)
			return
		}

		perms, err := svc.store.GetEffectivePermissions(r.Context(), user.Role, user.ID)
		if err != nil {
			http.Error(w, "authentication service unavailable", http.StatusInternalServerError)
			return
		}

		if cookieOnly {
			setAuthSessionCookie(w, r, token, svc.ttlDuration)
		}
		writeAudit(r, svc, "user.bootstrap", "/api/auth/bootstrap/register", "")
		respondJSON(w, loginResponse(token, cloneSessionUser(user, perms), cookieOnly))
	}
}

func loginHandler(svc *Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		setAuthNoStoreHeaders(w)
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		cookieOnly, modeErr := cookieOnlyAuthResponse(r)
		if modeErr != nil {
			http.Error(w, modeErr.Error(), http.StatusBadRequest)
			return
		}

		var req LoginRequest
		if err := decodeAuthJSON(w, r, &req); err != nil {
			writeAuthDecodeError(w, err)
			return
		}

		token, user, err := svc.LoginWithSource(
			r.Context(),
			strings.TrimSpace(req.Email),
			req.Password,
			loginRequestSource(r),
		)
		if err != nil {
			var rateErr *LoginRateLimitError
			switch {
			case errors.As(err, &rateErr):
				writeLoginRateLimit(w, rateErr)
			case errors.Is(err, ErrInvalidCredentials):
				http.Error(w, "invalid credentials", http.StatusUnauthorized)
			default:
				http.Error(w, "authentication service unavailable", http.StatusInternalServerError)
			}
			return
		}

		perms, err := svc.store.GetEffectivePermissions(r.Context(), user.Role, user.ID)
		if err != nil {
			http.Error(w, "authentication service unavailable", http.StatusInternalServerError)
			return
		}

		if cookieOnly {
			setAuthSessionCookie(w, r, token, svc.ttlDuration)
		}
		respondJSON(w, loginResponse(token, cloneSessionUser(user, perms), cookieOnly))
	}
}

func meHandler(svc *Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		setAuthNoStoreHeaders(w)
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
		if ac.CredentialSource == CredentialSourceCookie {
			reissueSessionCookie(w, r, svc)
		}

		respondJSON(w, map[string]any{"user": cloneSessionUser(user, ac.Perms)})
	}
}

func changePasswordHandler(svc *Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		setAuthNoStoreHeaders(w)
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		cookieOnly, modeErr := cookieOnlyAuthResponse(r)
		if modeErr != nil {
			http.Error(w, modeErr.Error(), http.StatusBadRequest)
			return
		}
		ac, ok := AuthFromContext(r)
		if !ok {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		var req ChangePasswordRequest
		if err := decodeAuthJSON(w, r, &req); err != nil {
			writeAuthDecodeError(w, err)
			return
		}
		if req.CurrentPassword == "" || req.NewPassword == "" {
			http.Error(w, "currentPassword and newPassword are required", http.StatusBadRequest)
			return
		}

		token, user, err := svc.ChangePasswordWithSource(
			r.Context(),
			ac.UserID,
			req.CurrentPassword,
			req.NewPassword,
			loginRequestSource(r),
		)
		if err != nil {
			var rateErr *LoginRateLimitError
			switch {
			case errors.As(err, &rateErr):
				writeLoginRateLimit(w, rateErr)
			case errors.Is(err, ErrCurrentPasswordFailed):
				// Reserve 401 for middleware session failure. The frontend treats a
				// protected API 401 as logout, so a password typo must be 403.
				http.Error(w, "current password is invalid", http.StatusForbidden)
			case errors.Is(err, ErrPasswordChanged):
				http.Error(w, "password changed concurrently; retry", http.StatusConflict)
			case writePasswordPolicyError(w, err):
			default:
				http.Error(w, "password change failed", http.StatusInternalServerError)
			}
			return
		}

		if cookieOnly {
			setAuthSessionCookie(w, r, token, svc.ttlDuration)
		}
		writeAudit(r, svc, "user.password.self", "/api/auth/password", ac.UserID)
		respondJSON(w, loginResponse(token, cloneSessionUser(user, ac.Perms), cookieOnly))
	}
}
