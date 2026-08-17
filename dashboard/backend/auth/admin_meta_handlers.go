package auth

import (
	"database/sql"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"strings"
)

func adminPermissionsHandler(svc *Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ac, ok := AuthFromContext(r)
		if !ok || !ac.Perms[PermUsersManage] {
			http.Error(w, "Forbidden", http.StatusForbidden)
			return
		}
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		perms, err := svc.store.ListRolePermissions(r.Context())
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		respondJSON(w, map[string]any{"rolePermissions": perms, "allPermissions": AllPermissions})
	}
}

func adminAuditLogsHandler(svc *Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ac, ok := AuthFromContext(r)
		if !ok || !ac.Perms[PermUsersManage] {
			http.Error(w, "Forbidden", http.StatusForbidden)
			return
		}
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		legacyResponse := usesLegacyAuditLogResponse(r)
		options, page, err := auditLogOptionsFromRequest(r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if legacyResponse {
			applyLegacyAuditLogDefaults(r, &options)
		}

		logs, total, err := svc.store.QueryAuditLogs(r.Context(), options)
		if err != nil {
			if errors.Is(err, ErrInvalidAuditLogFilter) {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		if legacyResponse {
			respondJSON(w, logs)
			return
		}

		respondJSON(w, AuditLogPageResponse{
			Logs:  logs,
			Total: total,
			Page:  page,
			Limit: options.Limit,
		})
	}
}

func adminUserPasswordHandler(svc *Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ac, ok := AuthFromContext(r)
		if !ok || !ac.Perms[PermUsersManage] {
			http.Error(w, "Forbidden", http.StatusForbidden)
			return
		}
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req struct {
			UserID   string `json:"userId"`
			Password string `json:"password"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid body", http.StatusBadRequest)
			return
		}
		if req.UserID == "" || req.Password == "" {
			http.Error(w, "userId and password are required", http.StatusBadRequest)
			return
		}
		hash, err := svc.HashPassword(req.Password)
		if err != nil {
			writePasswordHashError(w, err)
			return
		}
		idempotencyKey := credentialLifecycleIdempotencyKey(r)
		if idempotencyKey == "" {
			log.Printf(
				"auth credential lifecycle mutation without idempotency key: operation=%s actor_user_id=%s target_user_id=%s",
				CredentialLifecycleAdminPasswordReset,
				ac.UserID,
				req.UserID,
			)
		}
		result, err := svc.store.ResetUserPasswordWithAudit(r.Context(), CredentialLifecycleMutation{
			Operation:          CredentialLifecycleAdminPasswordReset,
			AuditAction:        legacyUserPasswordAuditAction,
			ActorUserID:        ac.UserID,
			TargetUserID:       req.UserID,
			PasswordHash:       hash,
			IdempotencyKey:     idempotencyKey,
			RequestFingerprint: svc.CredentialLifecycleFingerprint(CredentialLifecycleAdminPasswordReset, req.UserID, req.Password),
			Method:             r.Method,
			Path:               "/api/admin/users/password",
			IP:                 r.RemoteAddr,
			UserAgent:          r.UserAgent(),
			StatusCode:         http.StatusOK,
		})
		if err != nil {
			switch {
			case errors.Is(err, sql.ErrNoRows):
				http.Error(w, err.Error(), http.StatusNotFound)
				return
			case errors.Is(err, ErrCredentialLifecycleConflict):
				http.Error(w, err.Error(), http.StatusConflict)
				return
			}
			recordCredentialLifecycleTerminalFailure(CredentialLifecycleAdminPasswordReset, credentialLifecycleFailureStore)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if err := respondJSONWithError(w, map[string]bool{"ok": true, "replayed": result.Replayed}); err != nil {
			recordCredentialLifecycleTerminalFailure(
				CredentialLifecycleAdminPasswordReset,
				credentialLifecycleFailureResponseEncode,
			)
		}
	}
}

func credentialLifecycleIdempotencyKey(r *http.Request) string {
	for _, header := range []string{"Idempotency-Key", "X-Request-ID"} {
		value := strings.TrimSpace(r.Header.Get(header))
		if value != "" {
			return value
		}
	}
	return ""
}
