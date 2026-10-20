package auth

import (
	"database/sql"
	"encoding/json"
	"errors"
	"log"
	"net/http"
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
			log.Printf("failed to list role permissions: %v", err)
			http.Error(w, "failed to list permissions", http.StatusInternalServerError)
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
			log.Printf("failed to query audit logs: %v", err)
			http.Error(w, "failed to query audit logs", http.StatusInternalServerError)
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
		if _, err := svc.store.GetUserByID(r.Context(), req.UserID); err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				http.Error(w, "user not found", http.StatusNotFound)
				return
			}
			log.Printf("failed to load user: %v", err)
			http.Error(w, "failed to load user", http.StatusInternalServerError)
			return
		}

		hash, err := svc.HashPassword(req.Password)
		if err != nil {
			writePasswordHashError(w, err)
			return
		}
		if RejectRevokedMutation(w, r) {
			return
		}
		if err := svc.store.UpdatePasswordAuthorized(r.Context(), ac, req.UserID, hash); err != nil {
			if writeAdminMutationAuthorizationError(w, err) {
				return
			}
			log.Printf("failed to update password: %v", err)
			http.Error(w, "failed to update password", http.StatusInternalServerError)
			return
		}

		writeAudit(r, svc, "user.password", "/api/admin/users/password", ac.UserID)
		respondJSON(w, map[string]bool{"ok": true})
	}
}
