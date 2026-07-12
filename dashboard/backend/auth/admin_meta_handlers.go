package auth

import (
	"database/sql"
	"errors"
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
		if err := decodeAuthJSON(w, r, &req); err != nil {
			writeAuthDecodeError(w, err)
			return
		}
		if req.UserID == "" || req.Password == "" {
			http.Error(w, "userId and password are required", http.StatusBadRequest)
			return
		}
		if err := svc.resetPasswordAuthorized(
			r.Context(),
			usersManageAuthorization(ac),
			req.UserID,
			req.Password,
		); err != nil {
			switch {
			case errors.Is(err, sql.ErrNoRows):
				http.Error(w, "user not found", http.StatusNotFound)
			case errors.Is(err, ErrAuthorizationChanged):
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
			case writePasswordPolicyError(w, err):
			default:
				http.Error(w, "password reset failed", http.StatusInternalServerError)
			}
			return
		}

		writeAudit(r, svc, "user.password", "/api/admin/users/password", ac.UserID)
		respondJSON(w, map[string]bool{"ok": true})
	}
}
