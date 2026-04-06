package auth

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"sort"
	"strconv"
	"strings"
)

func userHasPermission(ctx context.Context, svc *Service, userID, role, status, permission string) (bool, error) {
	if status != defaultUserStatusActive {
		return false, nil
	}

	perms, err := svc.store.GetEffectivePermissions(ctx, role, userID)
	if err != nil {
		return false, err
	}
	return perms[permission], nil
}

func cloneSessionUser(user *User, perms map[string]bool) *User {
	if user == nil {
		return nil
	}

	sessionUser := *user
	if len(perms) == 0 {
		sessionUser.Permissions = nil
		return &sessionUser
	}

	keys := make([]string, 0, len(perms))
	for key, allowed := range perms {
		if allowed {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	sessionUser.Permissions = keys
	return &sessionUser
}

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

		token, err := svc.issueToken(user)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		perms, err := svc.store.GetEffectivePermissions(r.Context(), user.Role, user.ID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		writeSessionCookie(w, r, svc, token)
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

		writeSessionCookie(w, r, svc, token)
		respondJSON(w, LoginResponse{Token: token, User: cloneSessionUser(user, perms)})
	}
}

func logoutHandler(_ *Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		clearSessionCookie(w, r)
		w.WriteHeader(http.StatusNoContent)
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

func adminUsersCollectionHandler(svc *Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ac, ok := AuthFromContext(r)
		if !ok {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		canList := ac.Perms[PermUsersManage] || ac.Perms[PermUsersView]
		if !canList {
			http.Error(w, "Forbidden", http.StatusForbidden)
			return
		}

		switch r.Method {
		case http.MethodGet:
			handleAdminUsersList(w, r, svc)
		case http.MethodPost:
			handleAdminUsersCreate(w, r, svc, ac)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

func handleAdminUsersList(w http.ResponseWriter, r *http.Request, svc *Service) {
	users, err := svc.store.ListUsers(r.Context(), r.URL.Query().Get("status"), 100, 0)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	respondJSON(w, ListUsersResponse{Users: users})
}

func handleAdminUsersCreate(w http.ResponseWriter, r *http.Request, svc *Service, ac AuthContext) {
	if !ac.Perms[PermUsersManage] {
		http.Error(w, "Forbidden", http.StatusForbidden)
		return
	}

	var req struct {
		Email    string `json:"email"`
		Name     string `json:"name"`
		Password string `json:"password"`
		Role     string `json:"role"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid body", http.StatusBadRequest)
		return
	}
	if req.Email == "" || req.Password == "" {
		http.Error(w, "email and password are required", http.StatusBadRequest)
		return
	}

	hash, err := svc.HashPassword(req.Password)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	normalizedRole, err := normalizeRole(req.Role)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	user, err := svc.store.CreateUser(r.Context(), req.Email, req.Name, hash, normalizedRole, "active")
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	writeAudit(r, svc, "user.create", "/api/admin/users", ac.UserID)
	respondJSON(w, user)
}

func adminUserItemHandler(svc *Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ac, ok := AuthFromContext(r)
		if !ok {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		canView := ac.Perms[PermUsersManage] || ac.Perms[PermUsersView]
		if !canView {
			http.Error(w, "Forbidden", http.StatusForbidden)
			return
		}

		userID := strings.TrimPrefix(r.URL.Path, "/api/admin/users/")
		if userID == "" {
			http.Error(w, "user id required", http.StatusBadRequest)
			return
		}

		switch r.Method {
		case http.MethodGet:
			handleAdminUserGet(w, r, svc, userID)
		case http.MethodPatch:
			handleAdminUserPatch(w, r, svc, ac, userID)
		case http.MethodDelete:
			handleAdminUserDelete(w, r, svc, ac, userID)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

func handleAdminUserGet(w http.ResponseWriter, r *http.Request, svc *Service, userID string) {
	user, err := svc.store.GetUserByID(r.Context(), userID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	respondJSON(w, user)
}

func handleAdminUserPatch(
	w http.ResponseWriter,
	r *http.Request,
	svc *Service,
	ac AuthContext,
	userID string,
) {
	if !ac.Perms[PermUsersManage] {
		http.Error(w, "Forbidden", http.StatusForbidden)
		return
	}

	target, err := svc.store.GetUserByID(r.Context(), userID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	var req UpdateUserRequest
	if decodeErr := json.NewDecoder(r.Body).Decode(&req); decodeErr != nil {
		http.Error(w, "invalid body", http.StatusBadRequest)
		return
	}

	normalizedRole, normalizedStatus, err := normalizeUserUpdate(target, req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if userID == ac.UserID && (normalizedRole != target.Role || normalizedStatus != target.Status) {
		http.Error(w, "cannot change your own role or status", http.StatusConflict)
		return
	}

	if validationErr := validateRemainingUserManagers(
		r.Context(),
		svc,
		target,
		normalizedRole,
		normalizedStatus,
	); validationErr != nil {
		http.Error(w, validationErr.Error(), http.StatusConflict)
		return
	}

	user, err := svc.store.UpdateUserRoleOrStatus(r.Context(), userID, normalizedRole, normalizedStatus)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	writeAudit(r, svc, "user.update", "/api/admin/users/", ac.UserID)
	respondJSON(w, user)
}

func handleAdminUserDelete(
	w http.ResponseWriter,
	r *http.Request,
	svc *Service,
	ac AuthContext,
	userID string,
) {
	if !ac.Perms[PermUsersManage] {
		http.Error(w, "Forbidden", http.StatusForbidden)
		return
	}
	if userID == ac.UserID {
		http.Error(w, "cannot delete your own account", http.StatusConflict)
		return
	}

	target, err := svc.store.GetUserByID(r.Context(), userID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	if err := validateRemainingUserManagers(
		r.Context(),
		svc,
		target,
		target.Role,
		"inactive",
	); err != nil {
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}

	if err := svc.store.DeleteUser(r.Context(), userID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	writeAudit(r, svc, "user.delete", "/api/admin/users/", ac.UserID)
	w.WriteHeader(http.StatusNoContent)
}

func normalizeUserUpdate(target *User, req UpdateUserRequest) (string, string, error) {
	nextRole := strings.TrimSpace(req.Role)
	if nextRole == "" {
		nextRole = target.Role
	}
	normalizedRole, err := normalizeRole(nextRole)
	if err != nil {
		return "", "", err
	}

	nextStatus := strings.TrimSpace(req.Status)
	if nextStatus == "" {
		nextStatus = target.Status
	}
	if nextStatus != "active" && nextStatus != "inactive" {
		return "", "", errors.New("status must be active or inactive")
	}

	return normalizedRole, nextStatus, nil
}

func validateRemainingUserManagers(
	ctx context.Context,
	svc *Service,
	target *User,
	nextRole string,
	nextStatus string,
) error {
	currentlyManagesUsers, err := userHasPermission(
		ctx,
		svc,
		target.ID,
		target.Role,
		target.Status,
		PermUsersManage,
	)
	if err != nil || !currentlyManagesUsers {
		return err
	}

	willManageUsers, err := userHasPermission(
		ctx,
		svc,
		target.ID,
		nextRole,
		nextStatus,
		PermUsersManage,
	)
	if err != nil || willManageUsers {
		return err
	}

	remainingManagers, err := svc.store.CountActiveUsersWithPermission(ctx, PermUsersManage, target.ID)
	if err != nil {
		return err
	}
	if remainingManagers == 0 {
		return errors.New("cannot remove the last active user manager")
	}

	return nil
}

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

		limit := 100
		if rawLimit := r.URL.Query().Get("limit"); rawLimit != "" {
			if parsedLimit, err := strconv.Atoi(rawLimit); err == nil {
				limit = parsedLimit
			}
		}

		logs, err := svc.store.ListAuditLogs(
			r.Context(),
			r.URL.Query().Get("userId"),
			r.URL.Query().Get("action"),
			r.URL.Query().Get("resource"),
			limit,
			0,
		)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		respondJSON(w, logs)
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
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}

		hash, err := svc.HashPassword(req.Password)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if err := svc.store.UpdatePassword(r.Context(), req.UserID, hash); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		writeAudit(r, svc, "user.password", "/api/admin/users/password", ac.UserID)
		respondJSON(w, map[string]bool{"ok": true})
	}
}
