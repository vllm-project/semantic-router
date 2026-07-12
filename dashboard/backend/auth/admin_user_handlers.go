package auth

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"strings"
)

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
	page := positiveQueryInt(r, "page", 1, 1_000_000)
	limit := positiveQueryInt(r, "limit", 100, 200)
	options := UserListOptions{
		Status: r.URL.Query().Get("status"),
		Query:  r.URL.Query().Get("q"),
		Sort:   r.URL.Query().Get("sort"),
		Order:  r.URL.Query().Get("order"),
		Limit:  limit,
		Offset: (page - 1) * limit,
	}
	users, err := svc.store.ListUsers(r.Context(), options)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	total, err := svc.store.CountFilteredUsers(r.Context(), options)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	stats, err := svc.store.UserDirectoryStats(r.Context())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	respondJSON(w, ListUsersResponse{
		Users:      users,
		Total:      total,
		Page:       page,
		Limit:      limit,
		Active:     stats.Active,
		Privileged: stats.Privileged,
	})
}

func positiveQueryInt(r *http.Request, key string, fallback, maximum int) int {
	value, err := strconv.Atoi(r.URL.Query().Get(key))
	if err != nil || value < 1 {
		return fallback
	}
	if value > maximum {
		return maximum
	}
	return value
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
