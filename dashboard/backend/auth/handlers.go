package auth

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
)

type LoginRequest struct {
	Email    string `json:"email"`
	Password string `json:"password"`
}

type LoginResponse struct {
	Token string `json:"token"`
	User  *User  `json:"user"`
}

type ListUsersResponse struct {
	Users []*User `json:"users"`
}

type UpdateUserRequest struct {
	Role   string `json:"role"`
	Status string `json:"status"`
}

func AuthRoutes(svc *Service) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/auth/login", func(w http.ResponseWriter, r *http.Request) {
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
		respondJSON(w, LoginResponse{Token: token, User: user})
	})

	mux.HandleFunc("/api/auth/me", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		ac, ok := AuthFromContext(r)
		if !ok {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
		u, err := svc.GetByID(r.Context(), ac.UserID)
		if err != nil || u == nil {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
		respondJSON(w, map[string]any{"user": u})
	})

	return mux
}

func RegisterAdminRoutes(mux *http.ServeMux, svc *Service) {
	mux.HandleFunc("/api/admin/users", func(w http.ResponseWriter, r *http.Request) {
		ac, ok := AuthFromContext(r)
		if !ok || !ac.Perms[PermUsersManage] {
			http.Error(w, "Forbidden", http.StatusForbidden)
			return
		}

		switch r.Method {
		case http.MethodGet:
			status := r.URL.Query().Get("status")
			users, err := svc.store.ListUsers(r.Context(), status, 100, 0)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			respondJSON(w, ListUsersResponse{Users: users})
		case http.MethodPost:
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
			h, err := svc.HashPassword(req.Password)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			u, err := svc.store.CreateUser(r.Context(), req.Email, req.Name, h, req.Role, "active")
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			respondJSON(w, u)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	})

	mux.HandleFunc("/api/admin/users/", func(w http.ResponseWriter, r *http.Request) {
		ac, ok := AuthFromContext(r)
		if !ok || !ac.Perms[PermUsersManage] {
			http.Error(w, "Forbidden", http.StatusForbidden)
			return
		}

		id := strings.TrimPrefix(r.URL.Path, "/api/admin/users/")
		if id == "" {
			http.Error(w, "user id required", http.StatusBadRequest)
			return
		}

		switch r.Method {
		case http.MethodGet:
			u, err := svc.store.GetUserByID(r.Context(), id)
			if err != nil {
				http.Error(w, err.Error(), http.StatusNotFound)
				return
			}
			respondJSON(w, u)
		case http.MethodPatch:
			var req UpdateUserRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, "invalid body", http.StatusBadRequest)
				return
			}
			u, err := svc.store.UpdateUserRoleOrStatus(r.Context(), id, req.Role, req.Status)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			respondJSON(w, u)
		case http.MethodDelete:
			if err := svc.store.DeleteUser(r.Context(), id); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			w.WriteHeader(http.StatusNoContent)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	})

	mux.HandleFunc("/api/admin/permissions", func(w http.ResponseWriter, r *http.Request) {
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
	})

	mux.HandleFunc("/api/admin/audit-logs", func(w http.ResponseWriter, r *http.Request) {
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
		if l := r.URL.Query().Get("limit"); l != "" {
			if v, err := strconv.Atoi(l); err == nil {
				limit = v
			}
		}
		logs, err := svc.store.ListAuditLogs(r.Context(), r.URL.Query().Get("userId"), r.URL.Query().Get("action"), r.URL.Query().Get("resource"), limit, 0)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		respondJSON(w, logs)
	})

	mux.HandleFunc("/api/admin/users/password", func(w http.ResponseWriter, r *http.Request) {
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
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if err := svc.store.UpdatePassword(r.Context(), req.UserID, hash); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		respondJSON(w, map[string]bool{"ok": true})
	})
}

func respondJSON(w http.ResponseWriter, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	if err := enc.Encode(payload); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func withContext(ctx context.Context, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		next(w, r.WithContext(ctx))
	}
}
