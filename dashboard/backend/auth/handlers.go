package auth

import (
	"encoding/json"
	"net/http"
	"time"
)

type LoginRequest struct {
	Email    string `json:"email"`
	Password string `json:"password"`
}

type BootstrapRegistrationRequest struct {
	Email    string `json:"email"`
	Password string `json:"password"`
	Name     string `json:"name"`
}

type LoginResponse struct {
	Token string `json:"token"`
	User  *User  `json:"user"`
}

type ChangePasswordRequest struct {
	CurrentPassword string `json:"currentPassword"`
	NewPassword     string `json:"newPassword"`
}

type ListUsersResponse struct {
	Users      []*User `json:"users"`
	Total      int     `json:"total"`
	Page       int     `json:"page"`
	Limit      int     `json:"limit"`
	Active     int     `json:"active"`
	Privileged int     `json:"privileged"`
}

type BootstrapStatusResponse struct {
	CanRegister bool `json:"canRegister"`
}

type UpdateUserRequest struct {
	Role   *string `json:"role"`
	Status *string `json:"status"`
}

func AuthRoutes(svc *Service) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/auth/bootstrap/can-register", withAuthNoStore(bootstrapCanRegisterHandler(svc)))
	mux.HandleFunc("/api/auth/bootstrap/register", withAuthNoStore(bootstrapRegisterHandler(svc)))
	mux.HandleFunc("/api/auth/login", withAuthNoStore(loginHandler(svc)))
	mux.HandleFunc("/api/auth/login/", withAuthNoStore(loginHandler(svc)))
	mux.HandleFunc("/api/auth/logout", withAuthNoStore(logoutHandler(svc)))
	mux.HandleFunc("/api/auth/logout/", withAuthNoStore(logoutHandler(svc)))
	mux.HandleFunc("/api/auth/me", withAuthNoStore(meHandler(svc)))
	mux.HandleFunc("/api/auth/me/", withAuthNoStore(meHandler(svc)))
	mux.HandleFunc("/api/auth/password", withAuthNoStore(changePasswordHandler(svc)))
	mux.HandleFunc("/api/auth/password/", withAuthNoStore(changePasswordHandler(svc)))

	return mux
}

func RegisterAdminRoutes(mux *http.ServeMux, svc *Service) {
	mux.HandleFunc("/api/admin/users", withAuthNoStore(adminUsersCollectionHandler(svc)))
	mux.HandleFunc("/api/admin/users/", withAuthNoStore(adminUserItemHandler(svc)))
	mux.HandleFunc("/api/admin/permissions", withAuthNoStore(adminPermissionsHandler(svc)))
	mux.HandleFunc("/api/admin/audit-logs", withAuthNoStore(adminAuditLogsHandler(svc)))
	mux.HandleFunc("/api/admin/users/password", withAuthNoStore(adminUserPasswordHandler(svc)))
}

func writeAudit(r *http.Request, svc *Service, action, resource, actorID string) {
	err := svc.store.AddAuditLog(r.Context(), AuditLog{
		UserID:     actorID,
		Action:     action,
		Resource:   resource,
		Method:     r.Method,
		Path:       r.URL.Path,
		IP:         r.RemoteAddr,
		UserAgent:  r.UserAgent(),
		StatusCode: http.StatusOK,
		CreatedAt:  time.Now().Unix(),
	})
	reportAuditPersistenceError(err)
}

func respondJSON(w http.ResponseWriter, payload interface{}) {
	setAuthNoStoreHeaders(w)
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	if err := enc.Encode(payload); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}
