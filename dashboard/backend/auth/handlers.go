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
	Token string `json:"token,omitempty"`
	User  *User  `json:"user"`
}

type ListUsersResponse struct {
	Users []*User `json:"users"`
}

type BootstrapStatusResponse struct {
	CanRegister bool `json:"canRegister"`
}

type UpdateUserRequest struct {
	Role   string `json:"role"`
	Status string `json:"status"`
}

func AuthRoutes(svc *Service) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/auth/bootstrap/can-register", bootstrapCanRegisterHandler(svc))
	mux.HandleFunc("/api/auth/bootstrap/register", bootstrapRegisterHandler(svc))
	mux.HandleFunc("/api/auth/login", loginHandler(svc))
	mux.HandleFunc("/api/auth/login/", loginHandler(svc))
	mux.HandleFunc("/api/auth/logout", logoutHandler(svc))
	mux.HandleFunc("/api/auth/logout/", logoutHandler(svc))
	mux.HandleFunc("/api/auth/me", meHandler(svc))
	mux.HandleFunc("/api/auth/me/", meHandler(svc))

	return mux
}

func RegisterAdminRoutes(mux *http.ServeMux, svc *Service) {
	mux.HandleFunc("/api/admin/users", adminUsersCollectionHandler(svc))
	mux.HandleFunc("/api/admin/users/", adminUserItemHandler(svc))
	mux.HandleFunc("/api/admin/permissions", adminPermissionsHandler(svc))
	mux.HandleFunc("/api/admin/audit-logs", adminAuditLogsHandler(svc))
	mux.HandleFunc("/api/admin/users/password", adminUserPasswordHandler(svc))
}

func writeAudit(r *http.Request, svc *Service, action, resource, actorID string) {
	_ = svc.store.AddAuditLog(r.Context(), AuditLog{
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
}

func respondJSON(w http.ResponseWriter, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	if err := enc.Encode(payload); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}
