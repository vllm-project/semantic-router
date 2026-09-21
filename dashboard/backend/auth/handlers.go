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
	mux.HandleFunc("/api/auth/invitations/", publicInvitationHandler(svc))

	return mux
}

// RegisterAdminRoutes binds the user-administration API to its authorization
// contracts. Handlers that write their own audit rows declare delegated audit
// so the row carries the acted-on user rather than only the route.
func RegisterAdminRoutes(routes *PolicyMux, svc *Service) {
	const maxAdminBodyBytes = 64 << 10
	routes.HandleFunc(
		ProtectedRoute("/api/admin/users", PermUsersView, SensitivitySensitive, ResourceOwnerAuth, http.MethodGet),
		adminUsersCollectionHandler(svc),
	)
	routes.HandleFunc(
		Route("/api/admin/users/{id}",
			ReadPolicy(http.MethodGet, PermUsersView, SensitivitySensitive, ResourceOwnerAuth),
			DelegatedMutationPolicy(http.MethodPatch, PermUsersManage, "user.update", SensitivitySecret, ResourceOwnerAuth, maxAdminBodyBytes),
			DelegatedMutationPolicy(http.MethodDelete, PermUsersManage, "user.delete", SensitivitySecret, ResourceOwnerAuth, NoBodyLimit),
		),
		adminUserItemHandler(svc),
	)
	routes.HandleFunc(
		ProtectedDelegatedMutationRoute("/api/admin/users/password", PermUsersManage, "user.password", SensitivitySecret, ResourceOwnerAuth, maxAdminBodyBytes, http.MethodPost),
		adminUserPasswordHandler(svc),
	)
	routes.HandleFunc(
		ProtectedRoute("/api/admin/permissions", PermUsersManage, SensitivitySensitive, ResourceOwnerAuth, http.MethodGet),
		adminPermissionsHandler(svc),
	)
	routes.HandleFunc(
		ProtectedRoute("/api/admin/audit-logs", PermUsersManage, SensitivitySensitive, ResourceOwnerAuth, http.MethodGet),
		adminAuditLogsHandler(svc),
	)
	routes.HandleFunc(
		Route("/api/admin/invitations",
			ReadPolicy(http.MethodGet, PermUsersManage, SensitivitySensitive, ResourceOwnerAuth),
			DelegatedMutationPolicy(http.MethodPost, PermUsersManage, "invitation.create", SensitivitySecret, ResourceOwnerAuth, maxAdminBodyBytes),
		),
		adminInvitationsHandler(svc),
	)
	routes.HandleGroup([]RouteContract{
		ProtectedDelegatedMutationRoute("/api/admin/invitations/{id}", PermUsersManage, "invitation.revoke", SensitivitySecret, ResourceOwnerAuth, NoBodyLimit, http.MethodDelete),
		ProtectedDelegatedMutationRoute("/api/admin/invitations/{id}/rotate", PermUsersManage, "invitation.rotate", SensitivitySecret, ResourceOwnerAuth, NoBodyLimit, http.MethodPost),
	}, adminInvitationItemHandler(svc))
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
