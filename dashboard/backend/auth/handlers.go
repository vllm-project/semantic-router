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

type RouteRegistrar interface {
	HandleFunc(string, func(http.ResponseWriter, *http.Request))
}

func registerAdminPolicy(mux RouteRegistrar, contract RouteContract, handler http.HandlerFunc) {
	if policies, ok := mux.(*PolicyMux); ok {
		policies.HandlePolicyFunc(contract, handler)
		return
	}
	mux.HandleFunc(contract.Pattern, handler)
}

func RegisterAdminRoutes(mux RouteRegistrar, svc *Service) {
	registerAdminPolicy(mux, ProtectedRoute("/api/admin/users", PermUsersView, SensitivitySensitive, ResourceOwnerAuth, http.MethodGet), adminUsersCollectionHandler(svc))
	registerAdminPolicy(mux, Route("/api/admin/users/",
		ReadPolicy(http.MethodGet, PermUsersView, SensitivitySensitive, ResourceOwnerAuth),
		DelegatedMutationPolicy(http.MethodPatch, PermUsersManage, "user.update", SensitivitySecret, ResourceOwnerAuth, 64<<10),
		DelegatedMutationPolicy(http.MethodDelete, PermUsersManage, "user.delete", SensitivitySecret, ResourceOwnerAuth, 64<<10),
	), adminUserItemHandler(svc))
	registerAdminPolicy(mux, ProtectedRoute("/api/admin/permissions", PermUsersManage, SensitivitySensitive, ResourceOwnerAuth, http.MethodGet), adminPermissionsHandler(svc))
	registerAdminPolicy(mux, ProtectedRoute("/api/admin/audit-logs", PermUsersManage, SensitivitySecret, ResourceOwnerAuth, http.MethodGet), adminAuditLogsHandler(svc))
	registerAdminPolicy(mux, ProtectedDelegatedAuditRoute("/api/admin/users/password", PermUsersManage, "user.password", SensitivitySecret, ResourceOwnerAuth, 64<<10, http.MethodPost), adminUserPasswordHandler(svc))
	registerAdminPolicy(mux, Route("/api/admin/invitations",
		ReadPolicy(http.MethodGet, PermUsersManage, SensitivitySecret, ResourceOwnerAuth),
		DelegatedMutationPolicy(http.MethodPost, PermUsersManage, "invitation.create", SensitivitySecret, ResourceOwnerAuth, 64<<10),
	), adminInvitationsHandler(svc))
	registerAdminPolicy(mux, Route("/api/admin/invitations/",
		DelegatedMutationPolicy(http.MethodPost, PermUsersManage, "invitation.rotate", SensitivitySecret, ResourceOwnerAuth, 64<<10),
		DelegatedMutationPolicy(http.MethodDelete, PermUsersManage, "invitation.revoke", SensitivitySecret, ResourceOwnerAuth, 64<<10),
	), adminInvitationItemHandler(svc))
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
