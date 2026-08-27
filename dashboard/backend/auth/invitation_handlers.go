package auth

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
)

type invitationMutationResponse struct {
	Invitation *Invitation `json:"invitation"`
	Token      string      `json:"token,omitempty"`
}

type publicInvitationInfo struct {
	Email     string `json:"email"`
	Name      string `json:"name"`
	Role      string `json:"role"`
	ExpiresAt int64  `json:"expiresAt"`
}

func adminInvitationsHandler(svc *Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ac, ok := AuthFromContext(r)
		if !ok || !ac.Perms[PermUsersManage] {
			http.Error(w, "Forbidden", http.StatusForbidden)
			return
		}
		switch r.Method {
		case http.MethodGet:
			items, err := svc.store.ListInvitations(r.Context())
			if err != nil {
				http.Error(w, "failed to list invitations", http.StatusInternalServerError)
				return
			}
			respondJSON(w, map[string]any{"invitations": items})
		case http.MethodPost:
			var request struct {
				Email string `json:"email"`
				Name  string `json:"name"`
				Role  string `json:"role"`
			}
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				http.Error(w, "invalid body", http.StatusBadRequest)
				return
			}
			item, token, err := svc.CreateInvitation(r.Context(), request.Email, request.Name, request.Role, ac.UserID)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			writeAudit(r, svc, "invitation.create", "dashboard-invitations", ac.UserID)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusCreated)
			respondJSON(w, invitationMutationResponse{Invitation: item, Token: token})
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

func adminInvitationItemHandler(svc *Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ac, ok := AuthFromContext(r)
		if !ok || !ac.Perms[PermUsersManage] {
			http.Error(w, "Forbidden", http.StatusForbidden)
			return
		}
		path := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/admin/invitations/"), "/")
		parts := strings.Split(path, "/")
		if len(parts) == 0 || parts[0] == "" {
			http.Error(w, "invitation id required", http.StatusBadRequest)
			return
		}
		id := parts[0]
		switch {
		case r.Method == http.MethodPost && len(parts) == 2 && parts[1] == "rotate":
			item, token, err := svc.RotateInvitation(r.Context(), id)
			if err != nil {
				writeInvitationError(w, err)
				return
			}
			writeAudit(r, svc, "invitation.rotate", "dashboard-invitations/"+id, ac.UserID)
			respondJSON(w, invitationMutationResponse{Invitation: item, Token: token})
		case r.Method == http.MethodDelete && len(parts) == 1:
			if err := svc.store.RevokeInvitation(r.Context(), id); err != nil {
				writeInvitationError(w, err)
				return
			}
			writeAudit(r, svc, "invitation.revoke", "dashboard-invitations/"+id, ac.UserID)
			w.WriteHeader(http.StatusNoContent)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

func publicInvitationHandler(svc *Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		path := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/auth/invitations/"), "/")
		parts := strings.Split(path, "/")
		if len(parts) == 0 || parts[0] == "" {
			http.Error(w, "invitation token required", http.StatusBadRequest)
			return
		}
		token := parts[0]
		switch {
		case r.Method == http.MethodGet && len(parts) == 1:
			item, err := svc.InvitationInfo(r.Context(), token)
			if err != nil {
				writeInvitationError(w, err)
				return
			}
			respondJSON(w, publicInvitationInfo{
				Email: item.Email, Name: item.Name, Role: item.Role, ExpiresAt: item.ExpiresAt,
			})
		case r.Method == http.MethodPost && len(parts) == 2 && parts[1] == "accept":
			var request struct {
				Password string `json:"password"`
			}
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				http.Error(w, "invalid body", http.StatusBadRequest)
				return
			}
			signed, user, err := svc.AcceptInvitation(r.Context(), token, request.Password)
			if err != nil {
				if errors.Is(err, ErrPasswordTooLong) || errors.Is(err, ErrPasswordTooShort) {
					writePasswordHashError(w, err)
					return
				}
				writeInvitationError(w, err)
				return
			}
			perms, err := svc.store.GetEffectivePermissions(r.Context(), user.Role, user.ID)
			if err != nil {
				http.Error(w, "failed to open session", http.StatusInternalServerError)
				return
			}
			setAuthSessionCookie(w, r, signed, svc.ttlDuration)
			respondJSON(w, LoginResponse{Token: signed, User: cloneSessionUser(user, perms)})
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

func writeInvitationError(w http.ResponseWriter, err error) {
	if errors.Is(err, ErrInvitationUnavailable) {
		http.Error(w, "invitation is no longer available", http.StatusGone)
		return
	}
	http.Error(w, err.Error(), http.StatusBadRequest)
}
