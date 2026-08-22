package auth

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
)

const maxInvitationRequestBody = 64 << 10

func decodeInvitationRequest(w http.ResponseWriter, r *http.Request, value any) error {
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxInvitationRequestBody))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(value); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("request body must contain one JSON object")
		}
		return err
	}
	return nil
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
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			respondJSON(w, map[string]any{"items": items})
		case http.MethodPost:
			var req struct {
				Email          string `json:"email"`
				Name           string `json:"name"`
				Role           string `json:"role"`
				TeamID         string `json:"teamId"`
				TeamRole       string `json:"teamRole"`
				ExpiresInHours int    `json:"expiresInHours"`
				SendEmail      bool   `json:"sendEmail"`
			}
			if err := decodeInvitationRequest(w, r, &req); err != nil {
				http.Error(w, "invalid body", http.StatusBadRequest)
				return
			}
			item, err := svc.CreateInvitation(r.Context(), invitationInput{
				Email: req.Email, Name: req.Name, Role: req.Role, TeamID: req.TeamID, TeamRole: req.TeamRole,
				ExpiresInHours: req.ExpiresInHours, SendEmail: req.SendEmail, CreatedBy: ac.UserID,
			})
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			writeAudit(r, svc, "member.invite", "/api/admin/invitations", ac.UserID)
			respondJSON(w, item)
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
		if len(parts) == 2 && parts[1] == "resend" && r.Method == http.MethodPost {
			var req struct {
				SendEmail bool `json:"sendEmail"`
			}
			if err := decodeInvitationRequest(w, r, &req); err != nil {
				http.Error(w, "invalid body", http.StatusBadRequest)
				return
			}
			item, err := svc.ResendInvitation(r.Context(), id, req.SendEmail)
			if err != nil {
				http.Error(w, "pending invitation not found", http.StatusNotFound)
				return
			}
			writeAudit(r, svc, "member.invitation_resend", "/api/admin/invitations/", ac.UserID)
			respondJSON(w, item)
			return
		}
		if len(parts) != 1 || r.Method != http.MethodDelete {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		item, err := svc.store.RevokeInvitation(r.Context(), id)
		if err != nil {
			http.Error(w, "pending invitation not found", http.StatusNotFound)
			return
		}
		writeAudit(r, svc, "member.invitation_revoke", "/api/admin/invitations/", ac.UserID)
		respondJSON(w, item)
	}
}

func invitationInfoHandler(svc *Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		item, err := svc.InvitationInfo(r.Context(), r.URL.Query().Get("token"))
		if err != nil {
			http.Error(w, ErrInvitationUnavailable.Error(), http.StatusGone)
			return
		}
		respondJSON(w, map[string]any{
			"email": item.Email, "name": item.Name, "role": item.Role,
			"teamId": item.TeamID, "teamName": svc.modelTeamName(r.Context(), item.TeamID), "teamRole": item.TeamRole,
			"expiresAt": item.ExpiresAt,
		})
	}
}

func invitationAcceptHandler(svc *Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			Token    string `json:"token"`
			Name     string `json:"name"`
			Password string `json:"password"`
		}
		if err := decodeInvitationRequest(w, r, &req); err != nil {
			http.Error(w, "invalid body", http.StatusBadRequest)
			return
		}
		token, user, err := svc.AcceptInvitation(r.Context(), req.Token, req.Name, req.Password)
		if err != nil {
			status := http.StatusBadRequest
			if errors.Is(err, ErrInvitationUnavailable) {
				status = http.StatusGone
			}
			http.Error(w, err.Error(), status)
			return
		}
		perms, err := svc.store.GetEffectivePermissions(r.Context(), user.Role, user.ID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		setAuthSessionCookie(w, r, token, svc.ttlDuration)
		writeAudit(r, svc, "member.invitation_accept", "/api/auth/invitations/accept", user.ID)
		respondJSON(w, LoginResponse{Token: token, User: cloneSessionUser(user, perms)})
	}
}
