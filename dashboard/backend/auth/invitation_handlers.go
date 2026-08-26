package auth

import (
	"database/sql"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
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
		namespaceID, ok := invitationNamespace(w, r)
		if !ok {
			return
		}
		switch r.Method {
		case http.MethodGet:
			items, err := svc.ListInvitations(r.Context(), ac, namespaceID)
			if err != nil {
				writeInvitationAuthorityError(w, err)
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
			idempotencyKey, ok := invitationIdempotencyKey(w, r)
			if !ok {
				return
			}
			item, err := svc.CreateInvitation(r.Context(), ac, invitationInput{
				Email: req.Email, Name: req.Name, Role: req.Role,
				TeamID: req.TeamID, TeamRole: req.TeamRole, NamespaceID: namespaceID,
				IdempotencyKey: idempotencyKey, ExpiresInHours: req.ExpiresInHours,
				SendEmail: req.SendEmail, CreatedBy: ac.UserID,
			})
			if err != nil {
				writeInvitationAuthorityError(w, err)
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
		namespaceID, ok := invitationNamespace(w, r)
		if !ok {
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
			revision, revisionOK := invitationRevision(w, r)
			if !revisionOK {
				return
			}
			idempotencyKey, idempotencyOK := invitationIdempotencyKey(w, r)
			if !idempotencyOK {
				return
			}
			item, err := svc.ResendInvitation(r.Context(), ac, namespaceID, id, idempotencyKey, revision, req.SendEmail)
			if err != nil {
				writeInvitationAuthorityError(w, err)
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
		revision, ok := invitationRevision(w, r)
		if !ok {
			return
		}
		item, err := svc.RevokeInvitation(r.Context(), ac, namespaceID, id, revision)
		if err != nil {
			writeInvitationAuthorityError(w, err)
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
			"email": item.Email, "name": item.Name, "expiresAt": item.ExpiresAt,
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
		accepted, err := svc.AcceptInvitation(r.Context(), req.Token, req.Name, req.Password)
		if err != nil {
			status := http.StatusBadRequest
			if errors.Is(err, ErrInvitationUnavailable) {
				status = http.StatusGone
			}
			http.Error(w, err.Error(), status)
			return
		}
		setAuthSessionCookie(w, r, accepted.AccessToken, svc.ttlDuration)
		writeAudit(r, svc, "member.invitation_accept", "/api/auth/invitations/accept", accepted.User.ID)
		respondJSON(w, struct {
			User       *User                          `json:"user"`
			Onboarding managementapi.OnboardingResult `json:"onboarding"`
		}{User: accepted.User, Onboarding: accepted.Onboarding})
	}
}

func invitationNamespace(w http.ResponseWriter, r *http.Request) (string, bool) {
	values := r.Header.Values(managementapi.HeaderNamespaceID)
	if len(values) != 1 || values[0] != strings.TrimSpace(values[0]) {
		http.Error(w, "namespace is required", http.StatusBadRequest)
		return "", false
	}
	parsed, err := uuid.Parse(values[0])
	if err != nil || parsed.String() != values[0] {
		http.Error(w, "namespace is invalid", http.StatusBadRequest)
		return "", false
	}
	return values[0], true
}

func invitationIdempotencyKey(w http.ResponseWriter, r *http.Request) (string, bool) {
	values := r.Header.Values(managementapi.HeaderIdempotencyKey)
	if len(values) != 1 || values[0] == "" || values[0] != strings.TrimSpace(values[0]) || strings.Contains(values[0], ",") {
		http.Error(w, "Idempotency-Key is required", http.StatusBadRequest)
		return "", false
	}
	return values[0], true
}

func invitationRevision(w http.ResponseWriter, r *http.Request) (uint64, bool) {
	values := r.Header.Values(managementapi.HeaderIfMatch)
	if len(values) != 1 || !strings.HasPrefix(values[0], `"invitation:`) || !strings.HasSuffix(values[0], `"`) {
		http.Error(w, "If-Match is required", http.StatusPreconditionRequired)
		return 0, false
	}
	value := strings.TrimSuffix(strings.TrimPrefix(values[0], `"invitation:`), `"`)
	revision, err := strconv.ParseUint(value, 10, 64)
	if err != nil || revision == 0 {
		http.Error(w, "If-Match is invalid", http.StatusBadRequest)
		return 0, false
	}
	return revision, true
}

func writeInvitationAuthorityError(w http.ResponseWriter, err error) {
	var upstream *InvitationAuthorityError
	switch {
	case errors.Is(err, ErrInvitationUnavailable), errors.Is(err, sql.ErrNoRows):
		http.Error(w, ErrInvitationUnavailable.Error(), http.StatusGone)
	case errors.As(err, &upstream) && upstream.Status >= 400 && upstream.Status < 600:
		if upstream.RequestID != "" {
			w.Header().Set(managementapi.HeaderRequestID, upstream.RequestID)
		}
		http.Error(w, upstream.Error(), upstream.Status)
	case errors.Is(err, ErrInvitationAuthorityUnavailable):
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
	default:
		http.Error(w, err.Error(), http.StatusBadRequest)
	}
}
