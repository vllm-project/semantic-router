package handlers

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/accesscontrol"
	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

type SelfAccessControlHandler struct {
	service *accesscontrol.Service
}

func NewSelfAccessControlHandler(service *accesscontrol.Service) *SelfAccessControlHandler {
	return &SelfAccessControlHandler{service: service}
}

func (h *SelfAccessControlHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	setJSONContentType(w)
	ac, ok := auth.AuthFromContext(r)
	if !ok {
		writeAccessError(w, http.StatusUnauthorized, "authentication required")
		return
	}
	if strings.TrimSpace(ac.UserID) == "" {
		writeAccessError(w, http.StatusForbidden, "model access is not ready for this account")
		return
	}
	path := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/v1/access-control/self"), "/")
	parts := []string{}
	if path != "" {
		parts = strings.Split(path, "/")
	}
	resource := "overview"
	if len(parts) > 0 {
		resource = parts[0]
	}
	id := ""
	if len(parts) > 1 {
		id = parts[1]
	}
	actor := accesscontrol.Actor{ID: ac.UserID, Email: ac.Email}

	switch resource {
	case "overview":
		if r.Method != http.MethodGet {
			methodNotAllowed(w)
			return
		}
		result, err := h.service.SelfOverview(r.Context(), ac.UserID)
		writeAccessResult(w, result, err)
	case "teams":
		switch r.Method {
		case http.MethodGet:
			if id != "" {
				result, err := h.service.GetSelfTeam(r.Context(), ac.UserID, id)
				writeAccessResult(w, result, err)
				return
			}
			result, err := h.service.SelfTeamCatalog(r.Context(), ac.UserID)
			writeAccessResult(w, result, err)
		case http.MethodPut:
			if id == "" {
				writeAccessError(w, http.StatusBadRequest, "Team id is required")
				return
			}
			var item accesscontrol.Team
			if !decodeAccessJSON(w, r, &item) {
				return
			}
			item.ID = id
			result, err := h.service.SaveSelfTeam(r.Context(), actor, item)
			writeAccessResult(w, result, err)
		default:
			methodNotAllowed(w)
		}
	case "api-keys":
		h.handleAPIKeys(w, r, actor, id)
	case "usage":
		if r.Method != http.MethodGet {
			methodNotAllowed(w)
			return
		}
		result, err := h.service.SelfUsage(r.Context(), ac.UserID, boundedUsageSummaryFilter(r))
		writeAccessResult(w, result, err)
	case "request-logs":
		if r.Method != http.MethodGet {
			methodNotAllowed(w)
			return
		}
		if id != "" {
			result, err := h.service.SelfRequestLog(r.Context(), ac.UserID, id)
			writeAccessResult(w, result, err)
			return
		}
		filter := boundedUsageFilter(r)
		result, total, err := h.service.SelfRequestLogs(r.Context(), ac.UserID, filter)
		writeAccessPage(w, result, total, filter, err)
	default:
		writeAccessError(w, http.StatusNotFound, "resource not found")
	}
}

func (h *SelfAccessControlHandler) handleAPIKeys(
	w http.ResponseWriter,
	r *http.Request,
	actor accesscontrol.Actor,
	id string,
) {
	action := ""
	parts := strings.Split(strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/v1/access-control/self/api-keys"), "/"), "/")
	if len(parts) > 1 {
		action = parts[1]
	}
	switch r.Method {
	case http.MethodGet:
		if id != "" && action == "secret" {
			secret, err := h.service.RevealSelfAPIKey(r.Context(), actor, id)
			writeAccessResult(w, map[string]string{"secret": secret}, err)
			return
		}
		if id != "" {
			result, err := h.service.GetSelfAPIKey(r.Context(), actor.ID, id)
			writeAccessResult(w, result, err)
			return
		}
		result, err := h.service.ListSelfAPIKeys(r.Context(), actor.ID)
		writeAccessResult(w, map[string]any{"items": result, "total": len(result), "limit": 100, "offset": 0, "hasMore": false}, err)
	case http.MethodPost:
		if id != "" && action == "rotate" {
			result, err := h.service.RotateSelfAPIKey(r.Context(), actor, id)
			writeAccessResult(w, result, err)
			return
		}
		if id != "" {
			methodNotAllowed(w)
			return
		}
		var input struct {
			Name          string `json:"name"`
			OwnerType     string `json:"ownerType"`
			OwnerID       string `json:"ownerId"`
			ContextTeamID string `json:"contextTeamId"`
		}
		decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 32<<10))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&input); err != nil {
			writeAccessError(w, http.StatusBadRequest, "invalid request")
			return
		}
		result, err := h.service.CreateSelfAPIKey(
			r.Context(), actor, input.Name, input.OwnerType, input.OwnerID, input.ContextTeamID,
		)
		writeAccessResult(w, result, err)
	case http.MethodPatch:
		if id == "" {
			writeAccessError(w, http.StatusBadRequest, "API key id is required")
			return
		}
		var input struct {
			Status string `json:"status"`
		}
		if !decodeAccessJSON(w, r, &input) {
			return
		}
		result, err := h.service.SetSelfAPIKeyStatus(r.Context(), actor, id, input.Status)
		writeAccessResult(w, result, err)
	case http.MethodDelete:
		if id == "" || action != "" {
			writeAccessError(w, http.StatusBadRequest, "API key id is required")
			return
		}
		writeAccessResult(w, map[string]bool{"deleted": true}, h.service.DeleteSelfAPIKey(r.Context(), actor, id))
	default:
		methodNotAllowed(w)
	}
}
