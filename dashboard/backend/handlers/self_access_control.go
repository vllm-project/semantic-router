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
	if strings.TrimSpace(ac.AccessUserID) == "" {
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
	actor := accesscontrol.Actor{ID: ac.AccessUserID, Email: ac.Email}

	switch resource {
	case "overview":
		if r.Method != http.MethodGet {
			methodNotAllowed(w)
			return
		}
		result, err := h.service.SelfOverview(r.Context(), ac.AccessUserID)
		writeAccessResult(w, result, err)
	case "teams":
		if r.Method != http.MethodGet {
			methodNotAllowed(w)
			return
		}
		result, err := h.service.SelfTeams(r.Context(), ac.AccessUserID)
		writeAccessResult(w, map[string]any{"items": result}, err)
	case "api-keys":
		h.handleAPIKeys(w, r, actor, id)
	case "usage":
		if r.Method != http.MethodGet {
			methodNotAllowed(w)
			return
		}
		result, err := h.service.SelfUsage(r.Context(), ac.AccessUserID, boundedUsageSummaryFilter(r))
		writeAccessResult(w, result, err)
	case "request-logs":
		if r.Method != http.MethodGet {
			methodNotAllowed(w)
			return
		}
		if id != "" {
			result, err := h.service.SelfRequestLog(r.Context(), ac.AccessUserID, id)
			writeAccessResult(w, result, err)
			return
		}
		filter := boundedUsageFilter(r)
		result, total, err := h.service.SelfRequestLogs(r.Context(), ac.AccessUserID, filter)
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
			Name string `json:"name"`
		}
		decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 32<<10))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&input); err != nil {
			writeAccessError(w, http.StatusBadRequest, "invalid request")
			return
		}
		result, err := h.service.CreateSelfAPIKey(r.Context(), actor, input.Name)
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
	default:
		methodNotAllowed(w)
	}
}
