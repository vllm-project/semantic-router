package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/accesscontrol"
	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

type AccessControlHandler struct {
	service *accesscontrol.Service
}

func NewAccessControlHandler(service *accesscontrol.Service) *AccessControlHandler {
	return &AccessControlHandler{service: service}
}

func (h *AccessControlHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	setJSONContentType(w)
	path := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/v1/access-control"), "/")
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
	action := ""
	if len(parts) > 2 {
		action = parts[2]
	}

	switch resource {
	case "overview":
		h.handleOverview(w, r)
	case "users":
		h.handleUsers(w, r, id)
	case "teams":
		h.handleTeams(w, r, id)
	case "api-keys":
		h.handleAPIKeys(w, r, id, action)
	case "access-groups":
		h.handleAccessGroups(w, r, id)
	case "budgets":
		h.handleBudgets(w, r, id)
	case "usage":
		h.handleUsageSummary(w, r)
	case "request-logs":
		h.handleRequestLogs(w, r, id)
	case "audit-logs":
		h.handleAuditLogs(w, r)
	default:
		writeAccessError(w, http.StatusNotFound, "resource not found")
	}
}

func (h *AccessControlHandler) handleOverview(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	item, err := h.service.Overview(r.Context())
	writeAccessResult(w, item, err)
}

func (h *AccessControlHandler) handleUsers(w http.ResponseWriter, r *http.Request, id string) {
	switch r.Method {
	case http.MethodGet:
		if id != "" {
			item, err := h.service.GetUser(r.Context(), id)
			writeAccessResult(w, item, err)
			return
		}
		filter := accessListFilter(r)
		items, total, err := h.service.ListUsers(r.Context(), filter)
		writeAccessPage(w, items, total, filter, err)
	case http.MethodPost, http.MethodPut:
		var item accesscontrol.User
		if !decodeAccessJSON(w, r, &item) {
			return
		}
		if id != "" {
			item.ID = id
		}
		result, err := h.service.SaveUser(r.Context(), accessActor(r), item)
		writeAccessResult(w, result, err)
	case http.MethodDelete:
		if id == "" {
			writeAccessError(w, http.StatusBadRequest, "user id is required")
			return
		}
		writeAccessResult(w, map[string]bool{"deleted": true}, h.service.DeleteUser(r.Context(), accessActor(r), id))
	default:
		methodNotAllowed(w)
	}
}

func (h *AccessControlHandler) handleTeams(w http.ResponseWriter, r *http.Request, id string) {
	switch r.Method {
	case http.MethodGet:
		if id != "" {
			item, err := h.service.GetTeam(r.Context(), id)
			writeAccessResult(w, item, err)
			return
		}
		filter := accessListFilter(r)
		items, total, err := h.service.ListTeams(r.Context(), filter)
		writeAccessPage(w, items, total, filter, err)
	case http.MethodPost, http.MethodPut:
		var item accesscontrol.Team
		if !decodeAccessJSON(w, r, &item) {
			return
		}
		if id != "" {
			item.ID = id
		}
		result, err := h.service.SaveTeam(r.Context(), accessActor(r), item)
		writeAccessResult(w, result, err)
	case http.MethodDelete:
		if id == "" {
			writeAccessError(w, http.StatusBadRequest, "team id is required")
			return
		}
		writeAccessResult(w, map[string]bool{"deleted": true}, h.service.DeleteTeam(r.Context(), accessActor(r), id))
	default:
		methodNotAllowed(w)
	}
}

func (h *AccessControlHandler) handleAPIKeys(w http.ResponseWriter, r *http.Request, id, action string) {
	context, _ := auth.AuthFromContext(r)
	canManage := context.Perms[auth.PermAccessManage]
	switch r.Method {
	case http.MethodGet:
		if action == "secret" {
			if !canManage {
				writeAccessError(w, http.StatusForbidden, "API key secrets require administrator access")
				return
			}
			secret, err := h.service.RevealAPIKey(r.Context(), accessActor(r), id)
			writeAccessResult(w, map[string]string{"secret": secret}, err)
			return
		}
		if action != "" {
			writeAccessError(w, http.StatusNotFound, "resource not found")
			return
		}
		if id != "" {
			item, err := h.service.GetAPIKey(r.Context(), id)
			writeAccessResult(w, item, err)
			return
		}
		filter := accessListFilter(r)
		items, total, err := h.service.ListAPIKeys(r.Context(), filter)
		writeAccessPage(w, items, total, filter, err)
	case http.MethodPost:
		if action == "rotate" && id != "" {
			result, err := h.service.RotateAPIKey(r.Context(), accessActor(r), id)
			writeAccessResult(w, result, err)
			return
		}
		if id != "" || action != "" {
			writeAccessError(w, http.StatusNotFound, "resource not found")
			return
		}
		var item accesscontrol.APIKey
		if !decodeAccessJSON(w, r, &item) {
			return
		}
		result, err := h.service.CreateAPIKey(r.Context(), accessActor(r), item)
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
		result, err := h.service.SetAPIKeyStatus(r.Context(), accessActor(r), id, input.Status)
		writeAccessResult(w, result, err)
	case http.MethodPut:
		if id == "" {
			writeAccessError(w, http.StatusBadRequest, "API key id is required")
			return
		}
		var item accesscontrol.APIKey
		if !decodeAccessJSON(w, r, &item) {
			return
		}
		item.ID = id
		result, err := h.service.UpdateAPIKey(r.Context(), accessActor(r), item)
		writeAccessResult(w, result, err)
	default:
		methodNotAllowed(w)
	}
}

func (h *AccessControlHandler) handleAccessGroups(w http.ResponseWriter, r *http.Request, id string) {
	switch r.Method {
	case http.MethodGet:
		if id != "" {
			item, err := h.service.GetAccessGroup(r.Context(), id)
			writeAccessResult(w, item, err)
			return
		}
		filter := accessListFilter(r)
		items, total, err := h.service.ListAccessGroups(r.Context(), filter)
		writeAccessPage(w, items, total, filter, err)
	case http.MethodPost, http.MethodPut:
		var item accesscontrol.AccessGroup
		if !decodeAccessJSON(w, r, &item) {
			return
		}
		if id != "" {
			item.ID = id
		}
		result, err := h.service.SaveAccessGroup(r.Context(), accessActor(r), item)
		writeAccessResult(w, result, err)
	case http.MethodDelete:
		if id == "" {
			writeAccessError(w, http.StatusBadRequest, "access group id is required")
			return
		}
		writeAccessResult(w, map[string]bool{"deleted": true}, h.service.DeleteAccessGroup(r.Context(), accessActor(r), id))
	default:
		methodNotAllowed(w)
	}
}

func (h *AccessControlHandler) handleBudgets(w http.ResponseWriter, r *http.Request, id string) {
	switch r.Method {
	case http.MethodGet:
		if id != "" {
			item, err := h.service.GetBudget(r.Context(), id)
			writeAccessResult(w, item, err)
			return
		}
		filter := accessListFilter(r)
		items, total, err := h.service.ListBudgets(r.Context(), filter)
		writeAccessPage(w, items, total, filter, err)
	case http.MethodPost, http.MethodPut:
		var item accesscontrol.Budget
		if !decodeAccessJSON(w, r, &item) {
			return
		}
		if id != "" {
			item.ID = id
		}
		result, err := h.service.SaveBudget(r.Context(), accessActor(r), item)
		writeAccessResult(w, result, err)
	case http.MethodDelete:
		if id == "" {
			writeAccessError(w, http.StatusBadRequest, "budget id is required")
			return
		}
		writeAccessResult(w, map[string]bool{"deleted": true}, h.service.DeleteBudget(r.Context(), accessActor(r), id))
	default:
		methodNotAllowed(w)
	}
}

func (h *AccessControlHandler) handleUsageSummary(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	filter := boundedUsageSummaryFilter(r)
	item, err := h.service.Store().UsageSummary(r.Context(), filter)
	writeAccessResult(w, item, err)
}

func (h *AccessControlHandler) handleRequestLogs(w http.ResponseWriter, r *http.Request, id string) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	if id != "" {
		item, err := h.service.Store().GetUsage(r.Context(), id)
		if err == nil && !canViewAccessLogPayload(r) {
			item.Metadata = redactAccessLogMetadata(item.Metadata)
		}
		writeAccessResult(w, item, err)
		return
	}
	filter := boundedUsageFilter(r)
	items, err := h.service.Store().ListUsage(r.Context(), filter)
	if err != nil {
		writeAccessResult(w, nil, err)
		return
	}
	for index := range items {
		items[index].Metadata = nil
	}
	total, err := h.service.Store().CountUsage(r.Context(), filter)
	writeAccessPage(w, items, total, filter, err)
}

func canViewAccessLogPayload(r *http.Request) bool {
	context, ok := auth.AuthFromContext(r)
	return ok && context.Perms[auth.PermConfigWrite]
}

func redactAccessLogMetadata(metadata map[string]any) map[string]any {
	if len(metadata) == 0 {
		return nil
	}
	redacted := make(map[string]any, len(metadata))
	for key, value := range metadata {
		if key == "request" || key == "response" {
			continue
		}
		redacted[key] = value
	}
	redacted["payloadRedacted"] = true
	return redacted
}

func (h *AccessControlHandler) handleAuditLogs(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	filter := accessListFilter(r)
	items, err := h.service.Store().ListAudit(r.Context(), filter)
	if err != nil {
		writeAccessResult(w, nil, err)
		return
	}
	total, err := h.service.Store().CountAudit(r.Context(), filter)
	writeAccessPage(w, items, total, filter, err)
}

func accessActor(r *http.Request) accesscontrol.Actor {
	context, ok := auth.AuthFromContext(r)
	if !ok {
		return accesscontrol.Actor{}
	}
	return accesscontrol.Actor{ID: context.UserID, Email: context.Email}
}

func accessListFilter(r *http.Request) accesscontrol.ListFilter {
	query := r.URL.Query()
	filter := accesscontrol.ListFilter{
		Query: query.Get("q"), UserID: query.Get("userId"), TeamID: query.Get("teamId"),
		KeyID: query.Get("keyId"), Model: query.Get("model"), Granularity: query.Get("granularity"),
	}
	filter.Limit, _ = strconv.Atoi(query.Get("limit"))
	filter.Offset, _ = strconv.Atoi(query.Get("offset"))
	filter.TimezoneOffsetMinutes, _ = strconv.Atoi(query.Get("timezoneOffset"))
	if filter.TimezoneOffsetMinutes < -840 || filter.TimezoneOffsetMinutes > 840 {
		filter.TimezoneOffsetMinutes = 0
	}
	if value, err := time.Parse(time.RFC3339, query.Get("from")); err == nil {
		filter.From = &value
	}
	if value, err := time.Parse(time.RFC3339, query.Get("to")); err == nil {
		filter.To = &value
	}
	return filter
}

func boundedUsageFilter(r *http.Request) accesscontrol.ListFilter {
	filter := accessListFilter(r)
	now := time.Now().UTC()
	earliest := now.Add(-90 * 24 * time.Hour)
	if filter.From == nil {
		from := now.Add(-24 * time.Hour)
		filter.From = &from
	} else if filter.From.Before(earliest) {
		filter.From = &earliest
	}
	if filter.To != nil && filter.To.After(now) {
		filter.To = &now
	}
	return filter
}

func boundedUsageSummaryFilter(r *http.Request) accesscontrol.ListFilter {
	filter := accessListFilter(r)
	now := time.Now().UTC()
	earliest := now.Add(-366 * 24 * time.Hour)
	if filter.From == nil {
		from := now.Add(-24 * time.Hour)
		filter.From = &from
	} else if filter.From.Before(earliest) {
		filter.From = &earliest
	}
	if filter.To == nil || filter.To.After(now) {
		filter.To = &now
	}
	return filter
}

func decodeAccessJSON(w http.ResponseWriter, r *http.Request, target any) bool {
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 2<<20))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		writeAccessError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return false
	}
	return true
}

func writeAccessResult(w http.ResponseWriter, value any, err error) {
	if err != nil {
		log.Printf("access-control management operation failed: %v", err)
		status, message := accesscontrol.PublicError(err)
		writeAccessError(w, status, message)
		return
	}
	_ = json.NewEncoder(w).Encode(value)
}

func writeAccessPage(w http.ResponseWriter, items any, total int64, filter accesscontrol.ListFilter, err error) {
	if err != nil {
		writeAccessResult(w, nil, err)
		return
	}
	limit := filter.Limit
	if limit <= 0 || limit > 100 {
		limit = 100
	}
	offset := filter.Offset
	if offset < 0 {
		offset = 0
	}
	writeAccessResult(w, map[string]any{
		"items": items, "total": total, "limit": limit, "offset": offset,
		"hasMore": int64(offset+limit) < total,
	}, nil)
}

func writeAccessError(w http.ResponseWriter, status int, message string) {
	setJSONContentType(w)
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"message": message, "status": status}})
}

func methodNotAllowed(w http.ResponseWriter) {
	writeAccessError(w, http.StatusMethodNotAllowed, "method not allowed")
}

func setJSONContentType(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
}
