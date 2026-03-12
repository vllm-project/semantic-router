package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

// OpenClawMCPHandler exposes OpenClaw management operations as MCP tools.
// It reuses existing OpenClaw HTTP handlers to keep behavior consistent.
type OpenClawMCPHandler struct {
	openClaw *OpenClawHandler
	httpMCP  http.Handler
}

type clawCreateTeamArgs struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Vibe        string `json:"vibe"`
	Role        string `json:"role"`
	Principal   string `json:"principal"`
	Description string `json:"description"`
	LeaderID    string `json:"leader_id,omitempty"`
}

type clawUpdateTeamArgs struct {
	TeamID      string  `json:"team_id"`
	Name        *string `json:"name,omitempty"`
	Vibe        *string `json:"vibe,omitempty"`
	Role        *string `json:"role,omitempty"`
	Principal   *string `json:"principal,omitempty"`
	Description *string `json:"description,omitempty"`
	LeaderID    *string `json:"leader_id,omitempty"`
}

type clawDeleteTeamArgs struct {
	TeamID string `json:"team_id"`
}

type clawGetWorkerArgs struct {
	WorkerID string `json:"worker_id"`
}

type clawDeleteWorkerArgs struct {
	WorkerID string `json:"worker_id"`
}

type clawCreateWorkerArgs struct {
	TeamID     string   `json:"team_id"`
	Name       string   `json:"name"`
	Emoji      string   `json:"emoji"`
	Role       string   `json:"role"`
	Vibe       string   `json:"vibe"`
	Principles string   `json:"principles"`
	RoleKind   string   `json:"role_kind,omitempty"`
	Skills     []string `json:"skills,omitempty"`
}

type clawUpdateWorkerArgs struct {
	WorkerID   string  `json:"worker_id"`
	TeamID     *string `json:"team_id,omitempty"`
	Name       *string `json:"name,omitempty"`
	Emoji      *string `json:"emoji,omitempty"`
	Role       *string `json:"role,omitempty"`
	Vibe       *string `json:"vibe,omitempty"`
	Principles *string `json:"principles,omitempty"`
	RoleKind   *string `json:"role_kind,omitempty"`
}

func NewOpenClawMCPHandler(openClaw *OpenClawHandler) *OpenClawMCPHandler {
	mcpServer := server.NewMCPServer("openclaw-control-plane", "1.0.0")
	h := &OpenClawMCPHandler{openClaw: openClaw}
	h.registerTools(mcpServer)
	h.httpMCP = server.NewStreamableHTTPServer(mcpServer)
	return h
}

func (h *OpenClawMCPHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.httpMCP.ServeHTTP(w, r)
}

func (h *OpenClawMCPHandler) registerTools(mcpServer *server.MCPServer) {
	mcpServer.AddTool(clawListTeamsToolDefinition(), h.listTeamsTool)
	mcpServer.AddTool(clawCreateTeamToolDefinition(), h.createTeamTool)
	mcpServer.AddTool(clawUpdateTeamToolDefinition(), h.updateTeamTool)
	mcpServer.AddTool(clawDeleteTeamToolDefinition(), h.deleteTeamTool)
	mcpServer.AddTool(clawListWorkersToolDefinition(), h.listWorkersTool)
	mcpServer.AddTool(clawGetWorkerToolDefinition(), h.getWorkerTool)
	mcpServer.AddTool(clawCreateWorkerToolDefinition(), h.createWorkerTool)
	mcpServer.AddTool(clawUpdateWorkerToolDefinition(), h.updateWorkerTool)
	mcpServer.AddTool(clawDeleteWorkerToolDefinition(), h.deleteWorkerTool)
}

func (h *OpenClawMCPHandler) listTeamsTool(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	data, err := h.invokeOpenClawJSON(h.openClaw.TeamsHandler(), http.MethodGet, "/api/openclaw/teams", nil)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	return newJSONResult(data)
}

func (h *OpenClawMCPHandler) createTeamTool(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	var args clawCreateTeamArgs
	if err := request.BindArguments(&args); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("invalid arguments: %v", err)), nil
	}

	if strings.TrimSpace(args.Name) == "" {
		return mcp.NewToolResultError("name is required"), nil
	}

	payload := map[string]any{
		"id":          strings.TrimSpace(args.ID),
		"name":        strings.TrimSpace(args.Name),
		"vibe":        strings.TrimSpace(args.Vibe),
		"role":        strings.TrimSpace(args.Role),
		"principal":   strings.TrimSpace(args.Principal),
		"description": strings.TrimSpace(args.Description),
		"leaderId":    strings.TrimSpace(args.LeaderID),
	}

	data, err := h.invokeOpenClawJSON(h.openClaw.TeamsHandler(), http.MethodPost, "/api/openclaw/teams", payload)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	return newJSONResult(data)
}

func (h *OpenClawMCPHandler) updateTeamTool(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	var args clawUpdateTeamArgs
	if err := request.BindArguments(&args); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("invalid arguments: %v", err)), nil
	}

	teamID := strings.TrimSpace(args.TeamID)
	if teamID == "" {
		return mcp.NewToolResultError("team_id is required"), nil
	}

	teamPath := "/api/openclaw/teams/" + teamID
	existingAny, err := h.invokeOpenClawJSON(h.openClaw.TeamByIDHandler(), http.MethodGet, teamPath, nil)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	existing, ok := existingAny.(map[string]any)
	if !ok {
		return mcp.NewToolResultError("failed to decode existing team record"), nil
	}

	name := mergedString(args.Name, mapString(existing, "name"))
	if name == "" {
		return mcp.NewToolResultError("name cannot be empty"), nil
	}

	payload := map[string]any{
		"name":        name,
		"vibe":        mergedString(args.Vibe, mapString(existing, "vibe")),
		"role":        mergedString(args.Role, mapString(existing, "role")),
		"principal":   mergedString(args.Principal, mapString(existing, "principal")),
		"description": mergedString(args.Description, mapString(existing, "description")),
	}
	if args.LeaderID != nil {
		payload["leaderId"] = strings.TrimSpace(*args.LeaderID)
	}

	updated, err := h.invokeOpenClawJSON(h.openClaw.TeamByIDHandler(), http.MethodPut, teamPath, payload)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	return newJSONResult(updated)
}

func (h *OpenClawMCPHandler) deleteTeamTool(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	var args clawDeleteTeamArgs
	if err := request.BindArguments(&args); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("invalid arguments: %v", err)), nil
	}

	teamID := strings.TrimSpace(args.TeamID)
	if teamID == "" {
		return mcp.NewToolResultError("team_id is required"), nil
	}

	data, err := h.invokeOpenClawJSON(
		h.openClaw.TeamByIDHandler(),
		http.MethodDelete,
		"/api/openclaw/teams/"+teamID,
		nil,
	)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	return newJSONResult(data)
}

func (h *OpenClawMCPHandler) listWorkersTool(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	data, err := h.invokeOpenClawJSON(h.openClaw.WorkersHandler(), http.MethodGet, "/api/openclaw/workers", nil)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	return newJSONResult(data)
}

func (h *OpenClawMCPHandler) getWorkerTool(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	var args clawGetWorkerArgs
	if err := request.BindArguments(&args); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("invalid arguments: %v", err)), nil
	}

	workerID := strings.TrimSpace(args.WorkerID)
	if workerID == "" {
		return mcp.NewToolResultError("worker_id is required"), nil
	}

	data, err := h.invokeOpenClawJSON(
		h.openClaw.WorkerByIDHandler(),
		http.MethodGet,
		"/api/openclaw/workers/"+workerID,
		nil,
	)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	return newJSONResult(data)
}

func (h *OpenClawMCPHandler) createWorkerTool(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	var args clawCreateWorkerArgs
	if err := request.BindArguments(&args); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("invalid arguments: %v", err)), nil
	}

	teamID := strings.TrimSpace(args.TeamID)
	name := strings.TrimSpace(args.Name)
	emoji := strings.TrimSpace(args.Emoji)
	role := strings.TrimSpace(args.Role)
	vibe := strings.TrimSpace(args.Vibe)
	principles := strings.TrimSpace(args.Principles)
	if teamID == "" {
		return mcp.NewToolResultError("team_id is required"), nil
	}
	if name == "" {
		return mcp.NewToolResultError("name is required"), nil
	}
	if emoji == "" {
		return mcp.NewToolResultError("emoji is required"), nil
	}
	if role == "" {
		return mcp.NewToolResultError("role is required"), nil
	}
	if vibe == "" {
		return mcp.NewToolResultError("vibe is required"), nil
	}
	if principles == "" {
		return mcp.NewToolResultError("principles is required"), nil
	}

	identity := map[string]any{
		"name":       name,
		"emoji":      emoji,
		"role":       role,
		"vibe":       vibe,
		"principles": principles,
	}

	payload := map[string]any{
		"teamId":   teamID,
		"skills":   args.Skills,
		"identity": identity,
		"roleKind": normalizeRoleKind(args.RoleKind),
	}

	data, err := h.invokeOpenClawJSON(h.openClaw.WorkersHandler(), http.MethodPost, "/api/openclaw/workers?async=true", payload)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	return newJSONResult(data)
}

func (h *OpenClawMCPHandler) updateWorkerTool(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	var args clawUpdateWorkerArgs
	if err := request.BindArguments(&args); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("invalid arguments: %v", err)), nil
	}

	workerID := strings.TrimSpace(args.WorkerID)
	if workerID == "" {
		return mcp.NewToolResultError("worker_id is required"), nil
	}

	workerPath := "/api/openclaw/workers/" + workerID
	existingAny, err := h.invokeOpenClawJSON(h.openClaw.WorkerByIDHandler(), http.MethodGet, workerPath, nil)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	existing, ok := existingAny.(map[string]any)
	if !ok {
		return mcp.NewToolResultError("failed to decode existing worker record"), nil
	}

	payload := map[string]any{}
	if args.TeamID != nil {
		payload["teamId"] = strings.TrimSpace(*args.TeamID)
	}

	identityUpdated := args.Name != nil || args.Emoji != nil || args.Role != nil || args.Vibe != nil || args.Principles != nil
	if identityUpdated {
		payload["identity"] = map[string]any{
			"name":       mergedString(args.Name, mapString(existing, "agentName")),
			"emoji":      mergedString(args.Emoji, mapString(existing, "agentEmoji")),
			"role":       mergedString(args.Role, mapString(existing, "agentRole")),
			"vibe":       mergedString(args.Vibe, mapString(existing, "agentVibe")),
			"principles": mergedString(args.Principles, mapString(existing, "agentPrinciples")),
		}
	}
	if args.RoleKind != nil {
		payload["roleKind"] = normalizeRoleKind(*args.RoleKind)
	}

	if len(payload) == 0 {
		return mcp.NewToolResultError("provide at least one updatable field"), nil
	}

	updated, err := h.invokeOpenClawJSON(h.openClaw.WorkerByIDHandler(), http.MethodPut, workerPath, payload)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	return newJSONResult(updated)
}

func (h *OpenClawMCPHandler) deleteWorkerTool(_ context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	var args clawDeleteWorkerArgs
	if err := request.BindArguments(&args); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("invalid arguments: %v", err)), nil
	}

	workerID := strings.TrimSpace(args.WorkerID)
	if workerID == "" {
		return mcp.NewToolResultError("worker_id is required"), nil
	}

	data, err := h.invokeOpenClawJSON(
		h.openClaw.WorkerByIDHandler(),
		http.MethodDelete,
		"/api/openclaw/workers/"+workerID,
		nil,
	)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	return newJSONResult(data)
}

func (h *OpenClawMCPHandler) invokeOpenClawJSON(handler http.HandlerFunc, method, path string, payload any) (any, error) {
	var body io.Reader = http.NoBody
	if payload != nil {
		raw, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal payload: %w", err)
		}
		body = bytes.NewReader(raw)
	}

	req := httptest.NewRequest(method, path, body)
	req.Header.Set("Content-Type", "application/json")

	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code < http.StatusOK || rr.Code >= http.StatusMultipleChoices {
		return nil, fmt.Errorf("%s %s failed (%d): %s", method, path, rr.Code, normalizeAPIError(rr.Body.Bytes()))
	}

	if rr.Body.Len() == 0 {
		return map[string]any{"success": true}, nil
	}

	var data any
	if err := json.Unmarshal(rr.Body.Bytes(), &data); err != nil {
		return strings.TrimSpace(rr.Body.String()), nil
	}
	return data, nil
}

func normalizeAPIError(raw []byte) string {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" {
		return "unknown error"
	}

	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err == nil {
		if errMessage, ok := payload["error"].(string); ok && strings.TrimSpace(errMessage) != "" {
			return strings.TrimSpace(errMessage)
		}
	}
	return trimmed
}

func mapString(payload map[string]any, key string) string {
	value, ok := payload[key]
	if !ok {
		return ""
	}
	text, ok := value.(string)
	if !ok {
		return ""
	}
	return strings.TrimSpace(text)
}

func mergedString(next *string, current string) string {
	if next == nil {
		return strings.TrimSpace(current)
	}
	return strings.TrimSpace(*next)
}

func newJSONResult(data any) (*mcp.CallToolResult, error) {
	result, err := mcp.NewToolResultJSON(data)
	if err == nil {
		return result, nil
	}
	return mcp.NewToolResultText(fmt.Sprintf("%v", data)), nil
}
