package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/dashboard/backend/mcp"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
)

// MCPHandler handles MCP API requests.
type MCPHandler struct {
	manager      *mcp.Manager
	readonlyMode bool
}

// NewMCPHandler creates an MCP Handler.
func NewMCPHandler(manager *mcp.Manager, readonlyMode bool) *MCPHandler {
	return &MCPHandler{manager: manager, readonlyMode: readonlyMode}
}

// ListServersHandler handles GET /api/mcp/servers.
func (h *MCPHandler) ListServersHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if handledMCPPreflightOrMethod(w, r, http.MethodGet) {
			return
		}

		canManageOpenClaw := mayUseMCPBuiltin(r)
		states := h.manager.GetAllServerStates()
		views := make([]mcpServerStateView, 0, len(states))
		for _, state := range states {
			exposeTools := state.Config == nil || state.Config.ID != mcp.BuiltinOpenClawServerID || canManageOpenClaw
			views = append(views, newMCPServerStateView(state, exposeTools))
		}
		writeMCPJSON(w, http.StatusOK, map[string]any{"servers": views})
	}
}

// CreateServerHandler handles POST /api/mcp/servers.
func (h *MCPHandler) CreateServerHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if handledMCPPreflightOrMethod(w, r, http.MethodPost) || h.rejectReadonly(w) {
			return
		}

		var config mcp.ServerConfig
		if status, err := decodeMCPJSON(w, r, mcpConfigBodyLimit, &config); err != nil {
			http.Error(w, err.Error(), status)
			return
		}
		if config.ID == "" {
			config.ID = uuid.NewString()
		}
		if err := validateMCPServerID(config.ID); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if config.ID == mcp.BuiltinOpenClawServerID {
			http.Error(w, "reserved MCP server id", http.StatusConflict)
			return
		}
		if err := validateMCPServerConfig(&config); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if status, err := authorizeStdioMCP(r, h.manager, &config); err != nil {
			http.Error(w, err.Error(), status)
			return
		}
		if err := h.manager.AddServer(&config); err != nil {
			status, message := managerErrorStatus(err)
			http.Error(w, message, status)
			return
		}
		writeMCPJSON(w, http.StatusCreated, newMCPServerConfigView(&config))
	}
}

// UpdateServerHandler handles PUT /api/mcp/servers/:id.
func (h *MCPHandler) UpdateServerHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if handledMCPPreflightOrMethod(w, r, http.MethodPut) || h.rejectReadonly(w) {
			return
		}
		id := mcpServerOperationID(r.URL.Path, "")
		if id == "" {
			http.Error(w, "server id is required", http.StatusBadRequest)
			return
		}
		if id == mcp.BuiltinOpenClawServerID {
			if status, err := authorizeMCPBuiltin(r); err != nil {
				http.Error(w, err.Error(), status)
				return
			}
			http.Error(w, "built-in MCP server is managed internally", http.StatusForbidden)
			return
		}
		existing, ok := h.manager.GetServer(id)
		if !ok {
			http.Error(w, "MCP server not found", http.StatusNotFound)
			return
		}

		var patch mcpServerUpdateRequest
		if status, err := decodeMCPJSON(w, r, mcpConfigBodyLimit, &patch); err != nil {
			http.Error(w, err.Error(), status)
			return
		}
		updated, err := applyMCPServerUpdate(existing, &patch)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if err := validateMCPServerConfig(updated); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if status, err := authorizeStdioMCP(r, h.manager, updated); err != nil {
			http.Error(w, err.Error(), status)
			return
		}
		if err := h.manager.UpdateServer(updated); err != nil {
			status, message := managerErrorStatus(err)
			http.Error(w, message, status)
			return
		}
		writeMCPJSON(w, http.StatusOK, newMCPServerConfigView(updated))
	}
}

// DeleteServerHandler handles DELETE /api/mcp/servers/:id.
func (h *MCPHandler) DeleteServerHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if handledMCPPreflightOrMethod(w, r, http.MethodDelete) || h.rejectReadonly(w) {
			return
		}
		id := mcpServerOperationID(r.URL.Path, "")
		if id == "" {
			http.Error(w, "server id is required", http.StatusBadRequest)
			return
		}
		if id == mcp.BuiltinOpenClawServerID {
			if status, err := authorizeMCPBuiltin(r); err != nil {
				http.Error(w, err.Error(), status)
				return
			}
			http.Error(w, "built-in MCP server is managed internally", http.StatusForbidden)
			return
		}
		if err := h.manager.DeleteServer(id); err != nil {
			status, message := managerErrorStatus(err)
			http.Error(w, message, status)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}
}

// ConnectServerHandler handles POST /api/mcp/servers/:id/connect.
func (h *MCPHandler) ConnectServerHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if handledMCPPreflightOrMethod(w, r, http.MethodPost) || h.rejectReadonly(w) {
			return
		}
		id := mcpServerOperationID(r.URL.Path, "/connect")
		if !h.authorizeServerOperation(w, r, id) {
			return
		}
		config, ok := h.manager.GetServer(id)
		if !ok {
			http.Error(w, "MCP server not found", http.StatusNotFound)
			return
		}
		if status, err := authorizeStdioMCP(r, h.manager, config); err != nil {
			http.Error(w, err.Error(), status)
			return
		}

		ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
		defer cancel()
		if err := h.manager.Connect(ctx, id); err != nil {
			status, message := managerErrorStatus(err)
			http.Error(w, message, status)
			return
		}
		state, err := h.manager.GetServerStatus(id)
		if err != nil {
			status, message := managerErrorStatus(err)
			http.Error(w, message, status)
			return
		}
		writeMCPJSON(w, http.StatusOK, newMCPServerStateView(state, true))
	}
}

// DisconnectServerHandler handles POST /api/mcp/servers/:id/disconnect.
func (h *MCPHandler) DisconnectServerHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if handledMCPPreflightOrMethod(w, r, http.MethodPost) || h.rejectReadonly(w) {
			return
		}
		id := mcpServerOperationID(r.URL.Path, "/disconnect")
		if !h.authorizeServerOperation(w, r, id) {
			return
		}
		if err := h.manager.Disconnect(id); err != nil {
			http.Error(w, "MCP operation failed", http.StatusInternalServerError)
			return
		}
		state, err := h.manager.GetServerStatus(id)
		if err != nil {
			status, message := managerErrorStatus(err)
			http.Error(w, message, status)
			return
		}
		writeMCPJSON(w, http.StatusOK, newMCPServerStateView(state, true))
	}
}

// GetServerStatusHandler handles GET /api/mcp/servers/:id/status.
func (h *MCPHandler) GetServerStatusHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if handledMCPPreflightOrMethod(w, r, http.MethodGet) {
			return
		}
		id := mcpServerOperationID(r.URL.Path, "/status")
		if !h.authorizeServerOperation(w, r, id) {
			return
		}
		state, err := h.manager.GetServerStatus(id)
		if err != nil {
			status, message := managerErrorStatus(err)
			http.Error(w, message, status)
			return
		}
		writeMCPJSON(w, http.StatusOK, newMCPServerStateView(state, true))
	}
}

// TestConnectionHandler handles POST /api/mcp/servers/:id/test.
func (h *MCPHandler) TestConnectionHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if handledMCPPreflightOrMethod(w, r, http.MethodPost) || h.rejectReadonly(w) {
			return
		}
		id := mcpServerOperationID(r.URL.Path, "/test")
		if !h.authorizeServerOperation(w, r, id) {
			return
		}
		var request mcpServerUpdateRequest
		if status, err := decodeMCPJSON(w, r, mcpConfigBodyLimit, &request); err != nil {
			http.Error(w, err.Error(), status)
			return
		}
		if request.ID != nil && *request.ID != "" && *request.ID != id {
			http.Error(w, "server id does not match request path", http.StatusBadRequest)
			return
		}
		request.ID = &id
		config, err := resolveMCPServerTestConfig(h.manager, &request)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if err := validateMCPServerConfig(config); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if status, err := authorizeStdioMCP(r, h.manager, config); err != nil {
			http.Error(w, err.Error(), status)
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
		defer cancel()
		if err := h.manager.TestConnection(ctx, config); err != nil {
			writeMCPJSON(w, http.StatusOK, map[string]any{"success": false, "error": "connection failed"})
			return
		}
		writeMCPJSON(w, http.StatusOK, map[string]any{"success": true})
	}
}

// ListToolsHandler handles GET /api/mcp/tools.
func (h *MCPHandler) ListToolsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if handledMCPPreflightOrMethod(w, r, http.MethodGet) {
			return
		}
		canManageOpenClaw := mayUseMCPBuiltin(r)
		filtered := filterMCPTools(h.manager.GetAllTools(), canManageOpenClaw)
		writeMCPJSON(w, http.StatusOK, map[string]any{"tools": filtered})
	}
}

// ExecuteToolHandler handles POST /api/mcp/tools/execute.
func (h *MCPHandler) ExecuteToolHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if handledMCPPreflightOrMethod(w, r, http.MethodPost) {
			return
		}
		var request mcp.ToolExecuteRequest
		if status, err := decodeMCPJSON(w, r, mcpToolBodyLimit, &request); err != nil {
			http.Error(w, err.Error(), status)
			return
		}
		if !validateMCPToolRequest(w, &request) || !authorizeMCPToolRequest(w, r, &request) {
			return
		}
		result, err := h.manager.ExecuteTool(r.Context(), request.ServerID, request.ToolName, request.Arguments)
		if err != nil {
			http.Error(w, "tool execution failed", http.StatusBadGateway)
			return
		}
		writeMCPJSON(w, http.StatusOK, result)
	}
}

// ExecuteToolStreamHandler handles POST /api/mcp/tools/execute/stream.
func (h *MCPHandler) ExecuteToolStreamHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if handledMCPPreflightOrMethod(w, r, http.MethodPost) {
			return
		}
		var request mcp.ToolExecuteRequest
		if status, err := decodeMCPJSON(w, r, mcpToolBodyLimit, &request); err != nil {
			http.Error(w, err.Error(), status)
			return
		}
		if !validateMCPToolRequest(w, &request) || !authorizeMCPToolRequest(w, r, &request) {
			return
		}
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming not supported", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache, no-store")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Content-Type-Options", "nosniff")

		err := h.manager.ExecuteToolStreaming(
			r.Context(),
			request.ServerID,
			request.ToolName,
			request.Arguments,
			func(chunk mcp.StreamChunk) error {
				data, err := json.Marshal(chunk)
				if err != nil {
					return err
				}
				if _, err := w.Write([]byte("event: message\ndata: ")); err != nil {
					return err
				}
				if _, err := w.Write(data); err != nil {
					return err
				}
				if _, err := w.Write([]byte("\n\n")); err != nil {
					return err
				}
				flusher.Flush()
				return nil
			},
		)
		if err != nil {
			data, _ := json.Marshal(map[string]string{"error": "tool execution failed"})
			_, _ = w.Write([]byte("event: error\ndata: "))
			_, _ = w.Write(data)
			_, _ = w.Write([]byte("\n\n"))
			flusher.Flush()
		}
	}
}

func (h *MCPHandler) rejectReadonly(w http.ResponseWriter) bool {
	if !h.readonlyMode {
		return false
	}
	http.Error(w, "operation not allowed in readonly mode", http.StatusForbidden)
	return true
}

func (h *MCPHandler) authorizeServerOperation(w http.ResponseWriter, r *http.Request, id string) bool {
	if id == "" {
		http.Error(w, "server id is required", http.StatusBadRequest)
		return false
	}
	if id != mcp.BuiltinOpenClawServerID {
		return true
	}
	status, err := authorizeMCPBuiltin(r)
	if err != nil {
		http.Error(w, err.Error(), status)
		return false
	}
	return true
}

func handledMCPPreflightOrMethod(w http.ResponseWriter, r *http.Request, method string) bool {
	if middleware.HandleCORSPreflight(w, r) {
		return true
	}
	if r.Method != method {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return true
	}
	return false
}

func mcpServerOperationID(path, suffix string) string {
	path = strings.TrimPrefix(path, "/api/mcp/servers/")
	path = strings.TrimSuffix(path, suffix)
	id := strings.TrimSpace(path)
	if validateMCPServerID(id) != nil {
		return ""
	}
	return id
}

func validateMCPToolRequest(w http.ResponseWriter, request *mcp.ToolExecuteRequest) bool {
	if strings.TrimSpace(request.ServerID) == "" {
		http.Error(w, "server_id is required", http.StatusBadRequest)
		return false
	}
	if strings.TrimSpace(request.ToolName) == "" {
		http.Error(w, "tool_name is required", http.StatusBadRequest)
		return false
	}
	if len(request.Arguments) == 0 {
		request.Arguments = json.RawMessage(`{}`)
	}
	var arguments map[string]any
	if err := json.Unmarshal(request.Arguments, &arguments); err != nil || arguments == nil {
		http.Error(w, "arguments must be a JSON object", http.StatusBadRequest)
		return false
	}
	return true
}

func authorizeMCPToolRequest(w http.ResponseWriter, r *http.Request, request *mcp.ToolExecuteRequest) bool {
	if request.ServerID != mcp.BuiltinOpenClawServerID {
		return true
	}
	status, err := authorizeMCPBuiltin(r)
	if err != nil {
		http.Error(w, err.Error(), status)
		return false
	}
	return true
}

func filterMCPTools(tools []mcp.Tool, canManageOpenClaw bool) []mcp.Tool {
	filtered := make([]mcp.Tool, 0, len(tools))
	for _, tool := range tools {
		if tool.ServerID == mcp.BuiltinOpenClawServerID && !canManageOpenClaw {
			continue
		}
		filtered = append(filtered, tool)
	}
	return filtered
}

func writeMCPJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
