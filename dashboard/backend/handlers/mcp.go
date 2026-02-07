package handlers

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/dashboard/backend/mcp"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
)

// MCPHandler handles MCP related HTTP requests
type MCPHandler struct {
	manager      *mcp.Manager
	readonlyMode bool
}

// NewMCPHandler creates an MCP Handler
func NewMCPHandler(manager *mcp.Manager, readonlyMode bool) *MCPHandler {
	return &MCPHandler{
		manager:      manager,
		readonlyMode: readonlyMode,
	}
}

// ========== Server Config Handlers ==========

// ListServersHandler GET /api/mcp/servers - Get all server configurations
func (h *MCPHandler) ListServersHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		states := h.manager.GetAllServerStates()

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"servers": states,
		})
	}
}

// CreateServerHandler POST /api/mcp/servers - Create server configuration
func (h *MCPHandler) CreateServerHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		if h.readonlyMode {
			http.Error(w, "Operation not allowed in readonly mode", http.StatusForbidden)
			return
		}

		var config mcp.ServerConfig
		if err := json.NewDecoder(r.Body).Decode(&config); err != nil {
			http.Error(w, "Invalid request body: "+err.Error(), http.StatusBadRequest)
			return
		}

		// Generate ID
		if config.ID == "" {
			config.ID = uuid.New().String()
		}

		// Validate required fields
		if config.Name == "" {
			http.Error(w, "Name is required", http.StatusBadRequest)
			return
		}

		if config.Transport == "" {
			http.Error(w, "Transport is required", http.StatusBadRequest)
			return
		}

		// Validate transport type
		if config.Transport != mcp.TransportStdio && config.Transport != mcp.TransportStreamableHTTP {
			http.Error(w, "Invalid transport type. Must be 'stdio' or 'streamable-http'", http.StatusBadRequest)
			return
		}

		// Validate connection configuration
		switch config.Transport {
		case mcp.TransportStdio:
			if config.Connection.Command == "" {
				http.Error(w, "Command is required for stdio transport", http.StatusBadRequest)
				return
			}
		case mcp.TransportStreamableHTTP:
			if config.Connection.URL == "" {
				http.Error(w, "URL is required for streamable-http transport", http.StatusBadRequest)
				return
			}
		}

		if err := h.manager.AddServer(&config); err != nil {
			http.Error(w, "Failed to add server: "+err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(config)
	}
}

// UpdateServerHandler PUT /api/mcp/servers/:id - Update server configuration
func (h *MCPHandler) UpdateServerHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodPut {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		if h.readonlyMode {
			http.Error(w, "Operation not allowed in readonly mode", http.StatusForbidden)
			return
		}

		// Extract ID from URL
		id := strings.TrimPrefix(r.URL.Path, "/api/mcp/servers/")
		if id == "" {
			http.Error(w, "Server ID is required", http.StatusBadRequest)
			return
		}

		var config mcp.ServerConfig
		if err := json.NewDecoder(r.Body).Decode(&config); err != nil {
			http.Error(w, "Invalid request body: "+err.Error(), http.StatusBadRequest)
			return
		}

		config.ID = id

		if err := h.manager.UpdateServer(&config); err != nil {
			http.Error(w, "Failed to update server: "+err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(config)
	}
}

// DeleteServerHandler DELETE /api/mcp/servers/:id - Delete server configuration
func (h *MCPHandler) DeleteServerHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodDelete {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		if h.readonlyMode {
			http.Error(w, "Operation not allowed in readonly mode", http.StatusForbidden)
			return
		}

		// Extract ID from URL
		id := strings.TrimPrefix(r.URL.Path, "/api/mcp/servers/")
		if id == "" {
			http.Error(w, "Server ID is required", http.StatusBadRequest)
			return
		}

		if err := h.manager.DeleteServer(id); err != nil {
			http.Error(w, "Failed to delete server: "+err.Error(), http.StatusInternalServerError)
			return
		}

		w.WriteHeader(http.StatusNoContent)
	}
}

// ========== Connection Handlers ==========

// ConnectServerHandler POST /api/mcp/servers/:id/connect - Connect to server
func (h *MCPHandler) ConnectServerHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Extract ID from URL
		path := strings.TrimPrefix(r.URL.Path, "/api/mcp/servers/")
		id := strings.TrimSuffix(path, "/connect")
		if id == "" {
			http.Error(w, "Server ID is required", http.StatusBadRequest)
			return
		}

		// Use independent context with timeout instead of HTTP request context
		// This prevents MCP server process from being cancelled when HTTP request ends
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()

		if err := h.manager.Connect(ctx, id); err != nil {
			http.Error(w, "Failed to connect: "+err.Error(), http.StatusInternalServerError)
			return
		}

		state, _ := h.manager.GetServerStatus(id)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(state)
	}
}

// DisconnectServerHandler POST /api/mcp/servers/:id/disconnect - Disconnect from server
func (h *MCPHandler) DisconnectServerHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Extract ID from URL
		path := strings.TrimPrefix(r.URL.Path, "/api/mcp/servers/")
		id := strings.TrimSuffix(path, "/disconnect")
		if id == "" {
			http.Error(w, "Server ID is required", http.StatusBadRequest)
			return
		}

		if err := h.manager.Disconnect(id); err != nil {
			http.Error(w, "Failed to disconnect: "+err.Error(), http.StatusInternalServerError)
			return
		}

		state, _ := h.manager.GetServerStatus(id)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(state)
	}
}

// GetServerStatusHandler GET /api/mcp/servers/:id/status - Get server status
func (h *MCPHandler) GetServerStatusHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Extract ID from URL
		path := strings.TrimPrefix(r.URL.Path, "/api/mcp/servers/")
		id := strings.TrimSuffix(path, "/status")
		if id == "" {
			http.Error(w, "Server ID is required", http.StatusBadRequest)
			return
		}

		state, err := h.manager.GetServerStatus(id)
		if err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(state)
	}
}

// TestConnectionHandler POST /api/mcp/servers/:id/test - Test connection
func (h *MCPHandler) TestConnectionHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var config mcp.ServerConfig
		if err := json.NewDecoder(r.Body).Decode(&config); err != nil {
			http.Error(w, "Invalid request body: "+err.Error(), http.StatusBadRequest)
			return
		}

		// Use independent context with timeout
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		if err := h.manager.TestConnection(ctx, &config); err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"success": false,
				"error":   err.Error(),
			})
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
		})
	}
}

// ========== Tool Handlers ==========

// ListToolsHandler GET /api/mcp/tools - Get all tools
func (h *MCPHandler) ListToolsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		tools := h.manager.GetAllTools()

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"tools": tools,
		})
	}
}

// ExecuteToolHandler POST /api/mcp/tools/execute - Execute tool
func (h *MCPHandler) ExecuteToolHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req mcp.ToolExecuteRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			log.Printf("[MCP-Handler] ExecuteTool: failed to decode request: %v", err)
			http.Error(w, "Invalid request body: "+err.Error(), http.StatusBadRequest)
			return
		}

		log.Printf("[MCP-Handler] ExecuteTool: server_id=%s, tool_name=%s, arguments=%s",
			req.ServerID, req.ToolName, string(req.Arguments))

		if req.ServerID == "" {
			http.Error(w, "server_id is required", http.StatusBadRequest)
			return
		}

		if req.ToolName == "" {
			http.Error(w, "tool_name is required", http.StatusBadRequest)
			return
		}

		result, err := h.manager.ExecuteTool(r.Context(), req.ServerID, req.ToolName, req.Arguments)
		if err != nil {
			log.Printf("[MCP-Handler] ExecuteTool: failed: %v", err)
			http.Error(w, "Failed to execute tool: "+err.Error(), http.StatusInternalServerError)
			return
		}

		log.Printf("[MCP-Handler] ExecuteTool: success")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(result)
	}
}

// ExecuteToolStreamHandler POST /api/mcp/tools/execute/stream - Stream execute tool
func (h *MCPHandler) ExecuteToolStreamHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req mcp.ToolExecuteRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body: "+err.Error(), http.StatusBadRequest)
			return
		}

		if req.ServerID == "" {
			http.Error(w, "server_id is required", http.StatusBadRequest)
			return
		}

		if req.ToolName == "" {
			http.Error(w, "tool_name is required", http.StatusBadRequest)
			return
		}

		// Set SSE headers
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming not supported", http.StatusInternalServerError)
			return
		}

		err := h.manager.ExecuteToolStreaming(r.Context(), req.ServerID, req.ToolName, req.Arguments, func(chunk mcp.StreamChunk) error {
			data, _ := json.Marshal(chunk)
			_, err := w.Write([]byte("event: message\n"))
			if err != nil {
				return err
			}
			_, err = w.Write([]byte("data: " + string(data) + "\n\n"))
			if err != nil {
				return err
			}
			flusher.Flush()
			return nil
		})
		if err != nil {
			// Send error event
			errData, _ := json.Marshal(map[string]string{"error": err.Error()})
			_, _ = w.Write([]byte("event: error\n"))
			_, _ = w.Write([]byte("data: " + string(errData) + "\n\n"))
			flusher.Flush()
		}
	}
}
