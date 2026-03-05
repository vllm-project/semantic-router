package router

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/mcp"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
)

// SetupMCP configures MCP related routes
// Returns MCP Manager instance for lifecycle management
func SetupMCP(mux *http.ServeMux, cfg *config.Config, openClawHandler *handlers.OpenClawHandler) *mcp.Manager {
	if !cfg.MCPEnabled {
		log.Printf("MCP feature disabled")
		return nil
	}

	// Initialize MCP manager (in-memory only, no config persistence)
	mcpManager := mcp.NewManager()

	// Register built-in OpenClaw MCP endpoint and server config.
	if cfg.OpenClawEnabled && openClawHandler != nil {
		mux.Handle("/api/openclaw/mcp", handlers.NewOpenClawMCPHandler(openClawHandler))

		serverURL := fmt.Sprintf("http://127.0.0.1:%s/api/openclaw/mcp", cfg.Port)
		if err := mcpManager.AddServer(&mcp.ServerConfig{
			ID:          mcp.BuiltinOpenClawServerID,
			Name:        mcp.BuiltinOpenClawServerName,
			Description: "Built-in MCP server for OpenClaw team, worker, and connection management",
			Transport:   mcp.TransportStreamableHTTP,
			Connection: mcp.ConnectionConfig{
				URL: serverURL,
			},
			Enabled: false,
			Options: &mcp.ServerOptions{
				Timeout: 30000,
			},
		}); err != nil {
			log.Printf("Failed to register built-in OpenClaw MCP server: %v", err)
		} else {
			log.Printf("Built-in OpenClaw MCP endpoint registered: /api/openclaw/mcp (server id: %s)", mcp.BuiltinOpenClawServerID)
		}
	}

	// Create MCP handler
	mcpHandler := handlers.NewMCPHandler(mcpManager, cfg.ReadonlyMode)

	// Register MCP endpoints
	// Server configuration - GET list, POST create
	mux.HandleFunc("/api/mcp/servers", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		switch r.Method {
		case http.MethodGet:
			mcpHandler.ListServersHandler().ServeHTTP(w, r)
		case http.MethodPost:
			mcpHandler.CreateServerHandler().ServeHTTP(w, r)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	})

	// Server operations (update, delete, connect, disconnect, status, test)
	mux.HandleFunc("/api/mcp/servers/", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		path := r.URL.Path

		// Handle specific operations
		switch {
		case strings.HasSuffix(path, "/connect"):
			mcpHandler.ConnectServerHandler().ServeHTTP(w, r)
		case strings.HasSuffix(path, "/disconnect"):
			mcpHandler.DisconnectServerHandler().ServeHTTP(w, r)
		case strings.HasSuffix(path, "/status"):
			mcpHandler.GetServerStatusHandler().ServeHTTP(w, r)
		case strings.HasSuffix(path, "/test"):
			mcpHandler.TestConnectionHandler().ServeHTTP(w, r)
		default:
			// CRUD operations on server
			switch r.Method {
			case http.MethodPut:
				mcpHandler.UpdateServerHandler().ServeHTTP(w, r)
			case http.MethodDelete:
				mcpHandler.DeleteServerHandler().ServeHTTP(w, r)
			default:
				http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			}
		}
	})

	// Tools - GET list
	mux.HandleFunc("/api/mcp/tools", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		mcpHandler.ListToolsHandler().ServeHTTP(w, r)
	})

	// Tool execution - POST execute
	mux.HandleFunc("/api/mcp/tools/execute", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		mcpHandler.ExecuteToolHandler().ServeHTTP(w, r)
	})

	// Tool streaming execution - POST execute/stream
	mux.HandleFunc("/api/mcp/tools/execute/stream", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		mcpHandler.ExecuteToolStreamHandler().ServeHTTP(w, r)
	})

	log.Printf("MCP API endpoints registered: /api/mcp/*")

	// Auto-connect enabled servers in background
	go mcpManager.ConnectEnabled(context.Background())

	return mcpManager
}
