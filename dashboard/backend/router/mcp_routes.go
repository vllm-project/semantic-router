package router

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"strings"

	backendapp "github.com/vllm-project/semantic-router/dashboard/backend/app"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/mcp"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
)

// SetupMCP configures MCP related routes
// Returns MCP Manager instance for lifecycle management
func SetupMCP(mux *http.ServeMux, app *backendapp.App, openClawHandler *handlers.OpenClawHandler) *mcp.Manager {
	cfg := app.Config
	access := newRouteAccess(app)
	if !cfg.MCPEnabled {
		log.Printf("MCP feature disabled")
		return nil
	}

	// Initialize MCP manager (in-memory only, no config persistence)
	mcpManager := mcp.NewManager()
	registerBuiltinOpenClawMCP(mux, cfg.Port, access, mcpManager, cfg.OpenClawEnabled, openClawHandler)

	// Create MCP handler
	mcpHandler := handlers.NewMCPHandler(mcpManager, cfg.ReadonlyMode)
	registerMCPServerRoutes(mux, access, mcpHandler)
	registerMCPToolRoutes(mux, access, mcpHandler)

	log.Printf("MCP API endpoints registered: /api/mcp/*")

	// Auto-connect enabled servers in background
	go mcpManager.ConnectEnabled(context.Background())

	return mcpManager
}

func registerBuiltinOpenClawMCP(
	mux *http.ServeMux,
	port string,
	access routeAccess,
	mcpManager *mcp.Manager,
	enabled bool,
	openClawHandler *handlers.OpenClawHandler,
) {
	if !enabled || openClawHandler == nil {
		return
	}

	mux.Handle("/api/openclaw/mcp", access.admin(handlers.NewOpenClawMCPHandler(openClawHandler)))

	serverURL := fmt.Sprintf("http://127.0.0.1:%s/api/openclaw/mcp", port)
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
		return
	}

	log.Printf("Built-in OpenClaw MCP endpoint registered: /api/openclaw/mcp (server id: %s)", mcp.BuiltinOpenClawServerID)
}

func registerMCPServerRoutes(mux *http.ServeMux, access routeAccess, mcpHandler *handlers.MCPHandler) {
	mux.HandleFunc("/api/mcp/servers", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		switch r.Method {
		case http.MethodGet:
			access.admin(mcpHandler.ListServersHandler()).ServeHTTP(w, r)
		case http.MethodPost:
			access.admin(mcpHandler.CreateServerHandler()).ServeHTTP(w, r)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	})

	mux.HandleFunc("/api/mcp/servers/", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		serveMCPServerRoute(w, r, access, mcpHandler)
	})
}

func serveMCPServerRoute(w http.ResponseWriter, r *http.Request, access routeAccess, mcpHandler *handlers.MCPHandler) {
	switch {
	case strings.HasSuffix(r.URL.Path, "/connect"):
		access.admin(mcpHandler.ConnectServerHandler()).ServeHTTP(w, r)
	case strings.HasSuffix(r.URL.Path, "/disconnect"):
		access.admin(mcpHandler.DisconnectServerHandler()).ServeHTTP(w, r)
	case strings.HasSuffix(r.URL.Path, "/status"):
		access.admin(mcpHandler.GetServerStatusHandler()).ServeHTTP(w, r)
	case strings.HasSuffix(r.URL.Path, "/test"):
		access.admin(mcpHandler.TestConnectionHandler()).ServeHTTP(w, r)
	default:
		switch r.Method {
		case http.MethodPut:
			access.admin(mcpHandler.UpdateServerHandler()).ServeHTTP(w, r)
		case http.MethodDelete:
			access.admin(mcpHandler.DeleteServerHandler()).ServeHTTP(w, r)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

func registerMCPToolRoutes(mux *http.ServeMux, access routeAccess, mcpHandler *handlers.MCPHandler) {
	mux.HandleFunc("/api/mcp/tools", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		access.operator(mcpHandler.ListToolsHandler()).ServeHTTP(w, r)
	})

	mux.HandleFunc("/api/mcp/tools/execute", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		access.operator(mcpHandler.ExecuteToolHandler()).ServeHTTP(w, r)
	})

	mux.HandleFunc("/api/mcp/tools/execute/stream", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		access.operator(mcpHandler.ExecuteToolStreamHandler()).ServeHTTP(w, r)
	})
}
