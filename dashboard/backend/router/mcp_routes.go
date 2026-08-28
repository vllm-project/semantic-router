package router

import (
	"context"
	"fmt"
	"log"
	"net"
	"net/http"
	"strings"

	auth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/mcp"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

const internalOpenClawMCPPath = "/_internal/openclaw/mcp"

// SetupMCP configures MCP related routes
// Returns MCP Manager instance for lifecycle management
func SetupMCP(routes *auth.PolicyMux, cfg *config.Config, wf *workflowstore.Store, openClawHandler *handlers.OpenClawHandler) *mcp.Manager {
	if !cfg.MCPEnabled {
		log.Printf("MCP feature disabled")
		return nil
	}

	mcpManager, err := mcp.NewManager(wf)
	if err != nil {
		log.Fatalf("MCP manager: %v", err)
	}

	// Register built-in OpenClaw MCP endpoint and server config.
	if cfg.OpenClawEnabled && openClawHandler != nil {
		registerBuiltInOpenClawMCP(routes, cfg.Port, mcpManager, openClawHandler)
	}

	// Create MCP handler
	mcpHandler := handlers.NewMCPHandler(mcpManager, cfg.ReadonlyMode)
	registerMCPAPIRoutes(routes, mcpHandler)

	log.Printf("MCP API endpoints registered: /api/mcp/*")

	// Auto-connect enabled servers in background
	go mcpManager.ConnectEnabled(context.Background())

	return mcpManager
}

func registerBuiltInOpenClawMCP(
	routes *auth.PolicyMux,
	port string,
	mcpManager *mcp.Manager,
	openClawHandler *handlers.OpenClawHandler,
) {
	openClawMCPHandler := handlers.NewOpenClawMCPHandler(openClawHandler)
	routes.Handle(
		auth.ProtectedMutationRoute("/api/openclaw/mcp", auth.PermMcpManage, "openclaw.mcp", auth.SensitivitySecret, auth.ResourceOwnerOpenClaw, 2<<20, http.MethodPost),
		openClawMCPHandler,
	)
	routes.HandleFallback(internalOpenClawMCPPath, loopbackOnly(openClawMCPHandler))

	serverURL := fmt.Sprintf("http://127.0.0.1:%s%s", port, internalOpenClawMCPPath)
	if err := mcpManager.UpsertServer(&mcp.ServerConfig{
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

	log.Printf(
		"Built-in OpenClaw MCP endpoints registered: /api/openclaw/mcp (authenticated), %s (loopback-only) (server id: %s)",
		internalOpenClawMCPPath,
		mcp.BuiltinOpenClawServerID,
	)
}

func registerMCPAPIRoutes(routes *auth.PolicyMux, mcpHandler *handlers.MCPHandler) {
	// Server configuration - GET list, POST create
	routes.HandleFunc(auth.Route(
		"/api/mcp/servers",
		auth.ReadPolicy(http.MethodGet, auth.PermMcpRead, auth.SensitivitySensitive, auth.ResourceOwnerTools),
		auth.MutationPolicy(http.MethodPost, auth.PermMcpManage, "mcp.server.create", auth.SensitivitySecret, auth.ResourceOwnerTools, 2<<20),
	), func(w http.ResponseWriter, r *http.Request) {
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

	registerMCPServerOperationRoutes(routes, mcpHandler)
	registerMCPToolRoutes(routes, mcpHandler)
}

func registerMCPServerOperationRoutes(routes *auth.PolicyMux, mcpHandler *handlers.MCPHandler) {
	// Server operations (update, delete, connect, disconnect, status, test)
	routes.HandleFunc(auth.Route(
		"/api/mcp/servers/",
		auth.ReadPolicy(http.MethodGet, auth.PermMcpRead, auth.SensitivitySecret, auth.ResourceOwnerTools),
		auth.MutationPolicy(http.MethodPost, auth.PermMcpManage, "mcp.server.action", auth.SensitivitySecret, auth.ResourceOwnerTools, 2<<20),
		auth.MutationPolicy(http.MethodPut, auth.PermMcpManage, "mcp.server.update", auth.SensitivitySecret, auth.ResourceOwnerTools, 2<<20),
		auth.MutationPolicy(http.MethodDelete, auth.PermMcpManage, "mcp.server.delete", auth.SensitivitySecret, auth.ResourceOwnerTools, auth.NoBodyLimit),
	), func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		path := r.URL.Path

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
}

func registerMCPToolRoutes(routes *auth.PolicyMux, mcpHandler *handlers.MCPHandler) {
	// Tools - GET list
	routes.HandleFunc(auth.ProtectedRoute("/api/mcp/tools", auth.PermMcpRead, auth.SensitivitySensitive, auth.ResourceOwnerTools, http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		mcpHandler.ListToolsHandler().ServeHTTP(w, r)
	})

	// Tool execution - POST execute
	routes.HandleFunc(auth.ProtectedMutationRoute("/api/mcp/tools/execute", auth.PermToolsUse, "mcp.tool.execute", auth.SensitivitySecret, auth.ResourceOwnerTools, 4<<20, http.MethodPost), func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		mcpHandler.ExecuteToolHandler().ServeHTTP(w, r)
	})

	// Tool streaming execution - POST execute/stream
	routes.HandleFunc(auth.ProtectedMutationRoute("/api/mcp/tools/execute/stream", auth.PermToolsUse, "mcp.tool.execute_stream", auth.SensitivitySecret, auth.ResourceOwnerTools, 4<<20, http.MethodPost), func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		mcpHandler.ExecuteToolStreamHandler().ServeHTTP(w, r)
	})
}

func loopbackOnly(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !isLoopbackRequest(r.RemoteAddr) {
			http.Error(w, "Forbidden", http.StatusForbidden)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func isLoopbackRequest(remoteAddr string) bool {
	host := strings.TrimSpace(remoteAddr)
	if host == "" {
		return false
	}

	parsedHost, _, err := net.SplitHostPort(remoteAddr)
	if err == nil {
		host = parsedHost
	}

	if strings.EqualFold(host, "localhost") {
		return true
	}

	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}
