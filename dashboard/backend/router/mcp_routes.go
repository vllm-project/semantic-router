package router

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/base64"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/mcp"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

const (
	internalOpenClawMCPPath             = "/_internal/openclaw/mcp"
	internalOpenClawMCPCapabilityHeader = "X-VSR-Internal-MCP-Capability"
	internalOpenClawMCPCapabilityBytes  = 32
)

// SetupMCP configures MCP related routes
// Returns MCP Manager instance for lifecycle management
func SetupMCP(mux *http.ServeMux, cfg *config.Config, wf *workflowstore.Store, openClawHandler *handlers.OpenClawHandler) (*mcp.Manager, error) {
	if !cfg.MCPEnabled {
		log.Printf("MCP feature disabled")
		return nil, nil
	}

	mcpManager, err := mcp.NewManagerWithOptions(wf, mcp.ManagerOptions{
		AllowStdio: cfg.SecurityProfile == config.DashboardSecurityProfileDevelopment,
	})
	if err != nil {
		return nil, fmt.Errorf("create MCP manager: %w", err)
	}

	// Register built-in OpenClaw MCP endpoint and server config.
	if cfg.OpenClawEnabled && openClawHandler != nil {
		if err := registerBuiltInOpenClawMCP(mux, cfg.Port, mcpManager, openClawHandler, rand.Reader); err != nil {
			mcpManager.Close()
			return nil, fmt.Errorf("register built-in OpenClaw MCP: %w", err)
		}
	}

	// Create MCP handler
	mcpHandler := handlers.NewMCPHandler(mcpManager, cfg.ReadonlyMode)
	registerMCPAPIRoutes(mux, mcpHandler)

	log.Printf("MCP API endpoints registered: /api/mcp/*")

	// Auto-connect enabled servers in background
	go mcpManager.ConnectEnabled(context.Background())

	return mcpManager, nil
}

func registerBuiltInOpenClawMCP(
	mux *http.ServeMux,
	port string,
	mcpManager *mcp.Manager,
	openClawHandler *handlers.OpenClawHandler,
	random io.Reader,
) error {
	capability, err := generateInternalOpenClawMCPCapability(random)
	if err != nil {
		return err
	}
	openClawMCPHandler := handlers.NewOpenClawMCPHandler(openClawHandler)
	mux.Handle("/api/openclaw/mcp", openClawMCPHandler)
	mux.Handle(internalOpenClawMCPPath, internalOpenClawMCPOnly(openClawMCPHandler, capability))

	serverURL := fmt.Sprintf("http://127.0.0.1:%s%s", port, internalOpenClawMCPPath)
	if err := mcpManager.UpsertEphemeralServer(&mcp.ServerConfig{
		ID:          mcp.BuiltinOpenClawServerID,
		Name:        mcp.BuiltinOpenClawServerName,
		Description: "Built-in MCP server for OpenClaw team, worker, and connection management",
		Transport:   mcp.TransportStreamableHTTP,
		Connection: mcp.ConnectionConfig{
			URL: serverURL,
			Headers: map[string]string{
				internalOpenClawMCPCapabilityHeader: capability,
			},
		},
		Enabled: false,
		Security: &mcp.SecurityConfig{
			LocalOnly: true,
		},
		Options: &mcp.ServerOptions{
			Timeout: 30000,
		},
	}); err != nil {
		return err
	}

	log.Printf(
		"Built-in OpenClaw MCP endpoints registered: /api/openclaw/mcp (public), %s (loopback-only) (server id: %s)",
		internalOpenClawMCPPath,
		mcp.BuiltinOpenClawServerID,
	)
	return nil
}

func generateInternalOpenClawMCPCapability(random io.Reader) (string, error) {
	if random == nil {
		return "", fmt.Errorf("secure random source is unavailable")
	}
	buffer := make([]byte, internalOpenClawMCPCapabilityBytes)
	if _, err := io.ReadFull(random, buffer); err != nil {
		return "", fmt.Errorf("read secure random capability: %w", err)
	}
	return base64.RawURLEncoding.EncodeToString(buffer), nil
}

func registerMCPAPIRoutes(mux *http.ServeMux, mcpHandler *handlers.MCPHandler) {
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

	registerMCPServerOperationRoutes(mux, mcpHandler)
	registerMCPToolRoutes(mux, mcpHandler)
}

func registerMCPServerOperationRoutes(mux *http.ServeMux, mcpHandler *handlers.MCPHandler) {
	// Server operations (update, delete, connect, disconnect, status, test)
	mux.HandleFunc("/api/mcp/servers/", func(w http.ResponseWriter, r *http.Request) {
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

func registerMCPToolRoutes(mux *http.ServeMux, mcpHandler *handlers.MCPHandler) {
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

func internalOpenClawMCPOnly(next http.Handler, capability string) http.Handler {
	expected := []byte(capability)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		provided := r.Header.Values(internalOpenClawMCPCapabilityHeader)
		if !isLoopbackRequest(r.RemoteAddr) || len(provided) != 1 ||
			len(provided[0]) != len(expected) ||
			subtle.ConstantTimeCompare([]byte(provided[0]), expected) != 1 {
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
