package router

import (
	"bytes"
	"context"
	"encoding/json"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/mcp"
)

type loginResponse struct {
	Token string `json:"token"`
}

type mcpServersResponse struct {
	Servers []struct {
		Config struct {
			ID         string `json:"id"`
			Connection struct {
				URL string `json:"url"`
			} `json:"connection"`
		} `json:"config"`
	} `json:"servers"`
}

type mcpToolsResponse struct {
	Tools []struct {
		Name string `json:"name"`
	} `json:"tools"`
}

func TestLoopbackOnly(t *testing.T) {
	t.Parallel()

	handler := loopbackOnly(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))

	t.Run("allows loopback", func(t *testing.T) {
		t.Parallel()

		req := httptestRequest(http.MethodGet, internalOpenClawMCPPath, "127.0.0.1:3000")
		recorder := httptestRecorder()

		handler.ServeHTTP(recorder, req)

		if recorder.Code != http.StatusNoContent {
			t.Fatalf("status = %d, want %d", recorder.Code, http.StatusNoContent)
		}
	})

	t.Run("blocks non-loopback", func(t *testing.T) {
		t.Parallel()

		req := httptestRequest(http.MethodGet, internalOpenClawMCPPath, "203.0.113.10:3000")
		recorder := httptestRecorder()

		handler.ServeHTTP(recorder, req)

		if recorder.Code != http.StatusForbidden {
			t.Fatalf("status = %d, want %d", recorder.Code, http.StatusForbidden)
		}
	})
}

func TestBuiltInOpenClawMCPConnectsThroughInternalLoopbackRoute(t *testing.T) {
	t.Parallel()

	baseURL := startDashboardServer(t)
	token := loginAsBootstrapAdmin(t, baseURL)
	client := &http.Client{Timeout: 10 * time.Second}

	serversReq, err := http.NewRequest(http.MethodGet, baseURL+"/api/mcp/servers", nil)
	if err != nil {
		t.Fatalf("new servers request: %v", err)
	}
	serversReq.Header.Set("Authorization", "Bearer "+token)

	serversResp, err := client.Do(serversReq)
	if err != nil {
		t.Fatalf("list servers request failed: %v", err)
	}
	defer serversResp.Body.Close()

	if serversResp.StatusCode != http.StatusOK {
		t.Fatalf("list servers status = %d, want %d", serversResp.StatusCode, http.StatusOK)
	}

	var serversPayload mcpServersResponse
	decodeErr := json.NewDecoder(serversResp.Body).Decode(&serversPayload)
	if decodeErr != nil {
		t.Fatalf("decode servers response: %v", decodeErr)
	}

	if len(serversPayload.Servers) == 0 {
		t.Fatalf("expected built-in MCP server to be registered")
	}

	expectedURL := baseURL + internalOpenClawMCPPath
	if serversPayload.Servers[0].Config.ID != mcp.BuiltinOpenClawServerID {
		t.Fatalf("server id = %q, want %q", serversPayload.Servers[0].Config.ID, mcp.BuiltinOpenClawServerID)
	}
	if serversPayload.Servers[0].Config.Connection.URL != expectedURL {
		t.Fatalf("server url = %q, want %q", serversPayload.Servers[0].Config.Connection.URL, expectedURL)
	}

	connectReq, err := http.NewRequest(
		http.MethodPost,
		baseURL+"/api/mcp/servers/"+mcp.BuiltinOpenClawServerID+"/connect",
		nil,
	)
	if err != nil {
		t.Fatalf("new connect request: %v", err)
	}
	connectReq.Header.Set("Authorization", "Bearer "+token)

	connectResp, err := client.Do(connectReq)
	if err != nil {
		t.Fatalf("connect request failed: %v", err)
	}
	defer connectResp.Body.Close()

	if connectResp.StatusCode != http.StatusOK {
		t.Fatalf("connect status = %d, want %d", connectResp.StatusCode, http.StatusOK)
	}

	toolsReq, err := http.NewRequest(http.MethodGet, baseURL+"/api/mcp/tools", nil)
	if err != nil {
		t.Fatalf("new tools request: %v", err)
	}
	toolsReq.Header.Set("Authorization", "Bearer "+token)

	toolsResp, err := client.Do(toolsReq)
	if err != nil {
		t.Fatalf("list tools request failed: %v", err)
	}
	defer toolsResp.Body.Close()

	if toolsResp.StatusCode != http.StatusOK {
		t.Fatalf("list tools status = %d, want %d", toolsResp.StatusCode, http.StatusOK)
	}

	var toolsPayload mcpToolsResponse
	if err := json.NewDecoder(toolsResp.Body).Decode(&toolsPayload); err != nil {
		t.Fatalf("decode tools response: %v", err)
	}

	if len(toolsPayload.Tools) == 0 {
		t.Fatalf("expected claw MCP tools after connect")
	}

	foundListTeams := false
	for _, tool := range toolsPayload.Tools {
		if tool.Name == "claw_list_teams" {
			foundListTeams = true
			break
		}
	}
	if !foundListTeams {
		t.Fatalf("expected claw_list_teams in MCP tools, got %+v", toolsPayload.Tools)
	}
}

func startDashboardServer(t *testing.T) string {
	t.Helper()

	tempDir := t.TempDir()
	staticDir := filepath.Join(tempDir, "static")
	if err := os.MkdirAll(staticDir, 0o755); err != nil {
		t.Fatalf("mkdir static dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(staticDir, "index.html"), []byte("ok"), 0o644); err != nil {
		t.Fatalf("write index.html: %v", err)
	}

	configPath := filepath.Join(tempDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte("router: {}\n"), 0o644); err != nil {
		t.Fatalf("write config.yaml: %v", err)
	}

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}

	port := listener.Addr().(*net.TCPAddr).Port
	cfg := &config.Config{
		Port:                   strconv.Itoa(port),
		AuthDBPath:             filepath.Join(tempDir, "auth.db"),
		JWTSecret:              "test-secret",
		JWTExpiryHours:         1,
		BootstrapAdminEmail:    "admin@example.com",
		BootstrapAdminPassword: "secret-password",
		BootstrapAdminName:     "Admin",
		StaticDir:              staticDir,
		ConfigFile:             configPath,
		AbsConfigPath:          configPath,
		ConfigDir:              tempDir,
		RouterAPIURL:           "http://127.0.0.1:8080",
		RouterMetrics:          "http://127.0.0.1:9190/metrics",
		MCPEnabled:             true,
		OpenClawEnabled:        true,
		OpenClawDataDir:        filepath.Join(tempDir, "openclaw"),
		EvaluationEnabled:      false,
		MLPipelineEnabled:      false,
	}

	server := &http.Server{
		Handler:           Setup(cfg),
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		_ = server.Serve(listener)
	}()

	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(ctx)
	})

	return "http://" + listener.Addr().String()
}

func loginAsBootstrapAdmin(t *testing.T, baseURL string) string {
	t.Helper()

	body := bytes.NewBufferString(`{"email":"admin@example.com","password":"secret-password"}`)
	resp, err := http.Post(baseURL+"/api/auth/login", "application/json", body)
	if err != nil {
		t.Fatalf("login request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("login status = %d, want %d", resp.StatusCode, http.StatusOK)
	}

	var payload loginResponse
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		t.Fatalf("decode login response: %v", err)
	}
	if payload.Token == "" {
		t.Fatalf("expected login token")
	}
	return payload.Token
}

func httptestRecorder() *responseRecorder {
	return &responseRecorder{header: make(http.Header)}
}

func httptestRequest(method, path, remoteAddr string) *http.Request {
	req, err := http.NewRequest(method, "http://example.com"+path, nil)
	if err != nil {
		panic(err)
	}
	req.RemoteAddr = remoteAddr
	return req
}

type responseRecorder struct {
	header http.Header
	body   bytes.Buffer
	Code   int
}

func (r *responseRecorder) Header() http.Header {
	return r.header
}

func (r *responseRecorder) WriteHeader(statusCode int) {
	r.Code = statusCode
}

func (r *responseRecorder) Write(p []byte) (int, error) {
	if r.Code == 0 {
		r.Code = http.StatusOK
	}
	return r.body.Write(p)
}
