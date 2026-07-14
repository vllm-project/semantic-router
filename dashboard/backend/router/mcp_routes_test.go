package router

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
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
				URL     string          `json:"url"`
				Headers json.RawMessage `json:"headers"`
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

func TestInternalOpenClawMCPOnlyRequiresLoopbackAndCapability(t *testing.T) {
	t.Parallel()

	const capability = "test-capability-with-sufficient-entropy"
	handler := internalOpenClawMCPOnly(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}), capability)

	tests := []struct {
		name       string
		remoteAddr string
		headers    []string
		wantStatus int
	}{
		{name: "missing capability", remoteAddr: "127.0.0.1:3000", wantStatus: http.StatusForbidden},
		{name: "wrong capability", remoteAddr: "127.0.0.1:3000", headers: []string{"wrong"}, wantStatus: http.StatusForbidden},
		{name: "duplicate capability", remoteAddr: "127.0.0.1:3000", headers: []string{capability, capability}, wantStatus: http.StatusForbidden},
		{name: "non-loopback", remoteAddr: "203.0.113.10:3000", headers: []string{capability}, wantStatus: http.StatusForbidden},
		{name: "valid internal client", remoteAddr: "[::1]:3000", headers: []string{capability}, wantStatus: http.StatusNoContent},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			req := httptestRequest(http.MethodPost, internalOpenClawMCPPath, test.remoteAddr)
			for _, value := range test.headers {
				req.Header.Add(internalOpenClawMCPCapabilityHeader, value)
			}
			recorder := httptestRecorder()
			handler.ServeHTTP(recorder, req)
			if recorder.Code != test.wantStatus {
				t.Fatalf("status = %d, want %d", recorder.Code, test.wantStatus)
			}
		})
	}
}

func TestGenerateInternalOpenClawMCPCapabilityFailsClosed(t *testing.T) {
	t.Parallel()

	if _, err := generateInternalOpenClawMCPCapability(nil); err == nil {
		t.Fatal("expected nil random source to fail")
	}
	if _, err := generateInternalOpenClawMCPCapability(io.LimitReader(bytes.NewReader(make([]byte, 8)), 8)); err == nil {
		t.Fatal("expected short random source to fail")
	}

	first, err := generateInternalOpenClawMCPCapability(bytes.NewReader(bytes.Repeat([]byte{0x11}, internalOpenClawMCPCapabilityBytes)))
	if err != nil {
		t.Fatalf("generate capability: %v", err)
	}
	second, err := generateInternalOpenClawMCPCapability(bytes.NewReader(bytes.Repeat([]byte{0x22}, internalOpenClawMCPCapabilityBytes)))
	if err != nil {
		t.Fatalf("generate second capability: %v", err)
	}
	if first == "" || first == second || len(first) < 40 {
		t.Fatalf("unexpected generated capabilities: first_len=%d equal=%t", len(first), first == second)
	}
}

func TestSetupMCPOnlyAllowsStdioInExplicitDevelopmentProfile(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		profile   string
		wantStdio bool
	}{
		{name: "production", profile: config.DashboardSecurityProfileProduction, wantStdio: false},
		{name: "unset fails closed", profile: "", wantStdio: false},
		{name: "development", profile: config.DashboardSecurityProfileDevelopment, wantStdio: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			manager, err := SetupMCP(http.NewServeMux(), &config.Config{
				MCPEnabled:      true,
				SecurityProfile: test.profile,
			}, nil, nil)
			if err != nil {
				t.Fatalf("SetupMCP() error = %v", err)
			}
			if manager == nil || manager.AllowsStdio() != test.wantStdio {
				t.Fatalf("AllowsStdio() = %t, want %t", manager != nil && manager.AllowsStdio(), test.wantStdio)
			}
			manager.DisconnectAll()
		})
	}
}

func TestBuiltInOpenClawMCPConnectsThroughInternalLoopbackRoute(t *testing.T) {
	t.Parallel()

	baseURL := startDashboardServer(t)
	token := loginAsBootstrapAdmin(t, baseURL)
	client := &http.Client{Timeout: 10 * time.Second}

	t.Run("registers secret-free built-in endpoint", func(t *testing.T) {
		assertBuiltinMCPServerRegistered(t, client, baseURL, token)
	})
	t.Run("rejects internal requests without capability", func(t *testing.T) {
		assertInternalMCPRequiresCapability(t, client, baseURL)
	})
	t.Run("connects built-in endpoint", func(t *testing.T) {
		connectBuiltinMCPServer(t, client, baseURL, token)
	})
	t.Run("publishes built-in tools", func(t *testing.T) {
		assertBuiltinMCPToolsPublished(t, client, baseURL, token)
	})
}

func assertBuiltinMCPServerRegistered(t *testing.T, client *http.Client, baseURL, token string) {
	t.Helper()

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
	if len(serversPayload.Servers[0].Config.Connection.Headers) != 0 {
		t.Fatalf("internal capability header leaked in server response: %s", serversPayload.Servers[0].Config.Connection.Headers)
	}
}

func assertInternalMCPRequiresCapability(t *testing.T, client *http.Client, baseURL string) {
	t.Helper()

	unauthorizedInternalResp, err := client.Get(baseURL + internalOpenClawMCPPath)
	if err != nil {
		t.Fatalf("call internal endpoint without capability: %v", err)
	}
	defer unauthorizedInternalResp.Body.Close()
	if unauthorizedInternalResp.StatusCode != http.StatusForbidden {
		t.Fatalf("internal endpoint without capability status = %d, want 403", unauthorizedInternalResp.StatusCode)
	}
}

func connectBuiltinMCPServer(t *testing.T, client *http.Client, baseURL, token string) {
	t.Helper()

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
}

func assertBuiltinMCPToolsPublished(t *testing.T, client *http.Client, baseURL, token string) {
	t.Helper()

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
		JWTSecret:              "0123456789abcdef0123456789abcdef",
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
		WorkflowDBPath:         filepath.Join(tempDir, "workflow.sqlite"),
		ConfigProjectionDBPath: filepath.Join(tempDir, "config-projection.sqlite"),
		EvaluationEnabled:      false,
		MLPipelineEnabled:      false,
	}

	dashboard, err := Setup(cfg)
	if err != nil {
		t.Fatalf("Setup() error = %v", err)
	}
	server := &http.Server{
		Handler:           dashboard.Handler,
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		_ = server.Serve(listener)
	}()

	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(ctx)
		_ = dashboard.Close()
	})

	return "http://" + listener.Addr().String()
}

func loginAsBootstrapAdmin(t *testing.T, baseURL string) string {
	t.Helper()

	body := bytes.NewBufferString(`{"email":"admin@example.com","password":"secret-password"}`)
	req, err := http.NewRequest(http.MethodPost, baseURL+"/api/auth/login", body)
	if err != nil {
		t.Fatalf("create login request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-VSR-Auth-Mode", "bearer")
	resp, err := http.DefaultClient.Do(req)
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
