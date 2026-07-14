package handlers

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/mcp"
)

func TestMCPServerViewsNeverSerializeConnectionSecretsOrHostPaths(t *testing.T) {
	t.Parallel()

	config := &mcp.ServerConfig{
		ID:          "secret-server",
		Name:        "Secret server",
		Description: "safe description",
		Transport:   mcp.TransportStreamableHTTP,
		Connection: mcp.ConnectionConfig{
			Command: "/host/private/DataDir/bin/npx",
			Args:    []string{"--token", "argument-secret"},
			Env:     map[string]string{"API_KEY": "environment-secret"},
			Cwd:     "/host/private/DataDir/workspace",
			URL:     "https://user:url-password@example.test/mcp?token=query-secret#private",
			Headers: map[string]string{"Authorization": "Bearer header-secret"},
		},
		Security: &mcp.SecurityConfig{
			OAuth: &mcp.OAuthConfig{
				ClientID:         "public-client-id",
				ClientSecret:     "oauth-client-secret",
				AuthorizationURL: "https://oauth-user:oauth-pass@example.test/authorize?token=authorization-secret",
				TokenURL:         "https://example.test/token?client_secret=token-secret",
			},
		},
	}
	state := &mcp.ServerState{Config: config, Status: mcp.StatusError, Error: "dial tcp raw-upstream-secret"}
	encoded, err := json.Marshal(newMCPServerStateView(state, true))
	if err != nil {
		t.Fatal(err)
	}
	payload := string(encoded)
	for _, secret := range []string{
		"argument-secret",
		"environment-secret",
		"/host/private/DataDir",
		"query-secret",
		"header-secret",
		"oauth-client-secret",
		"oauth-pass",
		"authorization-secret",
		"token-secret",
		"raw-upstream-secret",
	} {
		if strings.Contains(payload, secret) {
			t.Fatalf("secret %q leaked in MCP response: %s", secret, payload)
		}
	}
	for _, expected := range []string{
		`"command":"npx"`,
		`"url":"https://example.test/mcp"`,
		`"arguments_configured":true`,
		`"environment_configured":true`,
		`"working_directory_configured":true`,
		`"headers_configured":true`,
		`"client_secret_configured":true`,
		`"error":"connection failed"`,
	} {
		if !strings.Contains(payload, expected) {
			t.Fatalf("expected %s in safe response: %s", expected, payload)
		}
	}
}

func TestMCPServerUpdatePreservesOmittedSecretsAndClearsExplicitValues(t *testing.T) {
	t.Parallel()

	existing := &mcp.ServerConfig{
		ID:        "server-1",
		Name:      "Before",
		Transport: mcp.TransportStreamableHTTP,
		Connection: mcp.ConnectionConfig{
			URL:     "https://example.test/mcp?token=url-secret",
			Args:    []string{"secret-argument"},
			Env:     map[string]string{"TOKEN": "environment-secret"},
			Cwd:     "/host/private/workspace",
			Headers: map[string]string{"Authorization": "header-secret"},
		},
		Security: &mcp.SecurityConfig{OAuth: &mcp.OAuthConfig{ClientSecret: "oauth-secret"}},
		Options:  &mcp.ServerOptions{Timeout: 30000, MaxRetries: 4},
	}
	name := "After"
	redactedURL, _ := safeMCPURL(existing.Connection.URL)
	patch := &mcpServerUpdateRequest{
		Name: &name,
		Connection: &mcpConnectionPatch{
			URL: &redactedURL,
		},
	}
	updated, err := applyMCPServerUpdate(existing, patch)
	if err != nil {
		t.Fatal(err)
	}
	if updated.Name != name || updated.Connection.URL != existing.Connection.URL ||
		updated.Connection.Headers["Authorization"] != "header-secret" ||
		updated.Connection.Env["TOKEN"] != "environment-secret" ||
		updated.Connection.Cwd != existing.Connection.Cwd ||
		updated.Security.OAuth.ClientSecret != "oauth-secret" ||
		updated.Options.Timeout != 30000 || updated.Options.MaxRetries != 4 {
		t.Fatalf("omitted or redacted values were not preserved: %+v", updated)
	}

	emptyHeaders := map[string]string{}
	emptyEnvironment := map[string]string{}
	emptyArgs := []string{}
	empty := ""
	cleared, err := applyMCPServerUpdate(updated, &mcpServerUpdateRequest{
		Connection: &mcpConnectionPatch{
			Args:    &emptyArgs,
			Env:     &emptyEnvironment,
			Cwd:     &empty,
			Headers: &emptyHeaders,
		},
		Security: &mcpSecurityPatch{OAuth: &mcpOAuthPatch{ClientSecret: &empty}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(cleared.Connection.Args) != 0 || len(cleared.Connection.Env) != 0 ||
		cleared.Connection.Cwd != "" || len(cleared.Connection.Headers) != 0 ||
		cleared.Security.OAuth.ClientSecret != "" {
		t.Fatalf("explicit empty values did not clear secrets: %+v", cleared)
	}
	replacedURL, err := applyMCPServerUpdate(updated, &mcpServerUpdateRequest{
		Connection: &mcpConnectionPatch{URL: &redactedURL, ReplaceURL: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	if replacedURL.Connection.URL != redactedURL || strings.Contains(replacedURL.Connection.URL, "url-secret") {
		t.Fatalf("replace_url sentinel did not replace redacted URL: %+v", replacedURL.Connection)
	}
}

func TestMCPTestConnectionResolvesRedactedExistingConfigWithoutLosingSecrets(t *testing.T) {
	t.Parallel()

	manager, err := mcp.NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	existing := &mcp.ServerConfig{
		ID:        "test-existing",
		Name:      "Existing",
		Transport: mcp.TransportStreamableHTTP,
		Connection: mcp.ConnectionConfig{
			URL:     "https://example.test/mcp?token=url-secret",
			Headers: map[string]string{"Authorization": "header-secret"},
		},
		Security: &mcp.SecurityConfig{OAuth: &mcp.OAuthConfig{ClientSecret: "oauth-secret"}},
	}
	if addErr := manager.AddServer(existing); addErr != nil {
		t.Fatal(addErr)
	}
	redactedURL, _ := safeMCPURL(existing.Connection.URL)
	resolved, err := resolveMCPServerTestConfig(manager, &mcpServerUpdateRequest{
		ID:         stringPointer(existing.ID),
		Name:       stringPointer(existing.Name),
		Transport:  transportPointer(existing.Transport),
		Connection: &mcpConnectionPatch{URL: &redactedURL},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resolved.Connection.URL != existing.Connection.URL ||
		resolved.Connection.Headers["Authorization"] != "header-secret" ||
		resolved.Security.OAuth.ClientSecret != "oauth-secret" {
		t.Fatalf("resolved test config lost secrets: %+v", resolved)
	}
}

func TestMCPCreateResponseIsSecretFreeWhileStorageRetainsCredentials(t *testing.T) {
	t.Parallel()

	manager, err := mcp.NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	handler := NewMCPHandler(manager, false)
	body := `{
		"id":"http-server",
		"name":"HTTP server",
		"transport":"streamable-http",
		"connection":{
			"url":"https://example.test/mcp?token=url-secret",
			"headers":{"Authorization":"Bearer response-header-secret"}
		},
		"enabled":true,
		"security":{"oauth":{"client_id":"client","client_secret":"response-oauth-secret","authorization_url":"https://example.test/authorize","token_url":"https://example.test/token"}}
	}`
	recorder := httptest.NewRecorder()
	request := newMCPJSONRequest(http.MethodPost, "/api/mcp/servers", body, auth.RoleAdmin, true)
	handler.CreateServerHandler().ServeHTTP(recorder, request)
	if recorder.Code != http.StatusCreated {
		t.Fatalf("status = %d, want 201; body=%s", recorder.Code, recorder.Body.String())
	}
	if strings.Contains(recorder.Body.String(), "response-header-secret") ||
		strings.Contains(recorder.Body.String(), "response-oauth-secret") ||
		strings.Contains(recorder.Body.String(), "url-secret") ||
		strings.Contains(recorder.Body.String(), `"headers":`) ||
		strings.Contains(recorder.Body.String(), `"client_secret":`) {
		t.Fatalf("create response leaked credential: %s", recorder.Body.String())
	}
	stored, ok := manager.GetServer("http-server")
	if !ok || stored.Connection.Headers["Authorization"] != "Bearer response-header-secret" ||
		stored.Security.OAuth.ClientSecret != "response-oauth-secret" ||
		!strings.Contains(stored.Connection.URL, "url-secret") {
		t.Fatalf("stored credentials were unexpectedly lost: %+v", stored)
	}
}

func TestMCPCreateRejectsExternalLocalOnlyEndpoint(t *testing.T) {
	t.Parallel()

	manager, err := mcp.NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(manager.Close)
	handler := NewMCPHandler(manager, false)
	body := `{
		"id":"external-local-only",
		"name":"External local-only server",
		"transport":"streamable-http",
		"connection":{"url":"https://mcp.example.test/mcp"},
		"security":{"local_only":true}
	}`
	recorder := httptest.NewRecorder()
	request := newMCPJSONRequest(http.MethodPost, "/api/mcp/servers", body, auth.RoleAdmin, true)
	handler.CreateServerHandler().ServeHTTP(recorder, request)
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", recorder.Code, recorder.Body.String())
	}
	if _, ok := manager.GetServer("external-local-only"); ok {
		t.Fatal("rejected local-only server was persisted")
	}
}

func TestMCPJSONBodiesAreBoundedStrictAndSingleValue(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		contentType string
		body        string
		wantStatus  int
	}{
		{
			name:        "unknown field",
			contentType: "application/json",
			body:        `{"name":"server","transport":"streamable-http","connection":{"url":"https://example.test/mcp"},"unexpected":true}`,
			wantStatus:  http.StatusBadRequest,
		},
		{
			name:        "trailing value",
			contentType: "application/json",
			body:        `{"name":"server","transport":"streamable-http","connection":{"url":"https://example.test/mcp"}} {}`,
			wantStatus:  http.StatusBadRequest,
		},
		{
			name:        "wrong content type",
			contentType: "text/plain",
			body:        `{"name":"server"}`,
			wantStatus:  http.StatusUnsupportedMediaType,
		},
		{
			name:        "route ambiguous server id",
			contentType: "application/json",
			body:        `{"id":"tenant/server","name":"server","transport":"streamable-http","connection":{"url":"https://example.test/mcp"}}`,
			wantStatus:  http.StatusBadRequest,
		},
		{
			name:        "oversized",
			contentType: "application/json",
			body:        `{"name":"` + strings.Repeat("x", mcpConfigBodyLimit) + `"}`,
			wantStatus:  http.StatusRequestEntityTooLarge,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			manager, err := mcp.NewManager(nil)
			if err != nil {
				t.Fatal(err)
			}
			handler := NewMCPHandler(manager, false)
			recorder := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodPost, "/api/mcp/servers", strings.NewReader(test.body))
			request.Header.Set("Content-Type", test.contentType)
			request = request.WithContext(auth.WithAuthContext(request.Context(), mcpAuthContext(auth.RoleAdmin, true)))
			handler.CreateServerHandler().ServeHTTP(recorder, request)
			if recorder.Code != test.wantStatus {
				t.Fatalf("status = %d, want %d; body=%s", recorder.Code, test.wantStatus, recorder.Body.String())
			}
		})
	}
}

func TestMCPProtocolBodyIsBoundedAndContainsOneLosslessJSONValue(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		contentType string
		body        []byte
		wantStatus  int
	}{
		{name: "valid", contentType: "application/json; charset=utf-8", body: []byte(`{"jsonrpc":"2.0","method":"ping"}`)},
		{name: "wrong content type", contentType: "text/plain", body: []byte(`{}`), wantStatus: http.StatusUnsupportedMediaType},
		{name: "trailing JSON", contentType: "application/json", body: []byte(`{} {}`), wantStatus: http.StatusBadRequest},
		{name: "invalid Unicode", contentType: "application/json", body: []byte{'{', '"', 'x', '"', ':', '"', 0xff, '"', '}'}, wantStatus: http.StatusBadRequest},
		{name: "oversized", contentType: "application/json", body: bytes.Repeat([]byte{' '}, mcpToolBodyLimit+1), wantStatus: http.StatusRequestEntityTooLarge},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			recorder := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodPost, "/api/openclaw/mcp", bytes.NewReader(test.body))
			request.Header.Set("Content-Type", test.contentType)
			status, err := boundMCPProtocolJSON(recorder, request)
			if test.wantStatus == 0 {
				if err != nil || status != 0 {
					t.Fatalf("boundMCPProtocolJSON() = (%d, %v), want success", status, err)
				}
				resetBody, readErr := io.ReadAll(request.Body)
				if readErr != nil || !bytes.Equal(resetBody, test.body) {
					t.Fatalf("reset body = %q, err=%v; want %q", resetBody, readErr, test.body)
				}
				return
			}
			if err == nil || status != test.wantStatus {
				t.Fatalf("boundMCPProtocolJSON() = (%d, %v), want status %d", status, err, test.wantStatus)
			}
		})
	}
}

func TestMCPStdioRequiresAdminAndExplicitDevelopmentCapability(t *testing.T) {
	t.Parallel()

	devManager, err := mcp.NewManagerWithOptions(nil, mcp.ManagerOptions{AllowStdio: true})
	if err != nil {
		t.Fatal(err)
	}
	devHandler := NewMCPHandler(devManager, false)
	writeRequest := newMCPJSONRequest(
		http.MethodPost,
		"/api/mcp/servers",
		`{"id":"write-stdio","name":"stdio","transport":"stdio","connection":{"command":"not-executed"}}`,
		auth.RoleWrite,
		true,
	)
	writeRecorder := httptest.NewRecorder()
	devHandler.CreateServerHandler().ServeHTTP(writeRecorder, writeRequest)
	if writeRecorder.Code != http.StatusForbidden {
		t.Fatalf("write role status = %d, want 403; body=%s", writeRecorder.Code, writeRecorder.Body.String())
	}

	adminRequest := newMCPJSONRequest(
		http.MethodPost,
		"/api/mcp/servers",
		`{"id":"admin-stdio","name":"stdio","transport":"stdio","connection":{"command":"not-executed"}}`,
		auth.RoleAdmin,
		true,
	)
	adminRecorder := httptest.NewRecorder()
	devHandler.CreateServerHandler().ServeHTTP(adminRecorder, adminRequest)
	if adminRecorder.Code != http.StatusCreated {
		t.Fatalf("development admin status = %d, want 201; body=%s", adminRecorder.Code, adminRecorder.Body.String())
	}

	productionManager, err := mcp.NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	productionHandler := NewMCPHandler(productionManager, false)
	productionRequest := newMCPJSONRequest(
		http.MethodPost,
		"/api/mcp/servers",
		`{"id":"production-stdio","name":"stdio","transport":"stdio","connection":{"command":"not-executed"}}`,
		auth.RoleAdmin,
		true,
	)
	productionRecorder := httptest.NewRecorder()
	productionHandler.CreateServerHandler().ServeHTTP(productionRecorder, productionRequest)
	if productionRecorder.Code != http.StatusForbidden {
		t.Fatalf("production admin status = %d, want 403; body=%s", productionRecorder.Code, productionRecorder.Body.String())
	}
}

func TestMCPBuiltinToolsRequireLiveOpenClawManagePermission(t *testing.T) {
	manager, err := mcp.NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	handler := NewMCPHandler(manager, false)
	body := `{"server_id":"` + mcp.BuiltinOpenClawServerID + `","tool_name":"claw_list_teams","arguments":{}}`

	writeService, _, writeToken := newOpenClawAuthorizationUser(t, auth.RoleWrite)
	writeRecorder := httptest.NewRecorder()
	writeRequest := httptest.NewRequest(http.MethodPost, "/api/mcp/tools/execute", strings.NewReader(body))
	writeRequest.Header.Set("Content-Type", "application/json")
	writeRequest.Header.Set("Authorization", "Bearer "+writeToken)
	auth.AuthenticateRequest(writeService)(handler.ExecuteToolHandler()).ServeHTTP(writeRecorder, writeRequest)
	if writeRecorder.Code != http.StatusForbidden {
		t.Fatalf("write status = %d, want 403; body=%s", writeRecorder.Code, writeRecorder.Body.String())
	}

	testRecorder := httptest.NewRecorder()
	testRequest := httptest.NewRequest(
		http.MethodPost,
		"/api/mcp/servers/"+mcp.BuiltinOpenClawServerID+"/test",
		strings.NewReader(`{"id":"`+mcp.BuiltinOpenClawServerID+`"}`),
	)
	testRequest.Header.Set("Content-Type", "application/json")
	testRequest.Header.Set("Authorization", "Bearer "+writeToken)
	auth.AuthenticateRequest(writeService)(handler.TestConnectionHandler()).ServeHTTP(testRecorder, testRequest)
	if testRecorder.Code != http.StatusForbidden {
		t.Fatalf("write builtin test status = %d, want 403; body=%s", testRecorder.Code, testRecorder.Body.String())
	}

	adminService, _, adminToken := newOpenClawAuthorizationUser(t, auth.RoleAdmin)
	adminRecorder := httptest.NewRecorder()
	adminRequest := httptest.NewRequest(http.MethodPost, "/api/mcp/tools/execute", strings.NewReader(body))
	adminRequest.Header.Set("Content-Type", "application/json")
	adminRequest.Header.Set("Authorization", "Bearer "+adminToken)
	auth.AuthenticateRequest(adminService)(handler.ExecuteToolHandler()).ServeHTTP(adminRecorder, adminRequest)
	if adminRecorder.Code != http.StatusBadGateway {
		t.Fatalf("admin status = %d, want 502 from disconnected test manager; body=%s", adminRecorder.Code, adminRecorder.Body.String())
	}
}

func TestMCPTestConnectionUsesPathServerID(t *testing.T) {
	t.Parallel()

	manager, err := mcp.NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(manager.Close)
	handler := NewMCPHandler(manager, false)
	recorder := httptest.NewRecorder()
	request := newMCPJSONRequest(
		http.MethodPost,
		"/api/mcp/servers/path-server/test",
		`{"id":"different-server","name":"server","transport":"streamable-http","connection":{"url":"https://example.test/mcp"}}`,
		auth.RoleAdmin,
		true,
	)
	handler.TestConnectionHandler().ServeHTTP(recorder, request)
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", recorder.Code, recorder.Body.String())
	}
}

func TestFilterMCPToolsHidesBuiltinWithoutOpenClawManage(t *testing.T) {
	t.Parallel()

	tools := []mcp.Tool{
		{ToolDefinition: mcp.ToolDefinition{Name: "external"}, ServerID: "external-server"},
		{ToolDefinition: mcp.ToolDefinition{Name: "claw_list_teams"}, ServerID: mcp.BuiltinOpenClawServerID},
	}
	filtered := filterMCPTools(tools, false)
	if len(filtered) != 1 || filtered[0].Name != "external" {
		t.Fatalf("filtered tools = %+v, want only external tool", filtered)
	}
	allowed := filterMCPTools(tools, true)
	if len(allowed) != len(tools) {
		t.Fatalf("allowed tools = %+v, want all tools", allowed)
	}
}

func newMCPJSONRequest(method, path, body, role string, manage bool) *http.Request {
	request := httptest.NewRequest(method, path, strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	return request.WithContext(auth.WithAuthContext(request.Context(), mcpAuthContext(role, manage)))
}

func mcpAuthContext(role string, manage bool) auth.AuthContext {
	permissions := map[string]bool{
		auth.PermMcpRead:  true,
		auth.PermToolsUse: true,
	}
	if manage {
		permissions[auth.PermMcpManage] = true
	}
	if role == auth.RoleAdmin {
		permissions[auth.PermOpenClaw] = true
	}
	return auth.AuthContext{Role: role, Perms: permissions}
}

func stringPointer(value string) *string {
	return &value
}

func transportPointer(value mcp.TransportType) *mcp.TransportType {
	return &value
}
