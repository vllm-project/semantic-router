package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/mcp"
)

var handlerSecretCanaries = []string{
	"handler-argument-token-canary",
	"handler-argument-password-canary",
	"handler-argument-api-key-canary",
	"handler-argument-header-canary",
	"handler-argument-env-canary",
	"handler-argument-unknown-canary",
	"handler-environment-canary",
	"handler-header-canary",
	"handler-oauth-canary",
	"handler-url-user-canary",
	"handler-url-password-canary",
	"handler-url-path-canary",
	"handler-url-query-token-canary",
	"handler-url-query-api-key-canary",
	"handler-url-query-signature-canary",
	"handler-url-query-credential-canary",
	"handler-url-query-auth-canary",
	"handler-url-fragment-canary",
	"handler-oauth-token-url-canary",
	"handler-url-query-code-canary",
	"handler-url-opaque-fragment-canary",
	"handler-url-query-key-canary",
}

func handlerConfigWithSecrets(id string) *mcp.ServerConfig {
	return &mcp.ServerConfig{
		ID:        id,
		Name:      "Handler secret server",
		Transport: mcp.TransportStdio,
		Connection: mcp.ConnectionConfig{
			Command: "secret-server",
			Args: []string{
				"--token", handlerSecretCanaries[0],
				"--password=" + handlerSecretCanaries[1],
				"--api-key", handlerSecretCanaries[2],
				"--header", "Authorization: Bearer " + handlerSecretCanaries[3],
				"-e", "GITHUB_TOKEN=" + handlerSecretCanaries[4],
				"--github-token", handlerSecretCanaries[5],
			},
			Env: map[string]string{"API_TOKEN": handlerSecretCanaries[6]},
			URL: "https://" + handlerSecretCanaries[9] + ":" + handlerSecretCanaries[10] + "@mcp.example.test/rpc/" + handlerSecretCanaries[11] +
				"?token=" + handlerSecretCanaries[12] +
				"&apiKey=" + handlerSecretCanaries[13] +
				"&signature=" + handlerSecretCanaries[14] +
				"&credential=" + handlerSecretCanaries[15] +
				"&auth=" + handlerSecretCanaries[16] +
				"&code=" + handlerSecretCanaries[19] +
				"&view=ordinary&" + handlerSecretCanaries[21] +
				"#access_token=" + handlerSecretCanaries[17] + "&opaque=" + handlerSecretCanaries[20],
			Headers: map[string]string{"Authorization": handlerSecretCanaries[7]},
		},
		Enabled: true,
		Security: &mcp.SecurityConfig{OAuth: &mcp.OAuthConfig{
			ClientID:         "handler-client",
			ClientSecret:     handlerSecretCanaries[8],
			AuthorizationURL: "https://auth.example.test/authorize",
			TokenURL: "https://auth.example.test/token?token=" + handlerSecretCanaries[18] +
				"&view=ordinary",
		}},
	}
}

func assertHandlerResponseRedacted(t *testing.T, body string) {
	t.Helper()
	for _, canary := range handlerSecretCanaries {
		if strings.Contains(body, canary) {
			t.Fatalf("MCP API response leaked %q: %s", canary, body)
		}
	}
	if !strings.Contains(body, mcp.RedactedValue) {
		t.Fatalf("MCP API response does not contain redaction placeholder: %s", body)
	}
}

func TestMCPReadResponsesRedactStoredServerSecrets(t *testing.T) {
	t.Parallel()
	manager, err := mcp.NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	config := handlerConfigWithSecrets("read-secret-server")
	if addErr := manager.AddServer(config); addErr != nil {
		t.Fatal(addErr)
	}
	handler := NewMCPHandler(manager, false)

	if got := auth.RequiredPermissions(http.MethodGet, "/api/mcp/servers"); !reflect.DeepEqual(got, []string{auth.PermMcpRead}) {
		t.Fatalf("list permissions = %q, want %q", got, []string{auth.PermMcpRead})
	}
	if got := auth.RequiredPermissions(http.MethodGet, "/api/mcp/servers/"+config.ID+"/status"); !reflect.DeepEqual(got, []string{auth.PermMcpRead}) {
		t.Fatalf("status permissions = %q, want %q", got, []string{auth.PermMcpRead})
	}

	tests := []struct {
		name    string
		path    string
		handler http.HandlerFunc
	}{
		{name: "list", path: "/api/mcp/servers", handler: handler.ListServersHandler()},
		{name: "status", path: "/api/mcp/servers/" + config.ID + "/status", handler: handler.GetServerStatusHandler()},
		{name: "disconnect", path: "/api/mcp/servers/" + config.ID + "/disconnect", handler: handler.DisconnectServerHandler()},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			method := http.MethodGet
			if test.name == "disconnect" {
				method = http.MethodPost
			}
			recorder := httptest.NewRecorder()
			test.handler.ServeHTTP(recorder, httptest.NewRequest(method, test.path, nil))
			if recorder.Code != http.StatusOK {
				t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body.String())
			}
			assertHandlerResponseRedacted(t, recorder.Body.String())
		})
	}
}

func TestMCPCreateAndUpdateResponsesRedactWhileRuntimeRetainsSecrets(t *testing.T) {
	t.Parallel()
	manager, err := mcp.NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	handler := NewMCPHandler(manager, false)
	config := handlerConfigWithSecrets("write-secret-server")

	createBody, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	createRecorder := httptest.NewRecorder()
	handler.CreateServerHandler().ServeHTTP(
		createRecorder,
		httptest.NewRequest(http.MethodPost, "/api/mcp/servers", bytes.NewReader(createBody)),
	)
	if createRecorder.Code != http.StatusCreated {
		t.Fatalf("create status=%d body=%s", createRecorder.Code, createRecorder.Body.String())
	}
	assertHandlerResponseRedacted(t, createRecorder.Body.String())

	masked := mcp.RedactedServerConfig(config)
	masked.Name = "Updated handler server"
	updateBody, err := json.Marshal(masked)
	if err != nil {
		t.Fatal(err)
	}
	updateRecorder := httptest.NewRecorder()
	handler.UpdateServerHandler().ServeHTTP(
		updateRecorder,
		httptest.NewRequest(http.MethodPut, "/api/mcp/servers/"+config.ID, bytes.NewReader(updateBody)),
	)
	if updateRecorder.Code != http.StatusOK {
		t.Fatalf("update status=%d body=%s", updateRecorder.Code, updateRecorder.Body.String())
	}
	assertHandlerResponseRedacted(t, updateRecorder.Body.String())

	stored, ok := manager.GetServer(config.ID)
	if !ok {
		t.Fatal("stored server missing")
	}
	if stored.Connection.Env["API_TOKEN"] != handlerSecretCanaries[6] ||
		stored.Connection.Headers["Authorization"] != handlerSecretCanaries[7] ||
		stored.Security.OAuth.ClientSecret != handlerSecretCanaries[8] ||
		!strings.Contains(stored.Connection.URL, handlerSecretCanaries[10]) {
		t.Fatalf("runtime config did not retain original secrets: %#v", stored)
	}
}
