package handlers

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/mcp"
)

// A stored streamable-http server, the shape the edit dialog round-trips.
func storedStreamableServer(id string) *mcp.ServerConfig {
	return &mcp.ServerConfig{
		ID:         id,
		Name:       "Stored server",
		Transport:  mcp.TransportStreamableHTTP,
		Connection: mcp.ConnectionConfig{URL: "http://mcp.internal:8100/mcp"},
	}
}

// The merge starts from the request body, so a partial update strips the
// transport and the URL from the merged result. The update is refused
// before it persists and the stored configuration keeps what the dialog
// did not touch (#4338).
func TestMCPUpdateRefusesAPartialBodyThatStripsTheTransport(t *testing.T) {
	t.Parallel()
	handler, manager := newSecurityTestHandler(t, storedStreamableServer("strip-transport"))

	recorder := serveMCPRequest(
		handler.UpdateServerHandler(),
		http.MethodPut,
		"/api/mcp/servers/strip-transport",
		map[string]any{"name": "Renamed server"},
	)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	if !strings.Contains(recorder.Body.String(), "Transport is required") {
		t.Fatalf("message must name the stripped invariant: %q", recorder.Body.String())
	}
	stored, ok := manager.GetServer("strip-transport")
	if !ok {
		t.Fatal("stored server disappeared")
	}
	if stored.Transport != mcp.TransportStreamableHTTP ||
		stored.Connection.URL != "http://mcp.internal:8100/mcp" {
		t.Fatalf("stored configuration was modified: %#v", stored)
	}
}

func TestMCPUpdateRefusesAnUnknownTransportOnTheMergedResult(t *testing.T) {
	t.Parallel()
	handler, manager := newSecurityTestHandler(t, storedStreamableServer("bad-transport"))

	recorder := serveMCPRequest(
		handler.UpdateServerHandler(),
		http.MethodPut,
		"/api/mcp/servers/bad-transport",
		map[string]any{"name": "Renamed server", "transport": "carrier-pigeon"},
	)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	if !strings.Contains(recorder.Body.String(), "Invalid transport type") {
		t.Fatalf("message must name the invalid transport: %q", recorder.Body.String())
	}
	if stored, _ := manager.GetServer("bad-transport"); stored.Transport != mcp.TransportStreamableHTTP {
		t.Fatalf("stored transport was modified: %#v", stored)
	}
}

// The collection test path dials the configuration it is given, so an
// incomplete one is refused before the dial (#4338).
func TestMCPTestConnectionRefusesAnIncompleteCollectionConfig(t *testing.T) {
	t.Parallel()
	handler, _ := newSecurityTestHandler(t, nil)

	recorder := serveMCPRequest(
		handler.TestConnectionHandler(),
		http.MethodPost,
		"/api/mcp/servers/test",
		map[string]any{"name": "Probe", "transport": "streamable-http"},
	)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	if !strings.Contains(recorder.Body.String(), "URL is required for streamable-http transport") {
		t.Fatalf("message must name the missing field: %q", recorder.Body.String())
	}
}

// A complete configuration the server cannot reach stays a test failure,
// so the refusal above is the validation and not every failure (#4338).
func TestMCPTestConnectionAnswersAnUnreachableServerAsATestFailure(t *testing.T) {
	t.Parallel()
	handler, _ := newSecurityTestHandler(t, nil)

	config := storedStreamableServer("")
	config.Connection.URL = "http://127.0.0.1:1/mcp"
	recorder := serveMCPRequest(
		handler.TestConnectionHandler(),
		http.MethodPost,
		"/api/mcp/servers/test",
		config,
	)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	var response struct {
		Success bool   `json:"success"`
		Error   string `json:"error"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("body=%s: %v", recorder.Body.String(), err)
	}
	if response.Success || response.Error == "" {
		t.Fatalf("an unreachable server must answer a test failure: %#v", response)
	}
}

// Create delegates to the same validation, and a refused configuration is
// not stored (#4338).
func TestMCPCreateStillRefusesAnIncompleteConfiguration(t *testing.T) {
	t.Parallel()
	handler, manager := newSecurityTestHandler(t, nil)

	recorder := serveMCPRequest(
		handler.CreateServerHandler(),
		http.MethodPost,
		"/api/mcp/servers",
		map[string]any{"id": "create-incomplete", "name": "No transport"},
	)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	if !strings.Contains(recorder.Body.String(), "Transport is required") {
		t.Fatalf("message must name the missing field: %q", recorder.Body.String())
	}
	if _, ok := manager.GetServer("create-incomplete"); ok {
		t.Fatal("an incomplete configuration was stored")
	}
}
