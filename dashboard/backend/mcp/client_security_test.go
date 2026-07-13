package mcp

import (
	"bytes"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestSchemaAndArgumentNormalizationLogsContainNoPayloadValues(t *testing.T) {
	var output bytes.Buffer
	previous := log.Writer()
	log.SetOutput(&output)
	t.Cleanup(func() { log.SetOutput(previous) })

	const secret = "never-log-this-secret"
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"credential": map[string]any{
				"type":    "string",
				"default": secret,
			},
		},
		"required": []any{"credential"},
	}
	args := map[string]any{"credential": []any{secret}}
	_ = transformInputSchema(schema)
	_ = coerceArgumentTypes(args, schema)
	_ = fillDefaultValues(map[string]any{}, schema)

	if strings.Contains(output.String(), secret) || strings.Contains(output.String(), "credential") {
		t.Fatalf("MCP normalization log leaked schema or argument content: %q", output.String())
	}
}

func TestValidateConnectionSecurityLocalOnlyAddresses(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		url     string
		wantErr bool
	}{
		{name: "IPv4 loopback", url: "http://127.0.0.1:9000/mcp"},
		{name: "IPv6 loopback", url: "https://[::1]:9000/mcp"},
		{name: "localhost", url: "http://localhost:9000/mcp"},
		{name: "localhost trailing dot", url: "http://localhost.:9000/mcp"},
		{name: "external hostname", url: "https://mcp.example.test/mcp", wantErr: true},
		{name: "non-loopback literal", url: "http://192.0.2.10/mcp", wantErr: true},
		{name: "localhost lookalike", url: "http://localhost.example.test/mcp", wantErr: true},
		{name: "unsupported scheme", url: "ftp://127.0.0.1/mcp", wantErr: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := &ServerConfig{
				Transport:  TransportStreamableHTTP,
				Connection: ConnectionConfig{URL: test.url},
				Security:   &SecurityConfig{LocalOnly: true},
			}
			err := ValidateConnectionSecurity(config)
			if test.wantErr && err == nil {
				t.Fatalf("ValidateConnectionSecurity(%q) succeeded, want error", test.url)
			}
			if !test.wantErr && err != nil {
				t.Fatalf("ValidateConnectionSecurity(%q) = %v, want nil", test.url, err)
			}
		})
	}
}

func TestSecureMCPHTTPClientDoesNotFollowRedirects(t *testing.T) {
	t.Parallel()

	destinationHit := make(chan string, 1)
	destination := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		destinationHit <- r.Header.Get("X-Test-Capability")
		w.WriteHeader(http.StatusNoContent)
	}))
	t.Cleanup(destination.Close)

	redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, destination.URL, http.StatusTemporaryRedirect)
	}))
	t.Cleanup(redirector.Close)

	config := &ServerConfig{
		Transport:  TransportStreamableHTTP,
		Connection: ConnectionConfig{URL: redirector.URL},
	}
	httpClient, err := newSecureMCPHTTPClient(config, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	request, err := http.NewRequest(http.MethodPost, redirector.URL, strings.NewReader("{}"))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("X-Test-Capability", "must-not-cross-origin")
	response, err := httpClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != http.StatusTemporaryRedirect {
		t.Fatalf("status = %d, want %d", response.StatusCode, http.StatusTemporaryRedirect)
	}

	select {
	case leaked := <-destinationHit:
		t.Fatalf("redirect destination was contacted with capability %q", leaked)
	case <-time.After(100 * time.Millisecond):
	}
}

func TestLocalOnlyMCPHTTPClientDisablesAmbientProxy(t *testing.T) {
	t.Parallel()

	config := &ServerConfig{
		Transport:  TransportStreamableHTTP,
		Connection: ConnectionConfig{URL: "http://127.0.0.1:9000/mcp"},
		Security:   &SecurityConfig{LocalOnly: true},
	}
	httpClient, err := newSecureMCPHTTPClient(config, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	transport, ok := httpClient.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("transport type = %T, want *http.Transport", httpClient.Transport)
	}
	if transport.Proxy != nil {
		t.Fatal("local-only MCP HTTP client retained ambient proxy resolution")
	}
}
