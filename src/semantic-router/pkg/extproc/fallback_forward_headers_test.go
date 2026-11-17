package extproc

import (
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
)

// A fallback attempt the Router sends itself carries the client headers its
// policy forwards, over the backend's static ones, and no other client
// header; its credential stays the Router's. The record names the headers it
// forwarded, never their values.
func TestFallbackAttemptCarriesOnlyTheForwardedClientHeaders(t *testing.T) {
	var mu sync.Mutex
	var received http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		received = r.Header.Clone()
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-forward","object":"chat.completion","created":1234567890,"model":"model-fallback-1",` +
			`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],` +
			`"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`))
	}))
	defer server.Close()
	address, err := url.Parse(server.URL)
	require.NoError(t, err)
	host, portText, err := net.SplitHostPort(address.Host)
	require.NoError(t, err)
	port, err := strconv.Atoi(portText)
	require.NoError(t, err)

	var policy fallback.FallbackPolicy
	require.NoError(t, yaml.Unmarshal([]byte("enabled: true\nforward_request_headers: [X-User-Id, x-session-tag]\n"), &policy))
	router, cfg := setupFallbackTestRouter(t, policy.WithDefaults())
	cfg.VLLMEndpoints[1].Address, cfg.VLLMEndpoints[1].Port = host, port
	withStatic := config.ProviderProfile{Type: "openai", BaseURL: server.URL, ExtraHeaders: map[string]string{"x-user-id": "router", "X-Team": "search"}}
	cfg.ProviderProfiles["prof-openai"] = withStatic
	candidate := cfg.ModelConfig["model-fallback-1"]
	candidate.AccessKeys = map[string]string{"openai": "sk-router-key"}
	cfg.ModelConfig["model-fallback-1"] = candidate

	attempt := func(client map[string]string) (http.Header, *RequestContext) {
		t.Helper()
		mu.Lock()
		received = nil
		mu.Unlock()
		ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
		ctx.UpstreamStatusCode = http.StatusServiceUnavailable
		ctx.Headers = client
		response := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
		require.NotNil(t, response.GetImmediateResponse(), "the candidate answers")
		mu.Lock()
		defer mu.Unlock()
		require.NotNil(t, received, "the candidate was called")
		return received, ctx
	}

	sent, ctx := attempt(map[string]string{"x-user-id": "alice", "x-other": "client-only", "authorization": "Bearer client-key"})
	require.Equal(t, []string{"alice"}, sent.Values("X-User-Id"), "the client's value replaces the static one")
	require.Equal(t, "search", sent.Get("X-Team"))
	require.Empty(t, sent.Values("X-Other"), "a client header the policy does not list stays off")
	require.Empty(t, sent.Values("X-Session-Tag"), "a listed header the client did not send is not sent")
	require.Equal(t, "Bearer sk-router-key", sent.Get("Authorization"))
	record, err := json.Marshal(ctx.FallbackRecord)
	require.NoError(t, err)
	require.Contains(t, string(record), `"forwarded_headers":["x-user-id"]`)
	require.NotContains(t, string(record), "alice")

	sent, _ = attempt(map[string]string{"x-other": "client-only"})
	require.Equal(t, "router", sent.Get("X-User-Id"), "without the client's value the static one stays")

	cfg.ProviderProfiles["prof-openai"] = config.ProviderProfile{Type: "openai", BaseURL: server.URL}
	sent, _ = attempt(map[string]string{"x-other": "client-only"})
	require.Empty(t, sent.Values("X-User-Id"), "nothing is sent for a header the client did not send")
}
