//go:build !windows

package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

func TestStandaloneEntrypointsOnlyWithoutClientKeys(t *testing.T) {
	arrived, release := make(chan struct{}, 8), make(chan struct{})
	close(release)
	backend := heldBackend(arrived, release)
	t.Cleanup(backend.Close)
	url, _, _ := startDrainTestGateway(t, backend.URL, `    decisions:
      - name: answer
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: m}]
entrypoints:
  - model_names: [company/public]
    recipe: default
global:
  router:
    entrypoints_only: true
    skip_processing: {enabled: true}
`)
	for _, tc := range []struct {
		model  string
		skip   bool
		status int
	}{
		{"m", false, http.StatusForbidden},
		{"m", true, http.StatusForbidden},
		{"vllm-sr/auto", false, http.StatusForbidden},
		{"company/public", false, http.StatusOK},
	} {
		body := `{"model":"` + tc.model + `","messages":[{"role":"user","content":"hi"}]}`
		req, err := http.NewRequest(http.MethodPost, url+"/v1/chat/completions", strings.NewReader(body))
		require.NoError(t, err)
		req.Header.Set("Content-Type", "application/json")
		if tc.skip {
			req.Header.Set("x-vsr-skip-processing", "true")
		}
		resp, err := http.DefaultClient.Do(req)
		require.NoError(t, err)
		out, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		require.NoError(t, err)
		require.Equal(t, tc.status, resp.StatusCode, "%s: %s", tc.model, out)
		if tc.status == http.StatusForbidden {
			require.Empty(t, arrived, "a refused request reached the backend")
		}
	}
	require.Len(t, arrived, 1, "only the public entrypoint should reach the backend")
}

// A standalone listener serves an entrypoint with client keys only to a
// request that presents one of them, and no refused request reaches a
// backend: not without a key, not with another, and not through the
// skip-processing opt-out. An admitted request reaches the backend without
// the client's key, and a public entrypoint beside it needs none.
func TestStandaloneEntrypointAdmitsOnlyItsClientKey(t *testing.T) {
	var mu sync.Mutex
	var received []string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		received = append(received, r.Header.Get("Authorization"))
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"c","object":"chat.completion","model":"m","choices":[{"index":0,`+
			`"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer backend.Close()
	calls := func() []string {
		mu.Lock()
		defer mu.Unlock()
		return append([]string(nil), received...)
	}

	t.Setenv(configsnapshot.HistoryDirEnv, t.TempDir())
	t.Setenv("FAST_CLIENT_KEY", "fast-1")
	previous := modelservice.DefaultManager()
	modelservice.SetDefault(nil)
	t.Cleanup(func() { modelservice.SetDefault(previous) })
	document := backendDocument(backend.URL, `    decisions:
      - name: answer
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: m}]
entrypoints:
  - model_names: [company/fast]
    recipe: default
  - model_names: [company/public]
    recipe: default
global:
  router:
    skip_processing: {enabled: true}
`)
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(document), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatal(err)
	}
	// The parser does not accept client_auth until every path enforces it.
	cfg.Entrypoints[0].ClientAuth = &config.EntrypointClientAuth{Type: config.ClientAuthBearerAPIKey, APIKeyEnvs: []string{"FAST_CLIENT_KEY"}}
	server, err := extproc.NewServer(path, 0, false, "", routerruntime.NewRegistry(cfg), extproc.WithConfigParts(upstreamPart(config.GatewayStandalone)))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	served := make(chan error, 1)
	go func() { served <- server.StartContextWithoutExtProc(ctx, nil) }()
	defer func() {
		cancel()
		<-served
	}()
	if err = server.WaitForServing(ctx); err != nil {
		t.Fatal(err)
	}
	handler, err := gateway.NewHandler(gateway.Options{Serving: nativeServing{pin: server.Pin, engine: routing.DefaultOptions}, Listener: "http-8899"})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(handler)
	defer front.Close()

	send := func(model string, header map[string]string) (int, string, http.Header) {
		t.Helper()
		body := `{"model":"` + model + `","messages":[{"role":"user","content":"hi"}]}`
		req, err := http.NewRequest(http.MethodPost, front.URL+"/v1/chat/completions", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")
		for name, value := range header {
			req.Header.Set(name, value)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		out, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, string(out), resp.Header
	}

	for _, attempt := range []struct {
		name   string
		header map[string]string
	}{
		{"no key", nil},
		{"another key", map[string]string{"Authorization": "Bearer fast-2"}},
		{"the key as api-key", map[string]string{"Api-Key": "fast-1"}},
		{"skip-processing opt-out", map[string]string{"x-vsr-skip-processing": "true"}},
	} {
		status, body, header := send("company/fast", attempt.header)
		var reply struct {
			Error struct {
				Code string `json:"code"`
			} `json:"error"`
		}
		_ = json.Unmarshal([]byte(body), &reply)
		if status != http.StatusUnauthorized || reply.Error.Code != "invalid_api_key" || header.Get("WWW-Authenticate") == "" {
			t.Fatalf("%s: status %d body %s, want 401 invalid_api_key with a challenge", attempt.name, status, body)
		}
	}
	if got := calls(); len(got) != 0 {
		t.Fatalf("refused requests reached the backend %d times", len(got))
	}

	if status, body, _ := send("company/fast", map[string]string{"Authorization": "Bearer fast-1"}); status != http.StatusOK {
		t.Fatalf("the key: status %d body %s, want 200", status, body)
	}
	if status, body, _ := send("company/public", map[string]string{"Authorization": "Bearer placeholder"}); status != http.StatusOK {
		t.Fatalf("a public entrypoint: status %d body %s, want 200", status, body)
	}
	if got := calls(); len(got) != 2 || got[0] != "" || got[1] != "" {
		t.Fatalf("the backend received Authorization %q, want two calls without the client's", got)
	}
}
