package gatewayparity

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

// Replay is enabled on every decision so a direct or fallback route leaves a
// record whose terminal state we can assert on the wire. Without this, an
// invalid response is still a 502 but no audit record survives to say why.
const replayGlobal = `
global:
  services:
    router_replay:
      store_backend: memory
  plugins:
    - type: router_replay
      configuration:
        enabled: true
`

// replayGatewayOver is gatewayOver with Replay recording enabled. It returns
// the router too, so the test can read the Replay records a request left.
func replayGatewayOver(t *testing.T, configYAML string, backends map[string]http.Handler) (*httptest.Server, *extproc.OpenAIRouter) {
	t.Helper()
	var replacements []string
	for placeholder, handler := range backends {
		server := httptest.NewServer(handler)
		t.Cleanup(server.Close)
		replacements = append(replacements, placeholder, strings.TrimPrefix(server.URL, "http://"))
	}
	configYAML = strings.NewReplacer(replacements...).Replace(configYAML)
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(configPath, []byte(configYAML), 0o600); err != nil {
		t.Fatal(err)
	}
	router, err := extproc.NewOpenAIRouter(configPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = router.Close() })
	set, err := upstream.Build(router.Config, upstream.Options{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = set.Close(ctx)
	})
	opts := routing.DefaultOptions
	opts.ExecutesFallback = true // the native gateway runs the fallback chain
	handler, err := gateway.NewHandler(gateway.Options{
		Serving:  gateway.Static(gateway.Serving{Engine: routing.NewEngine(extproc.NewRouterService(router), opts), Upstream: set}),
		Listener: "http-8899",
	})
	if err != nil {
		t.Fatal(err)
	}
	frontend := httptest.NewServer(handler)
	t.Cleanup(frontend.Close)
	return frontend, router
}

// replayRecords reads every record the router kept, across the per-decision
// recorders and the pre-decision one.
func replayRecords(router *extproc.OpenAIRouter) []routerreplay.RoutingRecord {
	var records []routerreplay.RoutingRecord
	seen := map[*routerreplay.Recorder]bool{}
	collect := func(recorder *routerreplay.Recorder) {
		if recorder == nil || seen[recorder] {
			return
		}
		seen[recorder] = true
		records = append(records, recorder.ListAllRecords()...)
	}
	collect(router.ReplayRecorder)
	for _, recorder := range router.ReplayRecorders {
		collect(recorder)
	}
	return records
}

// TestZeroLengthCompletionFailsReplayOnDirectAndFallbackRoutes drives a real
// response processor through the native gateway. A 200 with an empty body is
// an invalid completion on both the direct and the fallback route: the client
// gets a structured 502 and Replay records a terminal failure, not the abort
// a stream that ended without a terminal response would leave behind.
func TestZeroLengthCompletionFailsReplayOnDirectAndFallbackRoutes(t *testing.T) {
	const emptyBody = ""
	cases := []struct {
		name     string
		backends map[string]http.Handler
	}{
		{
			name: "direct",
			backends: map[string]http.Handler{
				"PRIMARY": &scriptedBackend{status: http.StatusOK, body: emptyBody},
			},
		},
		{
			name: "fallback",
			backends: map[string]http.Handler{
				"PRIMARY":   &scriptedBackend{status: http.StatusServiceUnavailable, body: openAIError("primary overloaded")},
				"SECONDARY": &scriptedBackend{status: http.StatusOK, body: emptyBody},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			frontend, router := replayGatewayOver(t, fallbackConfig+replayGlobal, tc.backends)
			got := postChat(t, frontend)

			if got.status != http.StatusBadGateway {
				t.Fatalf("status = %d, want 502: %s", got.status, got)
			}
			if got.header["content-type"] != "application/json" {
				t.Fatalf("content-type = %q, want application/json: %s", got.header["content-type"], got)
			}
			var envelope struct {
				Error struct {
					Type    string          `json:"type"`
					Code    json.RawMessage `json:"code"`
					Message string          `json:"message"`
				} `json:"error"`
			}
			if err := json.Unmarshal([]byte(got.body), &envelope); err != nil {
				t.Fatalf("the client error is not JSON: %v: %s", err, got.body)
			}
			if envelope.Error.Type == "" || len(envelope.Error.Code) == 0 || envelope.Error.Message == "" {
				t.Fatalf("client error is not structured: %s", got.body)
			}

			records := replayRecords(router)
			if len(records) != 1 {
				t.Fatalf("Replay records = %d, want 1: %+v", len(records), records)
			}
			record := records[0]
			if record.LifecycleState != routerreplay.LifecycleFailed || record.TerminalReason != "upstream_invalid_response" {
				t.Fatalf("Replay terminal = %q/%q, want %q/%q",
					record.LifecycleState, record.TerminalReason,
					routerreplay.LifecycleFailed, "upstream_invalid_response")
			}
			if record.ResponseStatus != http.StatusBadGateway {
				t.Fatalf("Replay response status = %d, want 502", record.ResponseStatus)
			}
		})
	}
}
