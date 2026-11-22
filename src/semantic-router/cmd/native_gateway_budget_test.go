//go:build !windows

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

// An unpriced call reaches the provider unless a budget covers it. Exercise
// canonical loading, listener identity, routing, refusal and real HTTP dispatch.
func TestStandaloneSpendBudgetScope(t *testing.T) {
	t.Setenv(configsnapshot.HistoryDirEnv, t.TempDir())
	previous := modelservice.DefaultManager()
	modelservice.SetDefault(nil)
	t.Cleanup(func() { modelservice.SetDefault(previous) })
	type budgetCall struct {
		user, groups string
		status       int
	}
	for _, tc := range []struct {
		name    string
		budgets string
		calls   []budgetCall
	}{
		{
			name: "user only", budgets: "users: [{name: alice-daily, match: {user: alice}, limit: 2, window: day}]",
			calls: []budgetCall{{"bob", "", http.StatusOK}, {"alice", "", http.StatusForbidden}, {"bob", "", http.StatusOK}},
		},
		{
			name: "group only", budgets: "users: [{name: engineering-daily, match: {group: engineering}, limit: 2, window: day}]",
			calls: []budgetCall{{"bob", "support", http.StatusOK}, {"bob", "support, engineering", http.StatusForbidden}},
		},
		{name: "run covers every request", budgets: "run: {limit: 2}", calls: []budgetCall{{"bob", "", http.StatusForbidden}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var dispatched atomic.Int32
			backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				dispatched.Add(1)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"id":"c","object":"chat.completion","model":"unpriced","choices":[{"index":0,"message":{"role":"assistant","content":"from backend"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
			}))
			defer backend.Close()
			document := fmt.Sprintf(`version: v0.5
listeners:
  - name: http-8899
    address: 127.0.0.1
    port: 8899
    identity:
      trust_headers: true
      trusted_peers: [127.0.0.0/8]
providers:
  defaults:
    model: unpriced
  models:
    - name: unpriced
      backend_refs: [{endpoint: %s, protocol: http, provider: vllm}]
global:
  stores:
    response_cache: {}
  services:
    budgets:
      %s
recipes:
  - name: default
`, strings.TrimPrefix(backend.URL, "http://"), tc.budgets)
			cfg, err := config.ParseYAMLBytes([]byte(document))
			if err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(t.TempDir(), "config.yaml")
			if err = os.WriteFile(path, []byte(document), 0o600); err != nil {
				t.Fatal(err)
			}
			server, err := extproc.NewServer(path, 0, false, "", routerruntime.NewRegistry(cfg),
				extproc.WithGatewayMode(config.GatewayStandalone), extproc.WithConfigParts(upstreamPart(config.GatewayStandalone)))
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			served := make(chan error, 1)
			go func() { served <- server.StartContextWithoutExtProc(ctx, nil) }()
			defer func() {
				cancel()
				if runErr := <-served; runErr != nil {
					t.Errorf("routing core shutdown: %v", runErr)
				}
			}()
			if err = server.WaitForServing(ctx); err != nil {
				t.Fatal(err)
			}
			handler, err := gateway.NewHandler(gateway.Options{Serving: nativeServing{pin: server.Pin}, Listener: "http-8899"})
			if err != nil {
				t.Fatal(err)
			}
			front := httptest.NewServer(handler)
			defer front.Close()
			client := front.Client()
			client.Timeout = 5 * time.Second
			for _, call := range tc.calls {
				before := dispatched.Load()
				request, err := http.NewRequest(http.MethodPost, front.URL+"/v1/chat/completions",
					strings.NewReader(`{"model":"unpriced","messages":[{"role":"user","content":"hello"}]}`))
				if err != nil {
					t.Fatal(err)
				}
				request.Header.Set("Content-Type", "application/json")
				request.Header.Set(headers.AuthzUserID, call.user)
				request.Header.Set(headers.AuthzUserGroups, call.groups)
				response, err := client.Do(request)
				if err != nil {
					t.Fatal(err)
				}
				body, readErr := io.ReadAll(response.Body)
				response.Body.Close()
				if readErr != nil {
					t.Fatal(readErr)
				}
				if response.StatusCode != call.status {
					t.Fatalf("user %q groups %q: status %d, want %d: %s", call.user, call.groups, response.StatusCode, call.status, body)
				}
				if call.status == http.StatusOK {
					if dispatched.Load() != before+1 || !strings.Contains(string(body), "from backend") {
						t.Fatalf("uncovered call did not reach the provider: calls %d -> %d, body %s", before, dispatched.Load(), body)
					}
					if response.Header.Get(headers.VSRBudgetScope) != "" || response.Header.Get(headers.VSRBudgetRemaining) != "" {
						t.Fatalf("uncovered call reports budget headers: %v", response.Header)
					}
				} else {
					var refused struct {
						Error struct{ Code string } `json:"error"`
					}
					if err := json.Unmarshal(body, &refused); err != nil || refused.Error.Code != "budget_unpriced" || dispatched.Load() != before {
						t.Fatalf("covered unpriced call: calls %d -> %d, body %s (%v)", before, dispatched.Load(), body, err)
					}
				}
			}
		})
	}
}
