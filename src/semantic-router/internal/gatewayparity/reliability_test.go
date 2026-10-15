package gatewayparity

import (
	"io"
	"net/http"
	"testing"
	"time"
)

// A decision whose reliability block cuts the provider model's five-minute
// route timeout to 200ms. Envoy mode sends the override as
// x-envoy-upstream-rq-timeout-ms (covered by the Kind profile); the native
// gateway applies the same override from the planned call.
const decisionTimeoutConfig = `
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    timeout: 300s
providers:
  defaults:
    model: slow-model
  models:
    - name: slow-model
      provider_model_id: slow-model
      api_format: openai
      backend_refs:
        - name: slow
          endpoint: SLOW
          protocol: http
          provider: vllm
routing:
  modelCards:
    - name: slow-model
  decisions:
    - name: bounded_route
      priority: 10
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: slow-model
          use_reasoning: false
      reliability:
        total_timeout: 200ms
`

func TestNativeGatewayAppliesTheDecisionsTotalTimeout(t *testing.T) {
	slow := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		select {
		case <-r.Context().Done():
		case <-time.After(5 * time.Second):
		}
	})
	frontend := gatewayOver(t, decisionTimeoutConfig, map[string]http.Handler{"SLOW": slow}, true)

	started := time.Now()
	got := postChat(t, frontend)
	elapsed := time.Since(started)
	if got.status != http.StatusGatewayTimeout {
		t.Fatalf("response = %s, want Envoy's 504 for a route timeout", got)
	}
	if elapsed < 200*time.Millisecond || elapsed > 2*time.Second {
		t.Fatalf("answered after %v, want the decision's 200ms rather than the route's 300s", elapsed)
	}
}
