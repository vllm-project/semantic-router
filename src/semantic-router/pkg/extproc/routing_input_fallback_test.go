package extproc

import (
	"context"
	"testing"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/parity"
)

// routingInputFallbackConfig leaves a harness's status turns out of the
// routing text and routes every request to one model.
const routingInputFallbackConfig = `
version: v0.5
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    timeout: 30s
providers:
  models:
    - name: gpt-oss
      backend_refs:
        - name: gpt-oss-primary
          endpoint: 127.0.0.1:18000
          protocol: http
          provider: vllm
recipes:
  - name: default
    decisions:
      - name: default_route
        priority: 10
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: gpt-oss
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
global:
  router:
    routing_input:
      skip_user_messages_matching: ['^\[harness status\]']
`

// A request whose every user message matches routing_input routes on its
// last user message, in both modes, and the Router logs that it did: one of
// user messages alone was refused with no_route, and one with an assistant
// turn routed on no text at all. A request with a message routing reads
// routes on it as before, without that log line.
func TestRoutingInputRoutesARequestWhoseEveryUserMessageMatches(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(routingInputFallbackConfig))
	if err != nil {
		t.Fatal(err)
	}
	viaSession := parity.NewRecorder(newParityRouter(t, cfg), routing.DefaultOptions)
	viaExtProc := parity.NewRecorder(&extprocStreamProcessor{router: newParityRouter(t, cfg)}, routing.DefaultOptions)
	for _, c := range []struct {
		name     string
		body     string
		fallback bool
	}{
		{
			name: "only-matching-user-messages",
			body: `{"model":"vllm-sr/auto","messages":[{"role":"user","content":"[harness status] tests passed"},` +
				`{"role":"user","content":"[harness status] lint clean"}]}`,
			fallback: true,
		},
		{
			name: "matching-user-messages-and-an-assistant-turn",
			body: `{"model":"vllm-sr/auto","messages":[{"role":"user","content":"[harness status] tests passed"},` +
				`{"role":"assistant","content":"Noted."},{"role":"user","content":"[harness status] lint clean"}]}`,
			fallback: true,
		},
		{
			name: "a-question-and-a-status-turn",
			body: `{"model":"vllm-sr/auto","messages":[{"role":"user","content":"Explain the release plan."},` +
				`{"role":"user","content":"[harness status] lint clean"}]}`,
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			core, logs := observer.New(zapcore.InfoLevel)
			t.Cleanup(zap.ReplaceGlobals(zap.New(core)))
			request := parity.Case{
				Name: c.name,
				Request: parity.CaseRequest{
					Method: "POST", Path: "/v1/chat/completions", Headers: [][]string{{"content-type", "application/json"}}, Body: c.body,
				},
				Upstream: &parity.Upstream{Status: 200, Headers: [][]string{{"content-type", "application/json"}}, Body: `{"id":"chatcmpl-1",` +
					`"object":"chat.completion","created":1,"model":"gpt-oss","choices":[{"index":0,"message":{"role":"assistant",` +
					`"content":"Done."},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`},
			}
			for mode, recorder := range map[string]*parity.Recorder{"session": viaSession, "ext_proc": viaExtProc} {
				got := recorder.Run(context.Background(), request)
				if got.Error != "" {
					t.Fatalf("%s: run error %q", mode, got.Error)
				}
				if got.Upstream == nil {
					status, body := 0, ""
					if got.Response != nil {
						status, body = got.Response.Status, string(got.Response.Body)
					}
					t.Fatalf("%s: the request did not reach the model: %d %s", mode, status, body)
				}
			}
			if logged := len(logs.FilterField(zap.String("event", "routing_input_fallback")).All()) > 0; logged != c.fallback {
				t.Fatalf("routing_input_fallback logged = %v, want %v", logged, c.fallback)
			}
		})
	}
}
