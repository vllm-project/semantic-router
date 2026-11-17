package extproc

import (
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// guardModesConfig routes __guarded__ prompts to a decision whose guard
// blocks the answer to a prompt that also says __refuse__; the guard reads a
// request-stage keyword, so no decision model is needed to decide it.
const guardModesConfig = `
version: v0.5
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    timeout: 300s
providers:
  defaults:
    model: guarded-model
  models:
    - name: guarded-model
      provider_model_id: guarded-model-instruct
      api_format: openai
      backend_refs:
        - name: guarded-primary
          endpoint: 127.0.0.1:18000
          protocol: http
          provider: vllm
recipes:
  - name: default
    signals:
      keywords:
        - name: guarded_route
          operator: OR
          keywords: ["__guarded__"]
        - name: refuse_answer
          operator: OR
          keywords: ["__refuse__"]
    decisions:
      - name: guarded
        priority: 10
        rules:
          operator: AND
          conditions:
            - type: keyword
              name: guarded_route
        modelRefs:
          - model: guarded-model
            use_reasoning: false
        plugins:
          - type: response_guard
            configuration:
              rules:
                - when: {type: keyword, name: refuse_answer}
                  action: block
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
`

// Both gateway modes hold a streamed answer for an enforcing guard: the
// routing engine buffers the body, as Envoy does on the Router's BUFFERED
// mode, whether it drives the ext_proc loop or the standalone sessions. An
// answer without such a guard still streams.
func TestBothGatewayModesHoldAStreamedAnswerForAnEnforcingGuard(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(guardModesConfig))
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	upstream := strings.Join(func() []string {
		var events []string
		for _, part := range streamOf(llmprotocol.OpenAIChatV1, benignAnswer) {
			events = append(events, string(part))
		}
		return events
	}(), "")
	for mode, engineFor := range looperAdapters {
		t.Run(mode, func(t *testing.T) {
			engine := engineFor(newParityRouter(t, cfg))
			for _, tc := range []struct {
				name, prompt string
				held         bool
				blocked      bool
			}{
				{"passed", "__guarded__ how fast is light?", true, false},
				{"blocked", "__guarded__ __refuse__ how fast is light?", true, true},
				{"no guard", "how fast is light?", false, false},
			} {
				t.Run(tc.name, func(t *testing.T) {
					out := serveStreamedAnswer(t, engine, tc.prompt, upstream)
					if held := out.Stream == nil; held != tc.held {
						t.Fatalf("held = %v, want %v", held, tc.held)
					}
					body := drainResponse(t, out)
					if out.Status != 200 || out.Header.Get("content-type") != "text/event-stream" {
						t.Fatalf("reply = %d %q, want a 200 event stream", out.Status, out.Header.Get("content-type"))
					}
					if tc.blocked {
						assertRefusedStream(t, llmprotocol.OpenAIChatV1, body, "Response blocked by the response guard", benignAnswer)
						return
					}
					if !strings.Contains(chatStreamContent(body), benignAnswer) || !strings.HasSuffix(body, "data: [DONE]\n\n") {
						t.Fatalf("client stream = %s, want the whole answer and its terminal event", body)
					}
				})
			}
		})
	}
}

// serveStreamedAnswer plans a streaming Chat request for prompt and answers
// it with upstream, the backend's event stream.
func serveStreamedAnswer(t *testing.T, engine routing.Engine, prompt, upstream string) *routing.Response {
	t.Helper()
	header := routing.Header{
		{Name: ":method", Value: "POST"},
		{Name: ":path", Value: "/v1/chat/completions"},
		{Name: ":authority", Value: "router"},
		{Name: ":scheme", Value: "http"},
		{Name: "content-type", Value: "application/json"},
		{Name: "x-request-id", Value: "guard-modes"},
	}
	body := `{"model":"vllm-sr/auto","stream":true,"messages":[{"role":"user","content":"` + prompt + `"}]}`
	plan, err := engine.Plan(t.Context(), &routing.Request{Header: header, Body: []byte(body)})
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	t.Cleanup(func() { plan.Finish(nil) })
	if plan.Immediate != nil {
		t.Fatalf("the Router answered the request itself: %d %s", plan.Immediate.Status, plan.Immediate.Body)
	}
	out, err := engine.Respond(t.Context(), plan, &routing.UpstreamResponse{
		Status: 200, Header: routing.Header{{Name: "content-type", Value: "text/event-stream"}}, Body: strings.NewReader(upstream),
	})
	if err != nil {
		t.Fatalf("respond: %v", err)
	}
	return out
}

func drainResponse(t *testing.T, out *routing.Response) string {
	t.Helper()
	if out.Stream == nil {
		return string(out.Body)
	}
	var body []byte
	for {
		chunk, err := out.Stream.Next()
		body = append(body, chunk...)
		if errors.Is(err, io.EOF) {
			return string(body)
		}
		if err != nil {
			t.Fatalf("stream: %v", err)
		}
	}
}
