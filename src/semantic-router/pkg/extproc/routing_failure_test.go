package extproc

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/parity"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

var routingFailures = []routingFailure{
	routingFailureModelNotFound,
	routingFailureNoRoute,
	routingFailureContextLength,
	routingFailureDecisionUnresolved,
	routingFailureNoEligibleModel,
}

// flowAliasCaptureConfig has the shape that broke the response-api-redis
// profile (#4651): its only backend model is also a Flow alias, so a request
// for the model evaluates only the workflows decision. The model's small
// context window lets an auto request outgrow it.
const flowAliasCaptureConfig = `
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    timeout: 30s
providers:
  defaults:
    model: gpt-oss
  models:
    - name: gpt-oss
      backend_refs:
        - name: gpt-oss-primary
          endpoint: 127.0.0.1:18000
          protocol: http
          provider: vllm
routing:
  modelCards:
    - name: gpt-oss
      context_window_size: 64
  signals:
    keywords:
      - name: plan_keywords
        operator: OR
        keywords: ["plan"]
  decisions:
    - name: workflow_route
      priority: 20
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: plan_keywords
      modelRefs:
        - model: gpt-oss
      algorithm:
        type: workflows
        workflows:
          mode: static
          roles:
            - name: worker
              models: [gpt-oss]
    - name: default_route
      priority: 10
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: gpt-oss
global:
  integrations:
    looper:
      flow:
        model_names: [gpt-oss]
        state:
          store_backend: memory
`

type routingFailureCase struct {
	name    string
	path    string
	body    string
	failure routingFailure
	// event is the log line that keeps the Router's own reason.
	event string
}

func routingFailureCases() []routingFailureCase {
	return []routingFailureCase{
		{
			name:    "flow-alias-captures-the-model",
			path:    "/v1/chat/completions",
			body:    `{"model":"gpt-oss","messages":[{"role":"user","content":"Hello there."}]}`,
			failure: routingFailureNoRoute,
			event:   "entrypoint_routing_no_selection",
		},
		{
			name:    "flow-alias-captures-the-model-messages",
			path:    "/v1/messages",
			body:    `{"model":"gpt-oss","max_tokens":16,"messages":[{"role":"user","content":"Hello there."}]}`,
			failure: routingFailureNoRoute,
			event:   "entrypoint_routing_no_selection",
		},
		{
			name:    "unknown-model",
			path:    "/v1/chat/completions",
			body:    `{"model":"no-such-model","messages":[{"role":"user","content":"Hello there."}]}`,
			failure: routingFailureModelNotFound,
			event:   "specified_model_not_found",
		},
		{
			name: "request-outgrows-every-context-window",
			path: "/v1/chat/completions",
			body: `{"model":"vllm-sr/auto","messages":[{"role":"user","content":"` +
				strings.Repeat("Summarize these meeting notes carefully. ", 40) + `"}]}`,
			failure: routingFailureContextLength,
			event:   "decision_context_ineligible",
		},
	}
}

func (c routingFailureCase) parityCase() parity.Case {
	headers := [][]string{{"content-type", "application/json"}}
	if c.path == "/v1/messages" {
		headers = append(headers, []string{"anthropic-version", "2023-06-01"})
	}
	return parity.Case{
		Name:    c.name,
		Request: parity.CaseRequest{Method: "POST", Path: c.path, Headers: headers, Body: c.body},
	}
}

// TestRoutingFailuresReturnTheirReasonCodeInBothModes runs each failure
// through the routing session, as standalone mode does, and the ext_proc
// adapter, as Envoy does. Both must answer with the same status and body,
// carrying the failure's code and message and nothing about the request,
// while the log keeps the Router's reason under the request id.
func TestRoutingFailuresReturnTheirReasonCodeInBothModes(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(flowAliasCaptureConfig))
	if err != nil {
		t.Fatal(err)
	}
	viaSession := parity.NewRecorder(newParityRouter(t, cfg), routing.DefaultOptions)
	viaExtProc := parity.NewRecorder(&extprocStreamProcessor{router: newParityRouter(t, cfg)}, routing.DefaultOptions)
	for _, c := range routingFailureCases() {
		t.Run(c.name, func(t *testing.T) {
			core, logs := observer.New(zapcore.WarnLevel)
			t.Cleanup(zap.ReplaceGlobals(zap.New(core)))

			got := viaSession.Run(context.Background(), c.parityCase())
			want := viaExtProc.Run(context.Background(), c.parityCase())
			if got.Error != "" || want.Error != "" {
				t.Fatalf("run errors: session %q, ext_proc %q", got.Error, want.Error)
			}
			if got.Upstream != nil {
				t.Fatalf("an unroutable request reached a backend: %v", got.Upstream.Header)
			}
			got.Evidence = routing.Evidence{}
			if diff := parity.Diff(want, got); diff != "" {
				t.Fatalf("the modes answer differently:\n%s", diff)
			}
			assertRoutingFailureBody(t, c, got.Response)
			assertRoutingFailureLogged(t, logs, c)
		})
	}
}

func assertRoutingFailureBody(t *testing.T, c routingFailureCase, response *parity.Message) {
	t.Helper()
	if response == nil || response.Status != c.failure.status {
		t.Fatalf("response = %+v, want status %d", response, c.failure.status)
	}
	var body struct {
		Type  string `json:"type"`
		Error struct {
			Type    string  `json:"type"`
			Code    *string `json:"code"`
			Message string  `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(response.Body, &body); err != nil {
		t.Fatalf("the error body is not JSON: %v: %s", err, response.Body)
	}
	if body.Error.Type != "invalid_request_error" || body.Error.Message != c.failure.message {
		t.Fatalf("error = %+v, want an invalid_request_error saying %q", body.Error, c.failure.message)
	}
	if c.path == "/v1/messages" {
		// Anthropic's error envelope has no code field.
		if body.Type != "error" || body.Error.Code != nil {
			t.Fatalf("not an Anthropic error envelope: %s", response.Body)
		}
		return
	}
	if body.Error.Code == nil || *body.Error.Code != c.failure.code {
		t.Fatalf("error code = %v, want %q: %s", body.Error.Code, c.failure.code, response.Body)
	}
}

func assertRoutingFailureLogged(t *testing.T, logs *observer.ObservedLogs, c routingFailureCase) {
	t.Helper()
	for _, entry := range logs.All() {
		fields := entry.ContextMap()
		if fields["event"] != c.event {
			continue
		}
		if fields["request_id"] != "parity-"+c.name || fields["code"] != c.failure.code {
			t.Fatalf("%s does not carry the request id and code: %v", c.event, fields)
		}
		if c.failure == routingFailureNoRoute &&
			(fields["model"] != "gpt-oss" || fields["looper_algorithm"] != config.DecisionAlgorithmWorkflows) {
			t.Fatalf("%s does not say the Flow alias captured the model: %v", c.event, fields)
		}
		return
	}
	t.Fatalf("no %s log line", c.event)
}

func TestRoutingRejectionsReturnTheirReasonCode(t *testing.T) {
	tests := []struct {
		name    string
		reject  func(*OpenAIRouter, *RequestContext) *ext_proc.ProcessingResponse
		failure routingFailure
		reason  string
	}{
		{
			name: "decision unresolved",
			reject: func(r *OpenAIRouter, ctx *RequestContext) *ext_proc.ProcessingResponse {
				return r.respondDecisionUnresolved(ctx, "entrypoint-model", &decision.DecisionUnresolvedError{Decision: "guarded"})
			},
			failure: routingFailureDecisionUnresolved,
			reason:  "decision_unresolved",
		},
		{
			name: "selection rejected every candidate",
			reject: func(r *OpenAIRouter, ctx *RequestContext) *ext_proc.ProcessingResponse {
				return r.respondSelectionRejected(ctx, "entrypoint-model", selection.ErrNoEligibleCandidates)
			},
			failure: routingFailureNoEligibleModel,
			reason:  "selection_rejected",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
			router := &OpenAIRouter{ReplayRecorder: recorder}
			replayConfig := config.DefaultRouterReplayPluginConfig()
			replayConfig.Enabled = true
			replayConfig.CaptureResponseBody = true
			ctx := &RequestContext{
				RequestID: "rejected-request", SourceFormat: llmprotocol.OpenAIChatV1,
				SemanticRequest: testNeutralRequest("entrypoint-model", "hello"), RouterReplayPluginConfig: &replayConfig,
			}

			immediate := test.reject(router, ctx).GetImmediateResponse()
			if immediate == nil || int(immediate.GetStatus().GetCode()) != test.failure.status {
				t.Fatalf("response = %+v, want HTTP %d", immediate, test.failure.status)
			}
			want := `{"error":{"type":"server_error","code":"` + test.failure.code + `","message":"` + test.failure.message + `","param":null}}`
			if string(immediate.GetBody()) != want {
				t.Fatalf("body = %s, want %s", immediate.GetBody(), want)
			}
			record, ok := recorder.GetRecord(ctx.RouterReplayID)
			if !ok || record.ResponseStatus != test.failure.status || record.TerminalReason != test.reason || record.ResponseBody != want {
				t.Fatalf("Replay does not keep what the client received: %+v", record)
			}
		})
	}
}

func TestRoutingFailureCodesAreDocumented(t *testing.T) {
	path := filepath.Join("..", "..", "..", "..", "website", "docs", "api", "router.md")
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, failure := range routingFailures {
		row := documentedRow(string(content), failure.code)
		if row == "" {
			t.Errorf("%s does not document the routing failure code %q", path, failure.code)
			continue
		}
		if !strings.Contains(row, strconv.Itoa(failure.status)) {
			t.Errorf("%s documents %q without its status %d: %s", path, failure.code, failure.status, row)
		}
	}
}

// documentedRow is the table row that starts with code.
func documentedRow(doc, code string) string {
	for _, line := range strings.Split(doc, "\n") {
		if strings.HasPrefix(line, "| `"+code+"` |") {
			return line
		}
	}
	return ""
}
