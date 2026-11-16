package extproc

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

// servedModelConfig routes vllm-sr/auto by keyword, with fallback on for the
// failover and steady decisions, and vllm-sr/cc as a Claude Code gateway whose
// housekeeping rule sends compactions and side requests to qwen-small.
// Protection holds every session key, and Replay records every request.
const servedModelConfig = `
version: v0.5
listeners:
  - {name: http-8899, address: 0.0.0.0, port: 8899, timeout: 30s}
providers:
  defaults: {model: main-model}
  models:
    - name: main-model
      provider_model_id: main-model
      api_format: openai
      pricing: {currency: USD, prompt_per_1m: 3.0, cached_input_per_1m: 0.3, cache_write_per_1m: 3.75, completion_per_1m: 15.0}
      backend_refs: [{name: fake-main, endpoint: BACKEND, protocol: http, provider: vllm}]
    - name: cheap-model
      provider_model_id: cheap-model
      api_format: openai
      pricing: {currency: USD, prompt_per_1m: 1.0, cached_input_per_1m: 0.1, cache_write_per_1m: 1.25, completion_per_1m: 5.0}
      backend_refs: [{name: fake-cheap, endpoint: BACKEND, protocol: http, provider: vllm}]
    - name: fail-model
      provider_model_id: fail-model
      api_format: openai
      backend_refs: [{name: fake-fail, endpoint: BACKEND, protocol: http, provider: vllm}]
    - name: think-model
      provider_model_id: think-model
      api_format: openai
      reasoning: {family: qwen3}
      backend_refs: [{name: fake-think, endpoint: BACKEND, protocol: http, provider: vllm}]
    - name: qwen-msg
      provider_model_id: qwen-msg
      api_format: messages
      reasoning: {family: claude-effort-adaptive}
      backend_refs: [{name: fake-msg, endpoint: BACKEND, protocol: http, provider: anthropic}]
    - name: qwen-small
      provider_model_id: qwen-small
      api_format: messages
      reasoning: {family: claude-effort-adaptive}
      backend_refs: [{name: fake-small, endpoint: BACKEND, protocol: http, provider: anthropic}]
    - name: fail-msg
      provider_model_id: fail-model
      api_format: messages
      backend_refs: [{name: fake-fail-msg, endpoint: BACKEND, protocol: http, provider: anthropic}]
    - name: plain-msg
      provider_model_id: plain-msg
      api_format: messages
      pricing: {currency: USD, prompt_per_1m: 1.0, cached_input_per_1m: 0.1, cache_write_per_1m: 1.25, completion_per_1m: 5.0}
      backend_refs: [{name: fake-plain-msg, endpoint: BACKEND, protocol: http, provider: anthropic}]
    - name: empty-msg
      provider_model_id: empty-model
      api_format: messages
      pricing: {currency: USD, prompt_per_1m: 1.0, cached_input_per_1m: 0.1, cache_write_per_1m: 1.25, completion_per_1m: 5.0}
      backend_refs: [{name: fake-empty-msg, endpoint: BACKEND, protocol: http, provider: anthropic}]
    - name: cut-msg
      provider_model_id: cut-model
      api_format: messages
      pricing: {currency: USD, prompt_per_1m: 1.0, cached_input_per_1m: 0.1, cache_write_per_1m: 1.25, completion_per_1m: 5.0}
      backend_refs: [{name: fake-cut-msg, endpoint: BACKEND, protocol: http, provider: anthropic}]
    - name: slow-msg
      provider_model_id: slow-model
      api_format: messages
      pricing: {currency: USD, prompt_per_1m: 1.0, cached_input_per_1m: 0.1, cache_write_per_1m: 1.25, completion_per_1m: 5.0}
      backend_refs: [{name: fake-slow-msg, endpoint: BACKEND, protocol: http, provider: anthropic}]
entrypoints:
  - {model_names: [vllm-sr/auto], recipe: agent, default: true}
  - {model_names: [vllm-sr/cc], recipe: cc}
recipes:
  - name: agent
    signals:
      keywords:
        - {name: failover, operator: OR, keywords: ["FAILOVER"], case_sensitive: true}
        - {name: steady, operator: OR, keywords: ["STEADY"], case_sensitive: true}
        - {name: nothink, operator: OR, keywords: ["NOTHINK"], case_sensitive: true}
        - {name: think, operator: OR, keywords: ["THINK"], case_sensitive: true}
        - {name: familyless, operator: OR, keywords: ["FAMILYLESS"], case_sensitive: true}
        - {name: blockme, operator: OR, keywords: ["BLOCKME"], case_sensitive: true}
        - {name: handoff, operator: OR, keywords: ["HANDOFFMSG"], case_sensitive: true}
        - {name: plainmsg, operator: OR, keywords: ["PLAINMSG"], case_sensitive: true}
        - {name: silent, operator: OR, keywords: ["SILENTANSWER"], case_sensitive: true}
        - {name: slow, operator: OR, keywords: ["SLOWSTREAM"], case_sensitive: true}
        - {name: cut, operator: OR, keywords: ["CUTSTREAM"], case_sensitive: true}
    decisions:
      - name: block
        priority: 300
        rules: {operator: AND, conditions: [{type: keyword, name: blockme}]}
        plugins: [{type: fast_response, configuration: {message: Blocked., block: true}}]
      - name: failover
        priority: 200
        rules: {operator: AND, conditions: [{type: keyword, name: failover}]}
        modelRefs: [{model: fail-model}, {model: cheap-model}]
        fallback: {enabled: true, max_attempts: 2}
      - name: steady
        priority: 200
        rules: {operator: AND, conditions: [{type: keyword, name: steady}]}
        modelRefs: [{model: cheap-model}, {model: main-model}]
        fallback: {enabled: true, max_attempts: 2}
      - name: think
        priority: 200
        rules: {operator: AND, conditions: [{type: keyword, name: think}]}
        modelRefs: [{model: think-model, use_reasoning: true}]
      - name: familyless
        priority: 200
        rules: {operator: AND, conditions: [{type: keyword, name: familyless}]}
        modelRefs: [{model: main-model, use_reasoning: true, reasoning_effort: high}]
      - name: handoff
        priority: 200
        rules: {operator: AND, conditions: [{type: keyword, name: handoff}]}
        modelRefs: [{model: fail-msg}, {model: plain-msg}]
        fallback: {enabled: true, max_attempts: 2}
      - name: plainmsg
        priority: 200
        rules: {operator: AND, conditions: [{type: keyword, name: plainmsg}]}
        modelRefs: [{model: plain-msg}]
      - name: silent
        priority: 200
        rules: {operator: AND, conditions: [{type: keyword, name: silent}]}
        modelRefs: [{model: empty-msg}]
      - name: slow
        priority: 200
        rules: {operator: AND, conditions: [{type: keyword, name: slow}]}
        modelRefs: [{model: slow-msg}]
      - name: cut
        priority: 200
        rules: {operator: AND, conditions: [{type: keyword, name: cut}]}
        modelRefs: [{model: cut-msg}]
      - name: nothink
        priority: 200
        rules: {operator: AND, conditions: [{type: keyword, name: nothink}]}
        modelRefs: [{model: think-model}]
      - name: default
        priority: 1
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: main-model}, {model: cheap-model}]
  - name: cc
    signals:
      agent:
        - {name: housekeeping, description: Compactions and side requests., request_classes: [compaction, auxiliary]}
    decisions:
      - name: housekeeping
        priority: 150
        rules: {type: agent, name: housekeeping}
        modelRefs: [{model: qwen-small, use_reasoning: true, reasoning_effort: low}]
      - name: default
        priority: 1
        rules: {operator: AND, conditions: []}
        modelRefs:
          - {model: qwen-msg, use_reasoning: true, reasoning_effort: medium}
          - {model: qwen-small, use_reasoning: true, reasoning_effort: low}
global:
  router:
    learning:
      enabled: true
      adaptation: {enabled: false}
      protection: {enabled: true, scope: session}
  services:
    router_replay: {store_backend: memory}
  plugins:
    - type: router_replay
      configuration: {enabled: true}
`

// servedModelBackend answers Chat Completions and Messages: fail-model with
// 503, every other model with usage that reads and writes the prompt cache.
type servedModelBackend struct {
	mu    sync.Mutex
	calls []map[string]any
}

func (b *servedModelBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)
	var body map[string]any
	_ = json.Unmarshal(raw, &body)
	b.mu.Lock()
	b.calls = append(b.calls, body)
	b.mu.Unlock()
	model, _ := body["model"].(string)
	switch model {
	case "empty-model":
		writeEmptyAnthropicAnswer(w, body["stream"] == true)
		return
	case "slow-model", "cut-model":
		writeStalledAnthropicStream(w, r, model == "cut-model")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if model == "fail-model" {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":{"message":"overloaded","type":"server_error"}}`))
		return
	}
	if strings.HasSuffix(r.URL.Path, "/messages") {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "msg-1", "type": "message", "role": "assistant", "model": model, "stop_reason": "end_turn",
			"content": []any{map[string]any{"type": "text", "text": "ok from " + model}},
			"usage":   map[string]any{"input_tokens": 200, "output_tokens": 100, "cache_read_input_tokens": 1500, "cache_creation_input_tokens": 300},
		})
		return
	}
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id": "chat-1", "object": "chat.completion", "created": 1, "model": model,
		"choices": []any{map[string]any{"index": 0, "finish_reason": "stop", "message": map[string]any{"role": "assistant", "content": "ok from " + model}}},
		"usage": map[string]any{
			"prompt_tokens": 2000, "completion_tokens": 100, "total_tokens": 2100,
			"prompt_tokens_details": map[string]any{"cached_tokens": 1500, "cache_write_tokens": 300},
		},
	})
}

// takeCalls returns the bodies the backend got since the last call.
func (b *servedModelBackend) takeCalls() []map[string]any {
	b.mu.Lock()
	defer b.mu.Unlock()
	calls := b.calls
	b.calls = nil
	return calls
}

type servedModelHarness struct {
	router  *OpenAIRouter
	engine  routing.Engine
	set     *upstream.Set
	backend *servedModelBackend
}

// newServedModelHarness serves servedModelConfig, after any edits, against a
// fake backend.
func newServedModelHarness(t *testing.T, edits ...func(string) string) *servedModelHarness {
	t.Helper()
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
	backend := &servedModelBackend{}
	server := httptest.NewServer(backend)
	t.Cleanup(server.Close)
	document := strings.ReplaceAll(servedModelConfig, "BACKEND", strings.TrimPrefix(server.URL, "http://"))
	for _, edit := range edits {
		document = edit(document)
	}
	cfg, err := config.ParseYAMLBytes([]byte(document))
	require.NoError(t, err)
	router := newLooperRouter(t, cfg)
	opts := routing.DefaultOptions
	opts.ExecutesFallback = true
	return &servedModelHarness{router: router, engine: routing.NewEngine(router, opts), set: router.upstreamSet(), backend: backend}
}

// servedCall is one request sent through the standalone gateway's path, with
// the Replay record it left and the models its upstream calls named.
type servedCall struct {
	status int
	body   []byte
	record routerreplay.RoutingRecord
	called []string
	bodies []map[string]any
}

// send serves one request the way the standalone gateway does: plan it, send
// the planned call with its fallback, run the response phases, and end it.
func (h *servedModelHarness) send(t *testing.T, path, body string, header ...routing.HeaderField) servedCall {
	t.Helper()
	request := append(routing.Header{
		{Name: ":method", Value: "POST"},
		{Name: ":path", Value: path},
		{Name: ":authority", Value: "router"},
		{Name: ":scheme", Value: "http"},
		{Name: "content-type", Value: "application/json"},
		{Name: "anthropic-version", Value: "2023-06-01"},
	}, header...)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	plan, err := h.engine.Plan(ctx, &routing.Request{Header: request, Body: []byte(body)})
	require.NoError(t, err)
	answer := plan.Immediate
	if answer == nil {
		result, err := h.set.Execute(ctx, plan.Call, looperListener)
		require.NoError(t, err)
		require.NotNil(t, result.Response)
		answer, err = h.engine.Respond(ctx, plan, &routing.UpstreamResponse{
			Status: result.Response.StatusCode, Header: routingHeaderOf(result.Response.Header), Body: result.Response.Body,
		})
		require.NoError(t, err)
		for answer.Stream != nil {
			chunk, err := answer.Stream.Next()
			answer.Body = append(answer.Body, chunk...)
			if err != nil {
				answer.Stream = nil
			}
		}
		_ = result.Response.Body.Close()
	}
	plan.Finish(nil)
	call := servedCall{status: answer.Status, body: answer.Body, bodies: h.backend.takeCalls()}
	for _, upstreamBody := range call.bodies {
		model, _ := upstreamBody["model"].(string)
		call.called = append(call.called, model)
	}
	id := answer.Header.Get("x-vsr-replay-id")
	require.NotEmpty(t, id, "the request left no Replay record")
	record, found := h.router.findRouterReplayRecord(id)
	require.True(t, found, "record %s", id)
	call.record = record
	return call
}

func chatBody(text string, extra string) string {
	return `{"model":"vllm-sr/auto","max_tokens":50` + extra + `,"messages":[{"role":"user","content":"` + text + `"}]}`
}

func claudeCodeHeaders(session, class string) []routing.HeaderField {
	return []routing.HeaderField{{Name: "x-claude-code-session-id", Value: session}, {Name: "x-claude-code-request-class", Value: class}}
}

func fallbackOutcomes(record routerreplay.RoutingRecord) []routerreplay.Outcome {
	var out []routerreplay.Outcome
	for _, outcome := range record.Outcomes {
		if outcome.Source == fallbackOutcomeSource {
			out = append(out, outcome)
		}
	}
	return out
}

func sessionPolicyString(record routerreplay.RoutingRecord, key string) string {
	value, _ := record.SessionPolicy[key].(string)
	return value
}

// After a fallback candidate answers, the record names it and keeps the failed
// attempt in its fallback outcome, and protection holds the session on it: the
// next request in that decision is not sent to the failing model again, and a
// request whose decision admits the candidate stays on its warm cache.
func TestFallbackAnswerOwnsTheRecordAndTheSession(t *testing.T) {
	h := newServedModelHarness(t)
	session := routing.HeaderField{Name: "x-claude-code-session-id", Value: "served-run"}
	first := h.send(t, "/v1/chat/completions", chatBody("plain question", ""), session)
	require.Equal(t, []string{"main-model"}, first.called)

	failover := h.send(t, "/v1/chat/completions", chatBody("FAILOVER question", ""), session)
	require.Equal(t, http.StatusOK, failover.status)
	require.Equal(t, []string{"fail-model", "cheap-model"}, failover.called)
	require.Equal(t, "cheap-model", failover.record.SelectedModel)
	outcomes := fallbackOutcomes(failover.record)
	require.Len(t, outcomes, 1)
	require.Equal(t, "completed", outcomes[0].Verdict)
	require.Equal(t, "fail-model", outcomes[0].Metadata["initial_model"])

	again := h.send(t, "/v1/chat/completions", chatBody("FAILOVER again", ""), session)
	require.Equal(t, []string{"cheap-model"}, again.called, "the session was sent to the failing model again")
	require.Equal(t, "cheap-model", again.record.SelectedModel)
	require.Equal(t, "cheap-model", sessionPolicyString(again.record, "current_model"))

	plain := h.send(t, "/v1/chat/completions", chatBody("plain again", ""), session)
	require.Equal(t, []string{"cheap-model"}, plain.called, "the session left the model that holds its cache")
	require.Equal(t, "session_scope_protect", sessionPolicyString(plain.record, "reason"))
}

// A request its first model answers has no fallback to report, whichever
// protocol the client speaks.
func TestFirstAnswerOnAFallbackRouteRecordsNoFallback(t *testing.T) {
	h := newServedModelHarness(t)
	for _, path := range []string{"/v1/chat/completions", "/v1/messages"} {
		call := h.send(t, path, chatBody("STEADY question", ""))
		require.Equal(t, []string{"cheap-model"}, call.called, path)
		require.Equal(t, "cheap-model", call.record.SelectedModel, path)
		require.Empty(t, fallbackOutcomes(call.record), path)
	}
}

// A record keeps the reasoning its model was sent: no effort when the decision
// turned reasoning off or the Router answered itself, the client's effort when
// the Router forwarded it, and the decision's when it set one. The mode is set
// only where the Router wrote reasoning for the model: on or off for a model
// with a reasoning family, none for a request passed through.
func TestReplayRecordsTheEffortItSent(t *testing.T) {
	h := newServedModelHarness(t)
	high := `,"reasoning_effort":"high"`
	off := h.send(t, "/v1/chat/completions", chatBody("NOTHINK question", high))
	require.Equal(t, []string{"think-model"}, off.called)
	require.Equal(t, map[string]any{"enable_thinking": false}, off.bodies[0]["chat_template_kwargs"])
	require.NotContains(t, off.bodies[0], "reasoning_effort")
	require.Equal(t, "off", off.record.ReasoningMode)
	require.Empty(t, off.record.ReasoningEffort)

	on := h.send(t, "/v1/chat/completions", chatBody("THINK question", ""))
	require.Equal(t, map[string]any{"enable_thinking": true}, on.bodies[0]["chat_template_kwargs"])
	require.Equal(t, "on", on.record.ReasoningMode)
	require.Empty(t, on.record.ReasoningEffort)

	blocked := h.send(t, "/v1/chat/completions", chatBody("BLOCKME question", high))
	require.Empty(t, blocked.called)
	require.Empty(t, blocked.record.ReasoningMode)
	require.Empty(t, blocked.record.ReasoningEffort)

	forwarded := h.send(t, "/v1/chat/completions", chatBody("plain question", high))
	require.Equal(t, "high", forwarded.bodies[0]["reasoning_effort"])
	require.Empty(t, forwarded.record.ReasoningMode)
	require.Equal(t, "high", forwarded.record.ReasoningEffort)

	passed := h.send(t, "/v1/chat/completions", chatBody("FAMILYLESS question", ""))
	require.Equal(t, []string{"main-model"}, passed.called)
	for _, field := range []string{"reasoning_effort", "chat_template_kwargs", "reasoning", "thinking"} {
		require.NotContains(t, passed.bodies[0], field)
	}
	require.Empty(t, passed.record.ReasoningMode)
	require.Empty(t, passed.record.ReasoningEffort)

	decided := h.send(t, "/v1/messages", `{"model":"vllm-sr/cc","max_tokens":50,"thinking":{"type":"adaptive"},`+
		`"output_config":{"effort":"high"},"messages":[{"role":"user","content":"question"}]}`, claudeCodeHeaders("effort-run", "main")...)
	require.Equal(t, map[string]any{"effort": "medium"}, decided.bodies[0]["output_config"])
	require.Equal(t, "on", decided.record.ReasoningMode)
	require.Equal(t, "medium", decided.record.ReasoningEffort)
}

// Claude Code's side requests, such as its session title and compactions,
// carry the main agent's session id. Each class is held under a key of its
// own, so the main agent keeps the model its own first request got, even when
// the title request comes first.
func TestSideRequestsLeaveTheMainAgentsModel(t *testing.T) {
	h := newServedModelHarness(t)
	body := `{"model":"vllm-sr/cc","max_tokens":50,"messages":[{"role":"user","content":"question"}]}`
	var models []string
	for _, class := range []string{"auxiliary", "main", "compaction", "main", "main", "auxiliary"} {
		call := h.send(t, "/v1/messages", body, claudeCodeHeaders("side-run", class)...)
		models = append(models, call.record.SelectedModel)
	}
	require.Equal(t, []string{"qwen-small", "qwen-msg", "qwen-small", "qwen-msg", "qwen-msg", "qwen-small"}, models)
}

// Under scope: session a subagent is held on the model its first request got,
// like the session itself, before a switch could be priced.
func TestSessionScopeHoldsASubagentBeforePricingASwitch(t *testing.T) {
	h := newServedModelHarness(t)
	main := routing.HeaderField{Name: "x-claude-code-session-id", Value: "agent-run"}
	agent := routing.HeaderField{Name: "x-claude-code-agent-id", Value: "agent-1"}
	h.send(t, "/v1/chat/completions", chatBody("plain question", ""), main)
	require.Equal(t, []string{"main-model"}, h.send(t, "/v1/chat/completions", chatBody("plain question", ""), main, agent).called)
	held := h.send(t, "/v1/chat/completions", chatBody("STEADY question", ""), main, agent)
	require.Equal(t, []string{"main-model"}, held.called)
	require.Equal(t, "session_scope_protect", sessionPolicyString(held.record, "reason"))
	require.NotContains(t, held.record.SessionPolicy, "cache_economics")
}
