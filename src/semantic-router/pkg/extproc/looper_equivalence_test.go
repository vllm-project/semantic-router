package extproc

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

// The Looper equivalence gate. Every Looper algorithm runs a deterministic
// fixture: scripted backends whose answers depend only on the request they
// get, and a client request. The record is the set of upstream calls (route,
// model and body after the hop's plugins) and the client's response, with
// volatile values normalized. The goldens were recorded on the loopback path,
// where each hop called back through the gateway, before the request-graph
// executor took the hops in process; the in-process hops reproduce them.
// Rewrite a golden only for a deliberate change of behavior, in a commit of
// its own.

var updateLooperGolden = flag.Bool("update-looper-golden", false,
	"rewrite the Looper equivalence goldens (only for a deliberate change of behavior)")

const (
	looperGoldenDir      = "testdata/looper-equivalence"
	looperListener       = "http-8899"
	looperClientRequest  = `{"model":"vllm-sr/auto","messages":[{"role":"user","content":"What is the capital of France?"}]}`
	looperClientStreamed = `{"model":"vllm-sr/auto","stream":true,"messages":[{"role":"user","content":"What is the capital of France?"}]}`
)

const looperEquivalenceConfig = `
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    timeout: 30s
providers:
  defaults:
    model: model-a
  models:
    - name: model-a
      provider_model_id: model-a
      reasoning: {family: qwen3}
      api_format: openai
      backend_refs: [{name: fake-a, endpoint: BACKEND, protocol: http, provider: vllm}]
    - name: model-b
      provider_model_id: model-b
      api_format: openai
      backend_refs: [{name: fake-b, endpoint: BACKEND, protocol: http, provider: vllm}]
    - name: model-c
      provider_model_id: model-c
      api_format: openai
      backend_refs: [{name: fake-c, endpoint: BACKEND, protocol: http, provider: vllm}]
    - name: judge
      provider_model_id: judge
      api_format: openai
      backend_refs: [{name: fake-judge, endpoint: BACKEND, protocol: http, provider: vllm}]
routing:
  modelCards:
    - name: model-a
    - name: model-b
    - name: model-c
    - name: judge
  decisions:
    - name: looper_route
      priority: 10
      rules:
        operator: AND
        conditions: []
DECISION
global:
  integrations:
    looper:
      timeout_seconds: 30
`

// looperFixture is one algorithm's deterministic scenario.
type looperFixture struct {
	name string
	// decision is the rest of routing.decisions[0]: its modelRefs and
	// algorithm, indented for the config above.
	decision string
	request  string
	// failing names models whose backend answers 503.
	failing []string
	// slow delays every successful answer, so a failure always arrives first
	// where the algorithm stops once it has enough answers.
	slow time.Duration
	// peak records the most calls in flight at once, for an algorithm that
	// caps its concurrency; slow answers make the cap the peak.
	peak bool
}

var looperFixtures = []looperFixture{
	{name: "ratings", request: looperClientRequest, slow: 100 * time.Millisecond, peak: true, decision: `
      modelRefs: [{model: model-a, use_reasoning: true}, {model: model-b}, {model: model-c}]
      algorithm:
        type: ratings
        ratings: {max_concurrent: 2, on_error: skip}
      plugins:
        - type: system_prompt
          configuration: {enabled: true, mode: insert, system_prompt: Answer in one sentence.}
        - type: header_mutation
          configuration:
            add: [{name: X-Fixture-Hop, value: per-hop}]`},
	{name: "ratings-partial-stream", request: looperClientStreamed, failing: []string{"model-b"}, decision: `
      modelRefs: [{model: model-a}, {model: model-b}, {model: model-c}]
      algorithm:
        type: ratings
        ratings: {max_concurrent: 3, on_error: skip}`},
	{name: "confidence", request: looperClientRequest, decision: `
      modelRefs: [{model: model-a}, {model: model-b}]
      algorithm:
        type: confidence
        confidence:
          confidence_method: avg_logprob
          threshold: 0.6
          escalation_order: small_to_large`},
	{name: "remom", request: looperClientRequest, decision: `
      modelRefs: [{model: model-a}, {model: model-b}]
      algorithm:
        type: remom
        remom:
          breadth_schedule: [3]
          model_distribution: round_robin
          synthesis_model: model-b
          max_concurrent: 2
          shuffle_seed: 7`},
	{name: "fusion-one-call", request: looperClientRequest, decision: `
      modelRefs: [{model: model-a}, {model: model-b}]
      algorithm:
        type: fusion
        fusion:
          model: judge
          analysis_models: [model-a, model-b, model-c]
          analysis_mode: one_call
          max_concurrent: 3
          min_successful_responses: 3
          on_error: skip`},
	{name: "fusion-separate-quorum", request: looperClientRequest, failing: []string{"model-c"}, slow: 50 * time.Millisecond, decision: `
      modelRefs: [{model: model-a}, {model: model-b}]
      algorithm:
        type: fusion
        fusion:
          model: judge
          analysis_models: [model-a, model-b, model-c]
          analysis_mode: separate
          max_concurrent: 3
          min_successful_responses: 2
          include_analysis: true
          on_error: skip`},
	{name: "prompt-selection", request: looperClientRequest, decision: `
      modelRefs: [{model: model-a}, {model: model-b}]
      algorithm:
        type: prompt
        prompt: {model: judge, instructions: Choose model-b for geography.}`},
	{name: "workflows-static", request: looperClientRequest, decision: `
      modelRefs: [{model: model-a}, {model: model-b}, {model: judge}]
      algorithm:
        type: workflows
        workflows:
          mode: static
          roles:
            - {name: thinker, models: [model-a]}
            - {name: worker, models: [model-b]}
          final: {model: judge}`},
}

// looperRecord is what the gate compares.
type looperRecord struct {
	Calls []looperCall `json:"calls"`
	// PeakConcurrency is the most calls in flight at once, when recorded.
	PeakConcurrency int         `json:"peak_concurrency,omitempty"`
	Response        looperReply `json:"response"`
}

// looperCall is one request a backend received, with the headers the
// fixtures set (the decision's header mutation, and one the client sends).
type looperCall struct {
	Route  string            `json:"route"`
	Model  string            `json:"model"`
	Header map[string]string `json:"header,omitempty"`
	Body   json.RawMessage   `json:"body"`
}

type looperReply struct {
	Status int               `json:"status"`
	Header map[string]string `json:"header"`
	Body   json.RawMessage   `json:"body,omitempty"`
	Events []json.RawMessage `json:"events,omitempty"`
}

// looperBackend answers every model from a script that depends only on the
// request, and records each request.
type looperBackend struct {
	failing []string
	slow    time.Duration
	mu      sync.Mutex
	calls   []looperCall
	running int
	peak    int
}

func (b *looperBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)
	var req struct {
		Model          string          `json:"model"`
		Stream         bool            `json:"stream"`
		Logprobs       bool            `json:"logprobs"`
		ResponseFormat json.RawMessage `json:"response_format"`
		Messages       []struct {
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(raw, &req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	call := looperCall{Route: r.Header.Get("X-Selected-Model"), Model: req.Model, Body: canonicalJSON(raw)}
	for name, values := range r.Header {
		if name = strings.ToLower(name); strings.HasPrefix(name, "x-fixture-") {
			if call.Header == nil {
				call.Header = map[string]string{}
			}
			call.Header[name] = strings.Join(values, ",")
		}
	}
	b.mu.Lock()
	b.calls = append(b.calls, call)
	b.running++
	b.peak = max(b.peak, b.running)
	b.mu.Unlock()
	defer func() {
		b.mu.Lock()
		b.running--
		b.mu.Unlock()
	}()
	for _, failing := range b.failing {
		if req.Model == failing {
			http.Error(w, `{"error":{"message":"overloaded","type":"server_error"}}`, http.StatusServiceUnavailable)
			return
		}
	}
	last := ""
	if n := len(req.Messages); n > 0 {
		last = string(req.Messages[n-1].Content)
	}
	content := scriptedAnswer(req.Model, last, len(req.ResponseFormat) > 0)
	time.Sleep(b.slow)
	if req.Stream {
		writeScriptedStream(w, req.Model, content)
		return
	}
	writeScriptedCompletion(w, req.Model, content, req.Logprobs, len(req.Messages))
}

// scriptedAnswer is a model's answer: the judge returns its analysis, or its
// choice of model, as JSON when asked for one, and every answer differs in
// length by model, so an algorithm that orders answers by length orders them
// the same way each run.
func scriptedAnswer(model, prompt string, wantsJSON bool) string {
	switch {
	case model == "judge" && strings.Contains(prompt, "consensus"):
		return `{"consensus":["Paris is the capital"],"contradictions":[],"unique_insights":["history"]}`
	case model == "judge" && wantsJSON:
		return `{"selected_model":"model-b","rationale":"scripted"}`
	}
	digest := sha256.Sum256([]byte(prompt))
	detail := map[string]string{"model-a": "", "model-b": " in detail", "model-c": " in much more detail", "judge": " finally"}
	return fmt.Sprintf("%s says Paris%s (%s)", model, detail[model], hex.EncodeToString(digest[:4]))
}

// scriptedLogprob makes model-a unsure and every other model sure.
func scriptedLogprob(model string) float64 {
	if model == "model-a" {
		return -2.5
	}
	return -0.05
}

func writeScriptedCompletion(w http.ResponseWriter, model, content string, logprobs bool, messages int) {
	choice := map[string]any{
		"index": 0, "message": map[string]any{"role": "assistant", "content": content}, "finish_reason": "stop",
	}
	if logprobs {
		tokens := []map[string]any{}
		for _, word := range strings.Fields(content) {
			lp := scriptedLogprob(model)
			tokens = append(tokens, map[string]any{
				"token": word, "logprob": lp,
				"top_logprobs": []map[string]any{{"token": word, "logprob": lp}, {"token": "x", "logprob": lp - 1.5}},
			})
		}
		choice["logprobs"] = map[string]any{"content": tokens}
	}
	body, _ := json.Marshal(map[string]any{
		"id": "chatcmpl-backend", "object": "chat.completion", "created": 1, "model": model,
		"choices": []any{choice},
		"usage":   map[string]any{"prompt_tokens": 10 * messages, "completion_tokens": 5, "total_tokens": 10*messages + 5},
	})
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(body)
}

func writeScriptedStream(w http.ResponseWriter, model, content string) {
	w.Header().Set("Content-Type", "text/event-stream")
	chunk := func(delta map[string]any, finish any, usage any) {
		event := map[string]any{
			"id": "chatcmpl-backend", "object": "chat.completion.chunk", "created": 1, "model": model,
			"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
		}
		if usage != nil {
			event["choices"] = []any{}
			event["usage"] = usage
		}
		data, _ := json.Marshal(event)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
	}
	chunk(map[string]any{"role": "assistant"}, nil, nil)
	for _, word := range strings.SplitAfter(content, " ") {
		chunk(map[string]any{"content": word}, nil, nil)
	}
	chunk(map[string]any{}, "stop", nil)
	chunk(nil, nil, map[string]any{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
}

// looperHarness is one fixture's Router and backends.
type looperHarness struct {
	router  *OpenAIRouter
	backend *looperBackend
	set     *upstream.Set
	peak    bool
}

func newLooperHarness(t *testing.T, fixture looperFixture) *looperHarness {
	t.Helper()
	backend := &looperBackend{failing: fixture.failing, slow: fixture.slow}
	backendServer := httptest.NewServer(backend)
	t.Cleanup(backendServer.Close)
	configYAML := strings.NewReplacer(
		"BACKEND", strings.TrimPrefix(backendServer.URL, "http://"),
		"DECISION", strings.TrimPrefix(fixture.decision, "\n"),
	).Replace(looperEquivalenceConfig)
	cfg, err := config.ParseYAMLBytes([]byte(configYAML))
	if err != nil {
		t.Fatalf("parse the %s fixture: %v", fixture.name, err)
	}
	router := newLooperRouter(t, cfg)
	return &looperHarness{router: router, backend: backend, set: router.upstreamSet(), peak: fixture.peak}
}

// newLooperRouter builds the fixture's Router on its startup snapshot, which
// owns an upstream set, as a Router's does in either gateway mode.
func newLooperRouter(t *testing.T, cfg *config.RouterConfig) *OpenAIRouter {
	t.Helper()
	router, err := buildOpenAIRouterFromConfig(cfg)
	if err != nil {
		t.Fatalf("build the router: %v", err)
	}
	t.Cleanup(func() { _ = router.Close() })
	parts := []configsnapshot.PartBuilder{{
		Component: configsnapshot.ComponentUpstream,
		Build: func(_ context.Context, candidate *configsnapshot.Snapshot, _ configsnapshot.Part) (configsnapshot.Part, error) {
			return upstream.Build(candidate.Config(), upstream.Options{})
		},
	}}
	snapshot, err := configsnapshot.NewManager(configsnapshot.Options{Parts: parts}).Install(context.Background(), configsnapshot.Update{
		Origin: configsnapshot.Origin{Source: configsnapshot.SourceStartup}, Config: cfg,
	})
	if err != nil {
		t.Fatalf("install the snapshot: %v", err)
	}
	t.Cleanup(func() { _ = snapshot.Release(cleanupContext(t)) })
	newRouterGeneration(router, snapshot)
	return router
}

func cleanupContext(t *testing.T) context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	t.Cleanup(cancel)
	return ctx
}

// looperAdapters are the two ways a client request reaches the routing core:
// the ext_proc gRPC loop Envoy drives, and routing sessions, as the
// standalone frontend drives them.
var looperAdapters = map[string]func(*OpenAIRouter) routing.Engine{
	"extproc": func(router *OpenAIRouter) routing.Engine {
		return routing.NewEngine(&extprocStreamProcessor{router: router}, routing.DefaultOptions)
	},
	"sessions": func(router *OpenAIRouter) routing.Engine {
		opts := routing.DefaultOptions
		opts.ExecutesFallback = true
		return routing.NewEngine(router, opts)
	},
}

func (h *looperHarness) serve(t *testing.T, engine routing.Engine, body string) looperRecord {
	t.Helper()
	header := routing.Header{
		{Name: ":method", Value: "POST"},
		{Name: ":path", Value: "/v1/chat/completions"},
		{Name: ":authority", Value: "router"},
		{Name: ":scheme", Value: "http"},
		{Name: "content-type", Value: "application/json"},
		{Name: "x-request-id", Value: "looper-equivalence"},
		{Name: "x-fixture-client", Value: "the client's own header"},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	plan, err := engine.Plan(ctx, &routing.Request{Header: header, Body: []byte(body)})
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	defer plan.Finish(nil)
	answer := plan.Immediate
	if answer == nil {
		answer = h.send(t, ctx, engine, plan)
	}
	h.backend.mu.Lock()
	calls := append([]looperCall(nil), h.backend.calls...)
	peak := h.backend.peak
	h.backend.calls, h.backend.peak = nil, 0
	h.backend.mu.Unlock()
	sort.SliceStable(calls, func(i, j int) bool {
		if calls[i].Route != calls[j].Route {
			return calls[i].Route < calls[j].Route
		}
		return string(calls[i].Body) < string(calls[j].Body)
	})
	record := looperRecord{Calls: calls, Response: normalizeLooperReply(answer)}
	if h.peak {
		record.PeakConcurrency = peak
	}
	return record
}

// send serves a planned call the way the standalone gateway does, for a
// request whose Router call chose the model rather than answered.
func (h *looperHarness) send(t *testing.T, ctx context.Context, engine routing.Engine, plan *routing.Plan) *routing.Response {
	t.Helper()
	result, err := h.set.Execute(ctx, plan.Call, looperListener)
	if err != nil || result.Response == nil {
		t.Fatalf("send the planned call: %v", err)
	}
	defer result.Response.Body.Close()
	out, err := engine.Respond(ctx, plan, &routing.UpstreamResponse{
		Status: result.Response.StatusCode, Header: routingHeaderOf(result.Response.Header), Body: result.Response.Body,
	})
	if err != nil {
		t.Fatalf("respond: %v", err)
	}
	if out.Stream != nil {
		var body []byte
		for {
			chunk, err := out.Stream.Next()
			body = append(body, chunk...)
			if err != nil {
				break
			}
		}
		out.Body, out.Stream = body, nil
	}
	return out
}

// volatileHeaders change on every run; the gate keeps them out.
var volatileHeader = regexp.MustCompile(`latency|-ms$|replay-id|request-id|trace`)

func normalizeLooperReply(resp *routing.Response) looperReply {
	reply := looperReply{Status: resp.Status, Header: map[string]string{}}
	for _, field := range resp.Header {
		if (field.Name == "content-type" || strings.HasPrefix(field.Name, "x-vsr-")) && !volatileHeader.MatchString(field.Name) {
			reply.Header[field.Name] = field.Value
		}
	}
	if strings.HasPrefix(reply.Header["content-type"], "text/event-stream") {
		for _, line := range bytes.Split(resp.Body, []byte("\n")) {
			if data, ok := bytes.CutPrefix(bytes.TrimSpace(line), []byte("data:")); ok {
				data = bytes.TrimSpace(data)
				if json.Valid(data) {
					reply.Events = append(reply.Events, normalizeVolatileJSON(data))
				} else {
					quoted, _ := json.Marshal(string(data))
					reply.Events = append(reply.Events, quoted)
				}
			}
		}
		return reply
	}
	reply.Body = normalizeVolatileJSON(resp.Body)
	return reply
}

// normalizeVolatileJSON replaces ids and timestamps, which a run mints anew,
// and re-encodes the value with sorted keys.
func normalizeVolatileJSON(raw []byte) json.RawMessage {
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		quoted, _ := json.Marshal(string(raw))
		return quoted
	}
	var walk func(any) any
	walk = func(v any) any {
		switch typed := v.(type) {
		case map[string]any:
			for key, inner := range typed {
				switch key {
				case "id", "created", "created_at", "latency_ms", "trace_id":
					typed[key] = "<volatile>"
				default:
					typed[key] = walk(inner)
				}
			}
		case []any:
			for i, inner := range typed {
				typed[i] = walk(inner)
			}
		}
		return v
	}
	out, _ := json.Marshal(walk(value))
	return out
}

func canonicalJSON(raw []byte) json.RawMessage {
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		quoted, _ := json.Marshal(string(raw))
		return quoted
	}
	out, _ := json.Marshal(value)
	return out
}

func looperGoldenPath(fixture string) string {
	return filepath.Join(looperGoldenDir, fixture+".json")
}

func encodeLooperRecord(t *testing.T, record looperRecord) []byte {
	t.Helper()
	data, err := json.MarshalIndent(record, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return append(data, '\n')
}

// TestLooperEquivalence holds every Looper algorithm, through both adapters,
// to the record of the loopback path.
func TestLooperEquivalence(t *testing.T) {
	for _, fixture := range looperFixtures {
		t.Run(fixture.name, func(t *testing.T) {
			path := looperGoldenPath(fixture.name)
			harness := newLooperHarness(t, fixture)
			records := map[string][]byte{}
			for _, adapter := range []string{"extproc", "sessions"} {
				records[adapter] = encodeLooperRecord(t, harness.serve(t, looperAdapters[adapter](harness.router), fixture.request))
			}
			if *updateLooperGolden {
				if err := os.MkdirAll(looperGoldenDir, 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, records["extproc"], 0o644); err != nil {
					t.Fatal(err)
				}
			}
			golden, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("read the golden: %v", err)
			}
			for _, adapter := range []string{"extproc", "sessions"} {
				if !bytes.Equal(golden, records[adapter]) {
					t.Errorf("%s differs from %s:\n%s", adapter, path, records[adapter])
				}
			}
		})
	}
}
