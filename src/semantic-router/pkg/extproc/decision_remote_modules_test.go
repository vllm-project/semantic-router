package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// moduleSystemOneConfig reads the domain and jailbreak signals and a decision
// signal with hosted-d3, a systemone deployment that is the decision model:
// no module names a model of its own, so every signal asks hosted-d3. Math
// that needs a tool goes to deep, anything else to fast.
const moduleSystemOneConfig = `
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    timeout: 30s
providers:
  defaults:
    model: fast
  models:
    - name: fast
      backend_refs: [{name: fast, endpoint: "BACKEND", protocol: http, weight: 1}]
    - name: deep
      backend_refs: [{name: deep, endpoint: "BACKEND", protocol: http, weight: 1}]
routing:
  signals:
    domains: [{name: math, description: Math requests., mmlu_categories: [math]}]
    jailbreak: [{name: attack, threshold: 0.75}]
    systemone:
      - name: needs_tools
        question: {type: noul, instructions: "Does the request need a tool?"}
  decisions:
    - name: attack
      priority: 300
      rules: {operator: AND, conditions: [{type: jailbreak, name: attack}]}
      modelRefs: [{model: fast}]
    - name: math
      priority: 200
      rules: {operator: AND, conditions: [{type: domain, name: math}, {type: systemone, name: needs_tools}]}
      modelRefs: [{model: deep}]
    - name: everything-else
      priority: 10
      rules: {operator: AND, conditions: []}
      modelRefs: [{model: fast}]
global:
  model_catalog:
    deployments:
      hosted-d3:
        provider: systemone
        endpoint: API/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: REMOTE_DECISION_TEST_KEY}
        timeout_ms: 3000
    system:
      decision_model: {deployment: hosted-d3}
`

// moduleSystemOneAPI is hosted-d3's System One API. It lists the types
// choice, noul, score and set (no span, no preset), chooses math, benign or
// the first option of a choice question, answers a noul question 0.9, and
// keeps the question IDs of each call.
type moduleSystemOneAPI struct {
	mu    sync.Mutex
	calls [][]string
}

func (a *moduleSystemOneAPI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Header.Get("Authorization") != "Bearer fake-decision-key" {
		http.Error(w, `{"error":{"code":"invalid_api_key"}}`, http.StatusUnauthorized)
		return
	}
	if r.Method == http.MethodGet && r.URL.Path == "/v1/systemone/models" {
		_, _ = w.Write([]byte(`{"data":[{"id":"vllm-sr/d3-ultra","question_types":["choice","noul","score","set"]}]}`))
		return
	}
	type question struct {
		Type     string         `json:"type"`
		Criteria map[string]any `json:"criteria"`
	}
	var body struct {
		Model     string              `json:"model"`
		Questions map[string]question `json:"questions"`
		States    map[string]struct {
			Questions map[string]question `json:"questions"`
		} `json:"states"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	var ids []string
	answer := func(questions map[string]question) map[string]any {
		answers := map[string]any{}
		for id, q := range questions {
			ids = append(ids, id)
			if q.Type == "noul" {
				answers[id] = map[string]any{"type": "noul", "noul": 0.9, "input_coverage": "complete"}
				continue
			}
			keys := make([]string, 0, len(q.Criteria))
			for key := range q.Criteria {
				keys = append(keys, key)
			}
			slices.Sort(keys)
			chosen := keys[0]
			for _, preferred := range []string{"math", "benign"} {
				if slices.Contains(keys, preferred) {
					chosen = preferred
				}
			}
			probabilities := map[string]float64{}
			for _, key := range keys {
				probabilities[key] = 0.1 / float64(len(keys)-1)
			}
			probabilities[chosen] = 0.9
			answers[id] = map[string]any{"type": "choice", "choice": chosen, "probabilities": probabilities, "confidence": 0.9, "input_coverage": "complete"}
		}
		return map[string]any{"model": body.Model, "answers": answers, "usage": map[string]int{"input_tokens": 12}}
	}
	response := answer(body.Questions)
	if len(body.States) > 0 {
		states := map[string]any{}
		for name, state := range body.States {
			states[name] = answer(state.Questions)
		}
		response["states"] = states
	}
	slices.Sort(ids)
	a.mu.Lock()
	a.calls = append(a.calls, ids)
	a.mu.Unlock()
	_ = json.NewEncoder(w).Encode(response)
}

func (a *moduleSystemOneAPI) answered() [][]string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return slices.Clone(a.calls)
}

func moduleSystemOneRouterConfig(t *testing.T, raw string) (*config.RouterConfig, *moduleSystemOneAPI) {
	t.Helper()
	api := &moduleSystemOneAPI{}
	server := httptest.NewServer(api)
	t.Cleanup(server.Close)
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) }))
	t.Cleanup(backend.Close)
	t.Setenv("REMOTE_DECISION_TEST_KEY", "fake-decision-key")
	cfg, err := config.ParseYAMLBytes([]byte(strings.NewReplacer(
		"BACKEND", strings.TrimPrefix(backend.URL, "http://"), "API", server.URL,
	).Replace(raw)))
	if err != nil {
		t.Fatal(err)
	}
	return cfg, api
}

func TestBuiltInSignalsAskASystemOneDecisionModelInOneCallPerStage(t *testing.T) {
	cfg, api := moduleSystemOneRouterConfig(t, moduleSystemOneConfig)
	later := counterValue(t, "vsr_model_runtime_stage_decision_calls_total", map[string]string{"deployment": "hosted-d3", "round": "later"})
	router := newLooperRouter(t, cfg)
	prepared := len(api.answered())
	engine := routing.NewEngine(router, routing.DefaultOptions)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	header := routing.Header{
		{Name: ":method", Value: "POST"},
		{Name: ":path", Value: "/v1/chat/completions"},
		{Name: ":authority", Value: "router"},
		{Name: ":scheme", Value: "http"},
		{Name: "content-type", Value: "application/json"},
		{Name: "x-request-id", Value: "systemone-modules"},
	}
	plan, err := engine.Plan(ctx, &routing.Request{Header: header, Body: []byte(
		`{"model":"vllm-sr/auto","messages":[{"role":"user","content":"Integrate x squared from 0 to 3 with a calculator."}]}`)})
	if err != nil {
		t.Fatal(err)
	}
	defer plan.Finish(nil)
	if plan.Call == nil || plan.Call.Route != "deep" {
		t.Fatalf("math that needs a tool routes to deep, got %+v", plan.Call)
	}
	calls := api.answered()[prepared:]
	want := []string{"domain_classifier:domain", "prompt_guard:attack", "systemone:needs_tools"}
	if len(calls) != 1 || !slices.Equal(calls[0], want) {
		t.Fatalf("the request stage asked hosted-d3 %v, want one call with %v", calls, want)
	}
	if got := counterValue(t, "vsr_model_runtime_stage_decision_calls_total", map[string]string{"deployment": "hosted-d3", "round": "later"}) - later; got != 0 {
		t.Fatalf("a stage made %v further calls to hosted-d3", got)
	}
}

func TestPIIAnswersAsAVerdictWhenTheSystemOneDecisionModelHasNoSpans(t *testing.T) {
	withPII := strings.Replace(moduleSystemOneConfig, "    jailbreak: [{name: attack, threshold: 0.75}]\n",
		"    jailbreak: [{name: attack, threshold: 0.75}]\n    pii: [{name: personal, threshold: 0.5}]\n", 1)
	withPII = strings.Replace(withPII, "conditions: [{type: jailbreak, name: attack}]", "conditions: [{type: jailbreak, name: attack}, {type: pii, name: personal}]", 1)
	withPII = strings.Replace(withPII, "rules: {operator: AND, conditions: [{type: jailbreak", "rules: {operator: OR, conditions: [{type: jailbreak", 1)
	cfg, _ := moduleSystemOneRouterConfig(t, withPII)
	fallback := map[string]string{"signal": "pii_classifier", "deployment": "hosted-d3"}
	fallbacks := counterValue(t, "vsr_signal_span_verdict_fallbacks_total", fallback)
	newLooperRouter(t, cfg)
	if got := counterValue(t, "vsr_signal_span_verdict_fallbacks_total", fallback) - fallbacks; got != 1 {
		t.Fatalf("span verdict fallbacks %v, want the PII fallback reported once", got)
	}

	masked := strings.Replace(withPII, "      modelRefs: [{model: deep}]\n",
		"      modelRefs: [{model: deep}]\n      plugins:\n        - type: masking\n          configuration: {enabled: true, signal: personal}\n", 1)
	cfg, _ = moduleSystemOneRouterConfig(t, masked)
	if _, err := buildOpenAIRouterFromConfig(cfg); err == nil || !strings.Contains(err.Error(), "masking replaces PII at the positions") {
		t.Fatalf("masking with a PII verdict: %v, want the configuration refused at load", err)
	}
}

// counterValue reads a counter of the default registry by name and labels;
// zero when it has not counted yet.
func counterValue(t *testing.T, name string, labels map[string]string) float64 {
	t.Helper()
	families, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, family := range families {
		if family.GetName() != name {
			continue
		}
		for _, metric := range family.GetMetric() {
			matched := 0
			for _, label := range metric.GetLabel() {
				if labels[label.GetName()] == label.GetValue() {
					matched++
				}
			}
			if matched == len(labels) {
				return metric.GetCounter().GetValue()
			}
		}
	}
	return 0
}
