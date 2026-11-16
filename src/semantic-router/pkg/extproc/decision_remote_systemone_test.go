package extproc

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/publicmodels"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// systemOneRoutingConfig sends engineering tickets to deep on a systemone
// deployment's answer. Its API sits behind a gateway with its own credential
// header and path prefix; any other answer, or none, goes to fast.
const systemOneRoutingConfig = `
version: v0.5
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
recipes:
  - name: default
    signals:
      systemone:
        - name: support_team
          deployment: gateway-d3
          question:
            type: choice
            instructions: Which team should handle this ticket?
            choices:
              - {key: billing, description: Payments and refunds}
              - {key: engineering, description: Outages and bugs}
    decisions:
      - name: engineering
        priority: 200
        rules:
          operator: AND
          on_unknown: no_match
          conditions: [{type: systemone, name: support_team, label: engineering}]
        modelRefs: [{model: deep}]
      - name: everything-else
        priority: 10
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: fast}]
global:
  router:
    list_backend_models: true
  model_catalog:
    deployments:
      gateway-d3:
        provider: systemone
        schema: vllm-sr/4.0
        endpoint: GATEWAY/Unified/v1/systemone
        model: vLLM-SR-d3-27B
        auth: {header: Ocp-Apim-Subscription-Key, prefix: "", api_key_env: REMOTE_DECISION_TEST_KEY}
        timeout_ms: 1500
`

// fakeSystemOneGateway answers System One questions under /Unified/v1 only
// for its subscription key, choosing what choice returns. It lists no models.
type fakeSystemOneGateway struct {
	choice atomic.Value // "engineering", "billing", "unavailable" or "slow"
	asked  atomic.Int64
	body   atomic.Value // the last question body
}

func (g *fakeSystemOneGateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost || r.URL.Path != "/Unified/v1/systemone" || r.Header.Get("Ocp-Apim-Subscription-Key") != "fake-gateway-key" || r.Header.Get("Authorization") != "" {
		http.NotFound(w, r)
		return
	}
	g.asked.Add(1)
	body, _ := io.ReadAll(r.Body)
	g.body.Store(string(body))
	choice, _ := g.choice.Load().(string)
	switch choice {
	case "unavailable":
		w.WriteHeader(http.StatusServiceUnavailable)
		return
	case "slow":
		select {
		case <-r.Context().Done():
		case <-time.After(5 * time.Second):
		}
		return
	}
	var request struct {
		Questions map[string]struct {
			Criteria map[string]any `json:"criteria"`
		} `json:"questions"`
	}
	_ = json.Unmarshal(body, &request)
	answers := map[string]any{}
	for id, question := range request.Questions {
		probabilities := map[string]float64{}
		for key := range question.Criteria {
			probabilities[key] = 0.05
		}
		probabilities[choice] = 1 - 0.05*float64(len(probabilities)-1)
		answers[id] = map[string]any{"type": "choice", "choice": choice, "probabilities": probabilities, "confidence": 0.9}
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"answers": answers, "model": "vLLM-SR-d3-27B", "usage": map[string]int{"input_tokens": 12}})
}

func TestSystemOneDeploymentRoutesChatAndPreview(t *testing.T) {
	gateway := &fakeSystemOneGateway{}
	gatewayServer := httptest.NewServer(gateway)
	defer gatewayServer.Close()
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) }))
	defer backend.Close()
	t.Setenv("REMOTE_DECISION_TEST_KEY", "fake-gateway-key")
	cfg, err := config.ParseYAMLBytes([]byte(strings.NewReplacer(
		"BACKEND", strings.TrimPrefix(backend.URL, "http://"), "GATEWAY", gatewayServer.URL,
	).Replace(systemOneRoutingConfig)))
	if err != nil {
		t.Fatal(err)
	}
	router := newLooperRouter(t, cfg)
	if _, pooled := cfg.ModelConfig["gateway-d3"]; pooled {
		t.Fatal("the decision model joined the request backend pool")
	}
	for _, cluster := range router.upstreamSet().Topology().Clusters {
		if cluster.Name == "gateway-d3" {
			t.Fatal("the decision model became an upstream cluster")
		}
	}
	for _, model := range publicmodels.NewOpenAIModelList(cfg, 0).Data {
		if model.ID == "gateway-d3" || model.ID == "vLLM-SR-d3-27B" {
			t.Fatalf("the decision model is listed as a request target: %+v", model)
		}
	}
	opts := routing.DefaultOptions
	opts.ExecutesFallback = true
	engine := routing.NewEngine(router, opts)
	// Each case asks about its own text: the generation caches an answer to
	// the same question about the same text.
	message := func(choice string) string {
		return "Our app has returned 500 errors since the 09:00 deploy. Case " + choice + "."
	}
	route := func(choice string) (string, time.Duration) {
		t.Helper()
		gateway.choice.Store(choice)
		header := routing.Header{
			{Name: ":method", Value: "POST"},
			{Name: ":path", Value: "/v1/chat/completions"},
			{Name: ":authority", Value: "router"},
			{Name: ":scheme", Value: "http"},
			{Name: "content-type", Value: "application/json"},
			{Name: "x-request-id", Value: "systemone-decision-" + choice},
		}
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()
		started := time.Now()
		plan, planErr := engine.Plan(ctx, &routing.Request{Header: header, Body: []byte(
			`{"model":"vllm-sr/auto","messages":[{"role":"user","content":"` + message(choice) + `"}]}`)})
		if planErr != nil {
			t.Fatalf("%s: plan: %v", choice, planErr)
		}
		defer plan.Finish(nil)
		if plan.Call == nil {
			t.Fatalf("%s: the request was answered instead of routed: %+v", choice, plan.Immediate)
		}
		return plan.Call.Route, time.Since(started)
	}

	if got, _ := route("engineering"); got != "deep" {
		t.Fatalf("an engineering answer routes to deep, got %q", got)
	}
	if body, _ := gateway.body.Load().(string); !strings.Contains(body, `"model":"vLLM-SR-d3-27B"`) ||
		!strings.Contains(body, `"criteria":{"billing":"Payments and refunds","engineering":"Outages and bugs"}`) {
		t.Fatalf("the gateway got %s", body)
	}
	if got, _ := route("billing"); got != "fast" {
		t.Fatalf("a billing answer falls to fast, got %q", got)
	}
	repeated := gateway.asked.Load()
	if got, _ := route("billing"); got != "fast" || gateway.asked.Load() != repeated {
		t.Fatalf("a repeated request routes on the cached answer without asking again: %q, asked %d more", got, gateway.asked.Load()-repeated)
	}
	if got, _ := route("unavailable"); got != "fast" {
		t.Fatalf("an unavailable API leaves the signal unknown and the request routed, got %q", got)
	}
	if got, took := route("slow"); got != "fast" || took > 4*time.Second {
		t.Fatalf("a slow API ends at the deployment's timeout: %q after %s", got, took)
	}

	gateway.choice.Store("engineering")
	asked := gateway.asked.Load()
	preview, err := router.ClassificationService.ClassifyIntentForEval(context.Background(), services.IntentRequest{Text: message("preview")})
	if err != nil {
		t.Fatal(err)
	}
	if gateway.asked.Load() != asked+1 || preview.SignalConfidences["systemone:support_team"] < 0.8 || preview.DecisionResult == nil || preview.DecisionResult.DecisionName != "engineering" {
		t.Fatalf("routing preview asks the deployment: confidences %v errors %v decision %+v", preview.SignalConfidences, preview.SignalErrors, preview.DecisionResult)
	}
}

func TestDecisionSelectorAsksASystemOneDeployment(t *testing.T) {
	gateway := &fakeSystemOneGateway{}
	gateway.choice.Store("deep")
	gatewayServer := httptest.NewServer(gateway)
	defer gatewayServer.Close()
	t.Setenv("REMOTE_DECISION_TEST_KEY", "fake-gateway-key")
	cfg, err := config.ParseYAMLBytes([]byte(`version: v0.5
providers:
  defaults: {model: fast}
  models:
    - name: fast
      backend_refs: [{name: fast, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
    - name: deep
      backend_refs: [{name: deep, endpoint: "127.0.0.1:8001", protocol: http, weight: 1}]
recipes:
  - name: default
    decisions:
      - name: choose
        priority: 100
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: fast}, {model: deep}]
        algorithm:
          type: systemone
          systemone:
            deployment: gateway-d3
            instructions: Which model should answer this request?
            candidates:
              fast: Quick answers to routine requests
              deep: Careful multi-step reasoning
global:
  model_catalog:
    deployments:
      gateway-d3:
        provider: systemone
        schema: vllm-sr/4.0
        endpoint: ` + gatewayServer.URL + `/Unified/v1/systemone
        model: vLLM-SR-d3-27B
        auth: {header: Ocp-Apim-Subscription-Key, prefix: "", api_key_env: REMOTE_DECISION_TEST_KEY}
`))
	if err != nil {
		t.Fatal(err)
	}
	cards, err := prepareDecisionSelectionCards(cfg, nil)
	if err != nil {
		t.Fatalf("a systemone selector needs no model runtime to prepare: %v", err)
	}
	remote, err := modelservice.NewRemoteSystemOne(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer remote.Close()
	router := &OpenAIRouter{Config: cfg, decisionCards: cards, systemOneDecider: remote}
	decision := cfg.GetDecisionByName("choose")
	result, err := router.newDecisionModelSelector(*decision.Algorithm.SystemOne).Select(t.Context(), &selection.SelectionContext{
		Query: "Prove that the square root of two is irrational.", DecisionName: decision.Name, CandidateModels: decision.ModelRefs,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.SelectedModel != "deep" || result.Method != selection.MethodSystemOne || gateway.asked.Load() != 1 {
		t.Fatalf("result %+v after %d remote calls", result, gateway.asked.Load())
	}
	if body, _ := gateway.body.Load().(string); !strings.Contains(body, `"criteria":{"fast":"Quick answers to routine requests","deep":"Careful multi-step reasoning"}`) {
		t.Fatalf("the selector's candidates go as criteria: %s", body)
	}
}
