package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/budget"
)

type nativeRoutingDecider func(context.Context, string, modelservice.Request) (modelservice.Response, error)

func (f nativeRoutingDecider) Decide(ctx context.Context, deployment string, request modelservice.Request) (modelservice.Response, error) {
	return f(ctx, deployment, request)
}

func TestSystemOneDecisionBudgetStartsAfterSignalsAndVariesByDecision(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`version: v0.3
listeners:
  - name: native
    port: 8801
    systemone: {models: [vllm-sr/auto]}
providers:
  models:
    - {name: kai, api_format: systemone, deployment: local-kai}
    - name: vega
      api_format: systemone
      provider_model_id: native-vega
      backend_refs: [{provider: systemone-compatible, base_url: http://unused.invalid/v1}]
entrypoints:
  - {api: systemone, model_names: [vllm-sr/auto], recipe: native}
global:
  model_catalog:
    deployments:
      local-kai: {provider: model_runtime, artifact: vllm-sr/Decision-2.0-Kai-0.6B, device: cpu}
routing: {}
recipes:
  - name: native
    routing:
      fallback: {enabled: false}
      signals:
        decision:
          - name: task_kind
            deployment: local-kai
            timeout_ms: 2000
            question:
              type: choice
              instructions: Does the original task need escalation?
              choices: [{key: simple, description: Routine task}, {key: hard, description: Complex task}]
      decisions:
        - name: simple
          rules: {type: decision, name: task_kind, label: simple}
          modelRefs: [{model: kai}]
          algorithm:
            type: cascade
            budget: {deadline: 500ms, max_calls: 1}
            quality: &quality
              type: uncalibrated
              acceptance:
                rules: [{question_type: noul, field: top_probability, predicate: {gte: 0}}]
            stages: [{name: answer, kind: native, model: kai}]
        - name: hard
          rules: {type: decision, name: task_kind, label: hard}
          modelRefs: [{model: kai}, {model: vega}]
          algorithm:
            type: cascade
            budget: {deadline: 1s, max_calls: 2}
            quality: *quality
            stages:
              - name: fast
                kind: native
                model: kai
                accept:
                  rules: [{question_type: noul, field: top_probability, predicate: {gte: 0.9}}]
              - {name: strong, kind: native, model: vega}
`))
	if err != nil {
		t.Fatal(err)
	}
	executors, err := prepareNativeExecutors(cfg)
	if err != nil {
		t.Fatal(err)
	}
	classifiers, err := classification.BuildRecipeClassifiers(cfg, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer classifiers.Close()
	classifier, ok := classifiers.ForRecipe("native")
	if !ok {
		t.Fatal("native recipe classifier unavailable")
	}
	router := &OpenAIRouter{Config: cfg, RecipeClassifiers: classifiers, nativeExecutors: executors}
	for _, tc := range []struct {
		name      string
		wantCalls []string
		cancel    bool
	}{
		{name: "simple", wantCalls: []string{"kai"}},
		{name: "hard", wantCalls: []string{"kai", "vega"}},
		{name: "canceled", cancel: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			classifier.SetDecisionDecider(nativeRoutingDecider(func(signalCtx context.Context, deployment string, task modelservice.Request) (modelservice.Response, error) {
				if deployment != "local-kai" || !strings.Contains(task.State, `"questions"`) {
					t.Error("signal did not receive the original native task")
				}
				deadline, bounded := signalCtx.Deadline()
				if !bounded || time.Until(deadline) <= time.Second {
					t.Error("algorithm deadline replaced the signal's own timeout")
				}
				// A model-backed signal may perform physical exchanges, but none
				// belongs to the subsequently selected algorithm's call ledger.
				for range 3 {
					if consumeErr := budget.Consume(signalCtx); consumeErr != nil {
						t.Error("signal consumed algorithm budget", consumeErr)
					}
				}
				if tc.cancel {
					cancel()
					return modelservice.Response{}, context.Canceled
				}
				return modelservice.Response{Answers: map[string]modelservice.Answer{
					"task_kind": {Type: "choice", Choice: tc.name, Probabilities: map[string]float64{"simple": 0.5, "hard": 0.5}},
				}}, nil
			}))
			var calls []string
			request := json.RawMessage(`{"model":"vllm-sr/auto","state":"an original task","questions":{"needs_reasoning":{"type":"noul","instructions":"Does this need reasoning?"}}}`)
			status, body, routeErr := router.RouteSystemOne(ctx, "vllm-sr/auto", request, func(callCtx context.Context, model string, original json.RawMessage) (int, []byte, error) {
				if string(original) != string(request) {
					t.Error("routing replaced the original native question")
				}
				if consumeErr := budget.Consume(callCtx); consumeErr != nil {
					return 0, nil, consumeErr
				}
				calls = append(calls, model)
				return http.StatusOK, []byte(`{"model":"actual","answers":{"needs_reasoning":{"type":"noul","noul":0.6}}}`), nil
			})
			if tc.cancel {
				if !errors.Is(routeErr, context.Canceled) || len(calls) != 0 {
					t.Fatalf("canceled signal launched backend: calls=%v err=%v", calls, routeErr)
				}
				return
			}
			if routeErr != nil || status != http.StatusOK || !reflect.DeepEqual(calls, tc.wantCalls) {
				t.Fatalf("status=%d calls=%v err=%v", status, calls, routeErr)
			}
			var response struct {
				Routing struct {
					Decision string `json:"decision"`
					Calls    int    `json:"model_calls"`
				} `json:"routing"`
			}
			if json.Unmarshal(body, &response) != nil || response.Routing.Decision != tc.name || response.Routing.Calls != len(tc.wantCalls) {
				t.Fatalf("wrong decision or budget accounting: %s", body)
			}
		})
	}
}
