package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/budget"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
)

type nativeRoutingDecider func(context.Context, string, modelservice.Request) (modelservice.Response, error)

type delayedReader struct {
	reader io.Reader
	delay  time.Duration
	waited bool
}

func (r *delayedReader) Read(buffer []byte) (int, error) {
	if !r.waited {
		time.Sleep(r.delay)
		r.waited = true
	}
	return r.reader.Read(buffer)
}

func (f nativeRoutingDecider) Decide(ctx context.Context, deployment string, request modelservice.Request) (modelservice.Response, error) {
	return f(ctx, deployment, request)
}

func TestSystemOneBudgetIncludesSignalsAndVariesByDecision(t *testing.T) {
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
        systemone:
          - name: task_kind
            deployment: local-kai
            timeout_ms: 2000
            question:
              type: choice
              instructions: Does the original task need escalation?
              choices: [{key: simple, description: Routine task}, {key: hard, description: Complex task}]
      decisions:
        - name: simple
          rules: {type: systemone, name: task_kind, label: simple}
          modelRefs: [{model: kai}]
          algorithm:
            type: cascade
            budget: {deadline: 150ms, max_calls: 2}
            quality: &quality
              type: uncalibrated
              acceptance:
                rules: [{question_type: noul, field: top_probability, predicate: {gte: 0}}]
            stages: [{name: answer, kind: native, model: kai}]
        - name: hard
          rules: {type: systemone, name: task_kind, label: hard}
          modelRefs: [{model: kai}, {model: vega}]
          algorithm:
            type: cascade
            budget: {deadline: 200ms, max_calls: 3}
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
		name        string
		decision    string
		deadline    time.Duration
		wantCalls   []string
		signalCalls int
		cancel      bool
		waitSignal  bool
	}{
		{name: "simple", decision: "simple", deadline: 150 * time.Millisecond, wantCalls: []string{"kai"}, signalCalls: 1},
		{name: "hard", decision: "hard", deadline: 200 * time.Millisecond, wantCalls: []string{"kai", "vega"}, signalCalls: 1},
		{name: "signals exhaust request budget", decision: "hard", signalCalls: 3},
		{name: "signals exhaust the selected decision budget", decision: "simple", deadline: 150 * time.Millisecond, signalCalls: 2},
		{name: "canceled", decision: "simple", signalCalls: 1, cancel: true},
		{name: "deadline during signal evaluation", decision: "hard", waitSignal: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			classifier.SetDecisionDecider(nativeRoutingDecider(func(signalCtx context.Context, deployment string, task modelservice.Request) (modelservice.Response, error) {
				if deployment != "local-kai" || !strings.Contains(task.State, `"questions"`) {
					t.Error("signal did not receive the original native task")
				}
				deadline, bounded := signalCtx.Deadline()
				if !bounded || time.Until(deadline) <= 0 || time.Until(deadline) > 200*time.Millisecond {
					t.Error("signals did not inherit the request deadline")
				}
				if tc.waitSignal {
					<-signalCtx.Done()
					return modelservice.Response{}, signalCtx.Err()
				}
				for range tc.signalCalls {
					if consumeErr := budget.Consume(signalCtx); consumeErr != nil {
						if tc.name != "signals exhaust request budget" {
							t.Error("signal unexpectedly exhausted request budget", consumeErr)
						}
					}
				}
				if tc.cancel {
					cancel()
					return modelservice.Response{}, context.Canceled
				}
				return modelservice.Response{Answers: map[string]modelservice.Answer{
					"systemone:task_kind": {Type: "choice", Choice: tc.decision, Probabilities: map[string]float64{"simple": 0.5, "hard": 0.5}},
				}}, nil
			}))
			var calls []string
			request := json.RawMessage(`{"model":"vllm-sr/auto","state":"an original task","questions":{"needs_reasoning":{"type":"noul","instructions":"Does this need reasoning?"}}}`)
			status, body, routeErr := router.RouteSystemOne(ctx, "vllm-sr/auto", request, func(callCtx context.Context, model string, original json.RawMessage) (int, []byte, error) {
				if string(original) != string(request) {
					t.Error("routing replaced the original native question")
				}
				if deadline, bounded := callCtx.Deadline(); !bounded || time.Until(deadline) > tc.deadline {
					t.Error("selected algorithm did not retain its request-relative deadline")
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
			if tc.waitSignal {
				var exhausted *systemone.BudgetExhaustedError
				if !errors.As(routeErr, &exhausted) || exhausted.Reason != "deadline" || status != http.StatusGatewayTimeout || len(calls) != 0 {
					t.Fatalf("signal deadline did not stop the request: status=%d calls=%v err=%v", status, calls, routeErr)
				}
				return
			}
			if strings.HasPrefix(tc.name, "signals exhaust") {
				var exhausted *systemone.BudgetExhaustedError
				if !errors.As(routeErr, &exhausted) || exhausted.Reason != "max_calls" || exhausted.Calls != tc.signalCalls ||
					status != http.StatusServiceUnavailable || len(calls) != 0 {
					t.Fatalf("signals did not stop at the applicable call budget: status=%d calls=%v err=%v", status, calls, routeErr)
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
			if json.Unmarshal(body, &response) != nil || response.Routing.Decision != tc.decision || response.Routing.Calls != len(tc.wantCalls)+tc.signalCalls {
				t.Fatalf("wrong decision or budget accounting: %s", body)
			}
		})
	}

	t.Run("deadline starts before public request body read", func(t *testing.T) {
		var signalCalls, backendCalls int
		classifier.SetDecisionDecider(nativeRoutingDecider(func(ctx context.Context, _ string, _ modelservice.Request) (modelservice.Response, error) {
			if err := budget.Consume(ctx); err != nil {
				return modelservice.Response{}, err
			}
			signalCalls++
			return modelservice.Response{}, nil
		}))
		listener := &cfg.Listeners[0]
		handler := systemone.Handler(cfg, listener, func(ctx context.Context, target string, body json.RawMessage) (int, []byte, error) {
			return router.RouteSystemOne(ctx, target, body, func(context.Context, string, json.RawMessage) (int, []byte, error) {
				backendCalls++
				return http.StatusOK, []byte(`{"answers":{}}`), nil
			})
		})
		request := httptest.NewRequest(http.MethodPost, "/v1/systemone", nil)
		request.Body = io.NopCloser(&delayedReader{reader: strings.NewReader(`{"model":"vllm-sr/auto","state":"original task","questions":{"needs_reasoning":{"type":"noul"}}}`), delay: 250 * time.Millisecond})
		response := httptest.NewRecorder()
		handler(response, request)
		if response.Code != http.StatusGatewayTimeout || signalCalls != 0 || backendCalls != 0 ||
			!strings.Contains(response.Body.String(), `"code":"systemone_budget_exhausted"`) ||
			!strings.Contains(response.Body.String(), `"model_calls":0`) {
			t.Fatalf("body read escaped the request deadline: status=%d signals=%d backend=%d body=%s", response.Code, signalCalls, backendCalls, response.Body.String())
		}
	})
}
