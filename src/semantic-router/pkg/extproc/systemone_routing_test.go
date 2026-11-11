package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

func TestSystemOneAutoUsesRecipeSignalsAndNativeBackends(t *testing.T) {
	var called []string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/systemone" {
			t.Error("native task entered Chat transport")
		}

		var request struct {
			Model string `json:"model"`
			State string `json:"state"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		called = append(called, request.Model)
		if request.State != "hello" {
			t.Error("routing changed the native state")
		}
		probability := 0.6
		if request.Model == "strong-native" {
			probability = 0.99
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"model":%q,"answers":{"task":{"type":"choice","choice":"chat","probabilities":{"chat":%g,"code":%g},"confidence":0.5}},"usage":{"input_tokens":10,"output_tokens":0},"meta":{"dtype":"float32"}}`, request.Model, probability, 1-probability)
	}))
	defer backend.Close()
	cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(`version: v0.5
listeners:
  - name: native
    port: 8801
    apis: {chat_completions: {}, responses: {}, messages: {}, images: {}, systemone: {schema: vllm-sr/4.0, models: [vllm-sr/auto]}}
providers:
  models:
    - name: fast
      api_format: systemone
      provider_model_id: fast-native
      backend_refs: [{provider: systemone-compatible, endpoint: %s/v1}]
    - name: strong
      api_format: systemone
      provider_model_id: strong-native
      backend_refs: [{provider: systemone-compatible, endpoint: %s/v1}]
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
  - {api: systemone, model_names: [vllm-sr/auto], recipe: native}
recipes:
  - name: default
  - name: native
    signals:
      keywords:
        - {name: greeting, operator: OR, keywords: [hello]}
    decisions:
      - name: greeting
        rules: {type: keyword, name: greeting}
        modelRefs: [{model: fast}, {model: strong}]
        algorithm:
          type: escalate
          budget: {deadline: 2s, max_calls: 2}
          escalate:
            quality:
              type: uncalibrated
              accept:
                native: [{question_type: choice, field: top_probability, gte: 0}]
            stages:
              - name: fast
                model: fast
                accept:
                  native: [{question_type: choice, field: top_probability, gte: 0.9}]
              - {name: strong, model: strong}
`, backend.URL, backend.URL)))
	if err != nil {
		t.Fatal(err)
	}
	classifiers, err := classification.BuildRecipeClassifiers(cfg)
	if err != nil {
		t.Fatal(err)
	}
	executors, err := prepareNativeExecutors(cfg)
	if err != nil {
		t.Fatal(err)
	}
	router := &OpenAIRouter{Config: cfg, RecipeClassifiers: classifiers, nativeExecutors: executors}
	remote, err := upstream.Build(cfg, upstream.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer remote.Close(context.Background())
	listener := &cfg.Listeners[0]
	handler := systemone.Handler(cfg, listener, systemone.ServingInvoker(cfg, listener.Name, router, nil, remote))
	body := `{"model":"vllm-sr/auto","state":"hello","questions":{"task":{"type":"choice","instructions":"Type?","criteria":{"chat":"Chat","code":"Code"}}}}`
	r := httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(body))
	w := httptest.NewRecorder()
	handler(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("native auto status=%d body=%s", w.Code, w.Body.String())
	}
	if !reflect.DeepEqual(called, []string{"fast-native", "strong-native"}) {
		t.Fatalf("backend calls=%v", called)
	}
	var response struct {
		Model   string `json:"model"`
		Routing struct {
			Recipe   string `json:"recipe"`
			Decision string `json:"decision"`
			Calls    int    `json:"model_calls"`
		} `json:"routing"`
	}
	if json.Unmarshal(w.Body.Bytes(), &response) != nil || response.Model != "vllm-sr/auto" || response.Routing.Recipe != "native" || response.Routing.Decision != "greeting" || response.Routing.Calls != 2 {
		t.Fatalf("routing metadata=%s", w.Body.String())
	}
	if strings.Contains(w.Body.String(), `"meta"`) {
		t.Fatal("internal provenance exposed without return_meta")
	}
	for _, returnMeta := range []bool{false, true} {
		withOptions := strings.TrimSuffix(body, "}") + fmt.Sprintf(`,"options":{"return_meta":%t}}`, returnMeta)
		r = httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(withOptions))
		w = httptest.NewRecorder()
		handler(w, r)
		if w.Code != http.StatusOK || strings.Contains(w.Body.String(), `"meta"`) != returnMeta {
			t.Fatalf("metadata visibility disagrees with client: %d %s", w.Code, w.Body.String())
		}
	}
	called = nil
	r = httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(strings.Replace(body, "hello", "goodbye", 1)))
	w = httptest.NewRecorder()
	handler(w, r)
	if w.Code != http.StatusServiceUnavailable || len(called) != 0 {
		t.Fatal("unmatched native decision used a Chat fallback")
	}
}

func TestSystemOneBudgetIncludesRemoteSignalAndTransportRetries(t *testing.T) {
	var signalCalls, fastAttempts, strongAttempts atomic.Int64
	signalMetricBefore := testutil.ToFloat64(metrics.SystemOneSignalCalls)
	stageMetricBefore := testutil.ToFloat64(metrics.SystemOneStageCalls.WithLabelValues("escalate", "fast", "fast"))
	decisionServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/systemone/models" {
			http.NotFound(w, r)
			return
		}
		if r.URL.Path != "/v1/systemone" {
			t.Errorf("remote decision used unexpected path %q", r.URL.Path)
		}
		var request struct {
			Questions map[string]json.RawMessage `json:"questions"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
		}
		if len(request.Questions) != 1 {
			t.Errorf("remote decision questions=%v", request.Questions)
		}
		signalCalls.Add(1)
		_ = json.NewEncoder(w).Encode(map[string]any{"answers": map[string]any{
			"systemone:task_kind": map[string]any{"type": "choice", "choice": "simple", "probabilities": map[string]float64{"simple": 1, "hard": 0}},
		}})
	}))
	defer decisionServer.Close()
	backendServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode backend request: %v", err)
			return
		}
		if request.Model == "fast-native" {
			if fastAttempts.Add(1) == 1 {
				http.Error(w, "retry", http.StatusInternalServerError)
				return
			}
		} else {
			strongAttempts.Add(1)
		}
		probability := 0.6
		if request.Model == "strong-native" {
			probability = 0.99
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": request.Model,
			"answers": map[string]any{
				"task": map[string]any{"type": "choice", "choice": "chat", "probabilities": map[string]float64{"chat": probability, "code": 1 - probability}},
			},
		})
	}))
	defer backendServer.Close()
	cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(`version: v0.5
listeners:
  - name: native
    port: 8801
    apis: {chat_completions: {}, responses: {}, messages: {}, images: {}, systemone: {schema: vllm-sr/4.0, models: [vllm-sr/auto]}}
providers:
  models:
    - name: fast
      api_format: systemone
      provider_model_id: fast-native
      reliability: {retry_count: 1, retry_on: 5xx, retriable_status_codes: [500]}
      backend_refs: [{provider: systemone-compatible, endpoint: %s/v1}]
    - name: strong
      api_format: systemone
      provider_model_id: strong-native
      backend_refs: [{provider: systemone-compatible, endpoint: %s/v1}]
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
  - {api: systemone, model_names: [vllm-sr/auto], recipe: native}
global:
  model_catalog:
    deployments:
      remote-kai: {provider: systemone, schema: vllm-sr/4.0, endpoint: %s/v1/systemone, model: vllm-sr/decision}
recipes:
  - name: default
  - name: native
    signals:
      systemone:
        - name: task_kind
          deployment: remote-kai
          timeout_ms: 1000
          question:
            type: choice
            instructions: Is the task simple?
            choices: [{key: simple}, {key: hard}]
    decisions:
      - name: simple
        rules: {type: systemone, name: task_kind, label: simple}
        modelRefs: [{model: fast}, {model: strong}]
        algorithm:
          type: escalate
          budget: {deadline: 2s, max_calls: 3}
          escalate:
            quality:
              type: uncalibrated
              accept:
                native: [{question_type: choice, field: top_probability, gte: 0.9}]
            stages:
              - name: fast
                model: fast
                accept:
                  native: [{question_type: choice, field: top_probability, gte: 0.9}]
              - {name: strong, model: strong}
`, backendServer.URL, backendServer.URL, decisionServer.URL)))
	if err != nil {
		t.Fatal(err)
	}
	classifiers, err := classification.BuildRecipeClassifiers(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer classifiers.Close()
	classifier, ok := classifiers.ForRecipe("native")
	if !ok {
		t.Fatal("native recipe classifier unavailable")
	}
	remoteSignals, err := modelservice.NewRemoteSystemOne(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer remoteSignals.Close()
	classifier.SetSystemOneDecider(remoteSignals)
	executors, err := prepareNativeExecutors(cfg)
	if err != nil {
		t.Fatal(err)
	}
	router := &OpenAIRouter{Config: cfg, RecipeClassifiers: classifiers, nativeExecutors: executors}
	remote, err := upstream.Build(cfg, upstream.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer remote.Close(context.Background())
	handler := systemone.Handler(cfg, &cfg.Listeners[0], systemone.ServingInvoker(cfg, "native", router, nil, remote))
	body := `{"model":"vllm-sr/auto","state":"hello","questions":{"task":{"type":"choice","instructions":"Type?","criteria":{"chat":"Chat","code":"Code"}}}}`
	request := httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(body))
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusServiceUnavailable || signalCalls.Load() != 1 || fastAttempts.Load() != 2 || strongAttempts.Load() != 0 ||
		!strings.Contains(response.Body.String(), `"code":"systemone_budget_exhausted"`) {
		t.Fatalf("status=%d signal calls=%d fast attempts=%d strong attempts=%d body=%s", response.Code, signalCalls.Load(), fastAttempts.Load(), strongAttempts.Load(), response.Body.String())
	}
	var result struct {
		Error struct {
			Calls int `json:"model_calls"`
		} `json:"error"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil || result.Error.Calls != int(signalCalls.Load()+fastAttempts.Load()+strongAttempts.Load()) {
		t.Fatalf("physical attempts missing from response accounting: %s", response.Body.String())
	}
	if got := testutil.ToFloat64(metrics.SystemOneSignalCalls) - signalMetricBefore; got != 1 {
		t.Fatalf("signal telemetry=%v, want 1 physical exchange", got)
	}
	if got := testutil.ToFloat64(metrics.SystemOneStageCalls.WithLabelValues("escalate", "fast", "fast")) - stageMetricBefore; got != 2 {
		t.Fatalf("stage telemetry=%v, want 2 physical exchanges including retry", got)
	}
	t.Logf("HTTP %d: model_calls=%d signal_calls=%d fast_attempts=%d strong_attempts=%d",
		response.Code, result.Error.Calls, signalCalls.Load(), fastAttempts.Load(), strongAttempts.Load())
}

func TestSystemOneBudgetDeadlineIncludesRemoteSignal(t *testing.T) {
	var signalCalls atomic.Int64
	decisionServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/systemone/models" {
			http.NotFound(w, r)
			return
		}
		signalCalls.Add(1)
		time.Sleep(250 * time.Millisecond)
		_ = json.NewEncoder(w).Encode(map[string]any{"answers": map[string]any{
			"systemone:task_kind": map[string]any{"type": "choice", "choice": "simple", "probabilities": map[string]float64{"simple": 1, "hard": 0}},
		}})
	}))
	defer decisionServer.Close()
	cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(`version: v0.5
listeners:
  - name: native
    port: 8801
    apis: {chat_completions: {}, responses: {}, messages: {}, images: {}, systemone: {schema: vllm-sr/4.0, models: [vllm-sr/auto]}}
providers:
  models:
    - name: fast
      api_format: systemone
      provider_model_id: fast-native
      backend_refs: [{provider: systemone-compatible, endpoint: 'http://unused.invalid/v1'}]
global:
  model_catalog:
    deployments:
      remote-kai: {provider: systemone, schema: vllm-sr/4.0, endpoint: %s/v1/systemone, model: vllm-sr/decision}
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
  - {api: systemone, model_names: [vllm-sr/auto], recipe: native}
recipes:
  - name: default
  - name: native
    signals:
      systemone:
        - name: task_kind
          deployment: remote-kai
          timeout_ms: 1000
          question:
            type: choice
            instructions: Is the task simple?
            choices: [{key: simple}, {key: hard}]
    decisions:
      - name: simple
        rules: {type: systemone, name: task_kind, label: simple}
        modelRefs: [{model: fast}]
        algorithm:
          type: escalate
          budget: {deadline: 100ms, max_calls: 2}
          escalate:
            quality:
              type: uncalibrated
              accept:
                native: [{question_type: noul, field: top_probability, gte: 0}]
            stages: [{name: answer, model: fast}]
`, decisionServer.URL)))
	if err != nil {
		t.Fatal(err)
	}
	classifiers, err := classification.BuildRecipeClassifiers(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer classifiers.Close()
	classifier, ok := classifiers.ForRecipe("native")
	if !ok {
		t.Fatal("native recipe classifier unavailable")
	}
	remoteSignals, err := modelservice.NewRemoteSystemOne(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer remoteSignals.Close()
	classifier.SetSystemOneDecider(remoteSignals)
	executors, err := prepareNativeExecutors(cfg)
	if err != nil {
		t.Fatal(err)
	}
	router := &OpenAIRouter{Config: cfg, RecipeClassifiers: classifiers, nativeExecutors: executors}
	var backendCalls atomic.Int64
	body := json.RawMessage(`{"model":"vllm-sr/auto","state":"hello","questions":{"task":{"type":"noul"}}}`)
	status, _, routeErr := router.RouteSystemOne(context.Background(), "vllm-sr/auto", body, func(context.Context, string, json.RawMessage) (int, []byte, error) {
		backendCalls.Add(1)
		return http.StatusOK, []byte(`{"answers":{"task":{"type":"noul","noul":0.9}}}`), nil
	})
	var exhausted *systemone.BudgetExhaustedError
	if !errors.As(routeErr, &exhausted) || exhausted.Reason != "deadline" || exhausted.Calls != 1 ||
		status != http.StatusGatewayTimeout || signalCalls.Load() != 1 || backendCalls.Load() != 0 {
		t.Fatalf("status=%d signal_calls=%d backend_calls=%d err=%v", status, signalCalls.Load(), backendCalls.Load(), routeErr)
	}
	t.Logf("HTTP %d: reason=%s model_calls=%d signal_calls=%d backend_calls=%d",
		status, exhausted.Reason, exhausted.Calls, signalCalls.Load(), backendCalls.Load())
}
