package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
		fmt.Fprintf(w, `{"model":%q,"answers":{"task":{"type":"choice","choice":"chat","probabilities":{"chat":%g,"code":%g},"confidence":0.5}},"usage":{"input_tokens":10,"output_tokens":0},"meta":{"profile":"exact"}}`, request.Model, probability, 1-probability)
	}))
	defer backend.Close()
	cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(`version: v0.3
listeners:
  - name: native
    port: 8801
    systemone: {models: [vllm-sr/auto]}
providers:
  models:
    - name: fast
      api_format: systemone
      provider_model_id: fast-native
      backend_refs: [{provider: systemone-compatible, base_url: %s/v1}]
    - name: strong
      api_format: systemone
      provider_model_id: strong-native
      backend_refs: [{provider: systemone-compatible, base_url: %s/v1}]
routing: {}
entrypoints:
  - {api: systemone, model_names: [vllm-sr/auto], recipe: native}
recipes:
  - name: native
    routing:
      signals:
        keywords:
          - {name: greeting, operator: OR, keywords: [hello]}
      decisions:
        - name: greeting
          rules: {type: keyword, name: greeting}
          modelRefs: [{model: fast}, {model: strong}]
          algorithm:
            type: cascade
            budget: {deadline: 2s, max_calls: 2}
            quality:
              type: uncalibrated
              acceptance:
                rules: [{question_type: choice, field: top_probability, predicate: {gte: 0}}]
            stages:
              - name: fast
                kind: native
                model: fast
                accept:
                  rules: [{question_type: choice, field: top_probability, predicate: {gte: 0.9}}]
              - {name: strong, kind: native, model: strong}
`, backend.URL, backend.URL)))
	if err != nil {
		t.Fatal(err)
	}
	classifiers, err := classification.BuildRecipeClassifiers(cfg, nil, nil, nil)
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
