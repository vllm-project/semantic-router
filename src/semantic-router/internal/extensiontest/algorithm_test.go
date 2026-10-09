package extensiontest

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configschema"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

const pickType = "test_pick"

// pickAlgorithm chooses the candidate its block names.
type pickAlgorithm struct {
	Pick string `json:"pick"`
}

func (p *pickAlgorithm) Select(_ context.Context, selCtx *selection.SelectionContext) (*selection.SelectionResult, error) {
	for i := range selCtx.CandidateModels {
		if candidate := selCtx.CandidateModels[i]; candidate.Model == p.Pick {
			return &selection.SelectionResult{SelectedModel: candidate.Model, SelectedCandidate: &candidate, Confidence: 1}, nil
		}
	}
	return nil, errors.New("the picked model is not a candidate")
}

func init() {
	if err := config.RegisterDecisionAlgorithm(config.NewDecisionAlgorithmType(
		config.AlgorithmCatalogEntry{Type: pickType, DisplayName: "Test Pick", Description: "Choose the named candidate.", Tier: "experimental"},
		config.AlgorithmOptions[pickAlgorithm]{
			Strict: true,
			Validate: func(_ string, p *pickAlgorithm) error {
				if p.Pick == "" {
					return errors.New("pick is required")
				}
				return nil
			},
		},
	)); err != nil {
		panic(err)
	}
}

func pickDocument(backend, block string) string {
	host := strings.TrimPrefix(backend, "http://")
	return `version: v0.3
listeners:
  - name: http-8899
    address: 127.0.0.1
    port: 8899
providers:
  defaults:
    model: a
  models:
    - name: a
      backend_refs:
        - endpoint: ` + host + `
          protocol: http
          provider: vllm
    - name: b
      backend_refs:
        - endpoint: ` + host + `
          protocol: http
          provider: vllm
global:
  stores:
    semantic_cache:
      enabled: false
routing:
  modelCards:
    - name: a
    - name: b
  signals:
    keywords:
      - name: greeting
        operator: OR
        keywords: ["hello"]
  decisions:
    - name: greet
      priority: 10
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: greeting
      modelRefs:
        - model: a
          use_reasoning: false
        - model: b
          use_reasoning: false
      algorithm:
        type: ` + pickType + `
` + block
}

func TestARegisteredAlgorithmChoosesTheCandidate(t *testing.T) {
	var mu sync.Mutex
	var selected string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		mu.Lock()
		selected = r.Header.Get("X-Selected-Model")
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, completion)
	}))
	defer backend.Close()
	router, err := extproc.NewOpenAIRouter(writeDocument(t, pickDocument(backend.URL, "        "+pickType+":\n          pick: b\n")))
	if err != nil {
		t.Fatal(err)
	}
	defer router.Close()
	set, err := upstream.Build(router.Config, upstream.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = set.Close(ctx)
	}()
	engine := routing.DefaultOptions
	engine.ExecutesFallback = true
	handler, err := gateway.NewHandler(gateway.Options{
		Serving:  gateway.Static(gateway.Serving{Engine: routing.NewEngine(extproc.NewRouterService(router), engine), Upstream: set}),
		Listener: "http-8899",
	})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(handler)
	defer front.Close()
	resp, err := http.Post(front.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"vllm-sr/auto","messages":[{"role":"user","content":"hello there"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
	mu.Lock()
	defer mu.Unlock()
	if resp.StatusCode != http.StatusOK || selected != "b" {
		t.Fatalf("status %d, the backend served %q; want the picked candidate b", resp.StatusCode, selected)
	}
}

func TestARegisteredAlgorithmOwnsItsBlock(t *testing.T) {
	for name, block := range map[string]string{
		"validator":     "        " + pickType + ":\n          pick: \"\"\n",
		"strict schema": "        " + pickType + ":\n          pick: b\n          colour: red\n",
		"unknown block": "        test_pik:\n          pick: b\n",
	} {
		_, err := config.ParseYAMLBytes([]byte(pickDocument("http://127.0.0.1:1", block)))
		if err == nil {
			t.Fatalf("%s: the configuration loaded", name)
		}
		if !strings.Contains(err.Error(), "test_pi") {
			t.Fatalf("%s: the error does not name the block: %v", name, err)
		}
	}
	cfg, err := config.ParseYAMLBytes([]byte(pickDocument("http://127.0.0.1:1", "        "+pickType+":\n          pick: b\n")))
	if err != nil {
		t.Fatal(err)
	}
	exported, err := yaml.Marshal(config.CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatal(err)
	}
	reloaded, err := config.ParseYAMLBytes(exported)
	if err != nil {
		t.Fatalf("the exported document does not load: %v", err)
	}
	payload, registered, err := config.DecodeDecisionAlgorithm("greet", reloaded.Decisions[0].Algorithm)
	if err != nil || !registered || payload.(*pickAlgorithm).Pick != "b" {
		t.Fatalf("after export and reload: %+v, %v, %v", payload, registered, err)
	}
	schema, err := configschema.GenerateFromSource("../../../..")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(schema), `"`+pickType+`"`) {
		t.Fatal("the generated schema does not hold the registered algorithm's block")
	}
	retired := config.NewDecisionAlgorithmType(config.AlgorithmCatalogEntry{Type: "elo"}, config.AlgorithmOptions[pickAlgorithm]{})
	if err := config.RegisterDecisionAlgorithm(retired); err == nil {
		t.Fatal("a retired algorithm type registered again")
	}
}
