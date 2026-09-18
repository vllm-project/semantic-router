//go:build !windows && cgo

package apiserver

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

// Controlled typed handles verify HTTP->live generation->prepared inference.
// These are not quality or hardware tests of the published Vela artifacts.
func diagnosticTestHandle[I, O any](t *testing.T, runtime *native.Runtime, recipe, name, contract string, capability binding.Capability, infer func(I) (O, error)) *binding.Resolved[I, O] {
	t.Helper()
	return diagnosticTestContextHandle(t, runtime, recipe, name, contract, capability, func(_ context.Context, input I) (O, error) { return infer(input) })
}

func diagnosticTestContextHandle[I, O any](t *testing.T, runtime *native.Runtime, recipe, name, contract string, capability binding.Capability, infer func(context.Context, I) (O, error)) *binding.Resolved[I, O] {
	t.Helper()
	task, err := binding.Register(binding.NewRegistry(runtime.ObserveBinding), contract, func(I) error { return nil }, func(I, O) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	resource, err := runtime.Pool.Acquire(context.Background(), binding.ResourceIdentity{Artifact: "models/" + name, Revision: "prepared-weights", Provider: "test", Device: "cpu", Precision: "fp32"}, "", nil, func(context.Context) (io.Closer, error) { return io.NopCloser(strings.NewReader("")), nil })
	if err != nil {
		t.Fatal(err)
	}
	capability.Contract, capability.Provider, capability.Device, capability.Precision = contract, "test", "cpu", "fp32"
	handle, err := task.Resolve(binding.Identity{Recipe: recipe, Name: name, Deployment: "selected-deployment", Contract: contract, Adapter: "typed-test"}, capability, resource, func(ctx context.Context, _ io.Closer, input I) (O, error) { return infer(ctx, input) })
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handle.Close() })
	handle.Ready()
	return handle
}

func diagnosticRequest(t *testing.T, server *httptest.Server, path, body string) (int, []byte) {
	t.Helper()
	response, err := server.Client().Post(server.URL+apiDiagnosticsPath+"/models/"+path, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	data, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	return response.StatusCode, data
}

func TestModelDiagnosticsHTTPAllVelaTaskContractsAndExplicitScope(t *testing.T) {
	runtime := native.New(nil)
	cfg, service := preparedInventoryService(t, runtime, &config.RouterConfig{Recipes: []config.RoutingRecipe{{Name: config.DefaultRecipeName}, {Name: "private"}}})
	registry := routerruntime.NewRegistry(cfg)
	registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{Config: cfg, ClassificationService: service})
	server := httptest.NewServer((&ClassificationAPIServer{config: cfg, runtimeRegistry: registry}).setupRoutes())
	t.Cleanup(server.Close)
	var calls atomic.Int32
	usage := &tasks.InputUsage{OriginalTokens: 3, ProcessedTokens: 3}
	labels := []string{"first", "second"}
	for _, name := range []string{"domain_classifier", "prompt_guard", "feedback_detector", "fact_check_classifier", "modality_detector", "safety.unsafe"} {
		diagnosticTestHandle(t, runtime, "default", name, "label_distribution.v1", binding.Capability{Labels: labels}, func(input string) (tasks.LabelDistribution, error) {
			calls.Add(1)
			if input != "query" {
				t.Errorf("wrong input %q", input)
			}
			return tasks.LabelDistribution{Probabilities: []float32{.2, .8}, Input: usage}, nil
		})
		status, body := diagnosticRequest(t, server, "labels", fmt.Sprintf(`{"recipe":"default","binding":%q,"text":"query"}`, name))
		var response ModelDiagnosticResponse[ModelDiagnosticLabelsResult]
		if err := json.Unmarshal(body, &response); err != nil {
			t.Fatal(err)
		}
		if status != 200 || response.Binding.Name != name || response.Binding.Recipe != "default" || response.Binding.Revision != "prepared-weights" || response.Binding.Provider != "test" || !reflect.DeepEqual(response.Result.Probabilities, []float32{.2, .8}) {
			t.Fatalf("%s: %d %s", name, status, body)
		}
	}
	diagnosticTestHandle(t, runtime, "default", "classifier.risk", "label_scores.v1", binding.Capability{Labels: labels}, func(string) (tasks.LabelScores, error) {
		calls.Add(1)
		return tasks.LabelScores{Scores: []float32{.8, .9}, Input: usage}, nil
	})
	status, body := diagnosticRequest(t, server, "label-scores", `{"recipe":"default","binding":"classifier.risk","text":"query"}`)
	if status != 200 || !strings.Contains(string(body), `"scores":[0.8,0.9]`) {
		t.Fatalf("hazard: %d %s", status, body)
	}
	available := true
	diagnosticTestHandle(t, runtime, "default", "pii_classifier", "token_spans.v1", binding.Capability{Labels: []string{"O", "B-person"}}, func(input string) (tasks.TokenClassificationResult, error) {
		calls.Add(1)
		return tasks.TokenClassificationResult{Entities: []tasks.TokenEntity{{EntityType: "person", Start: 0, End: len(input), Text: input, Confidence: .9}}, ScoresAvailable: &available, Input: usage}, nil
	})
	status, body = diagnosticRequest(t, server, "tokens", `{"recipe":"default","binding":"pii_classifier","text":"张"}`)
	if status != 200 || !strings.Contains(string(body), `"end":3`) {
		t.Fatalf("pii byte offsets: %d %s", status, body)
	}
	diagnosticTestHandle(t, runtime, "default", "embedding", "embedding.v1", binding.Capability{Embedding: &binding.EmbeddingCapability{Dimension: 2, Layer: 7}}, func(input embedding.TextRequest) (tasks.EmbeddingResult, error) {
		calls.Add(1)
		if input.Options.Dimension != 2 || input.Options.Layer != 7 {
			t.Errorf("prepared representation lost: %+v", input)
		}
		return tasks.EmbeddingResult{Embedding: []float32{.3, .4}, Input: usage}, nil
	})
	status, body = diagnosticRequest(t, server, "embeddings", `{"recipe":"default","binding":"embedding","text":"query"}`)
	if status != 200 || !strings.Contains(string(body), `"embedding":[0.3,0.4]`) {
		t.Fatalf("embedding: %d %s", status, body)
	}
	diagnosticTestHandle(t, runtime, "default", "rag.reranker", "relevance_scores.v1", binding.Capability{}, func(input []tasks.QueryDocument) (tasks.RelevanceScores, error) {
		calls.Add(1)
		if len(input) != 2 || input[0].Document != "a" || input[1].Document != "b" {
			t.Errorf("pairs/order lost: %+v", input)
		}
		return tasks.RelevanceScores{Scores: []float32{-2, 7}, Inputs: []tasks.InputUsage{*usage, *usage}}, nil
	})
	status, body = diagnosticRequest(t, server, "rerank", `{"recipe":"default","binding":"rag.reranker","pairs":[{"query":"q","document":"a"},{"query":"q","document":"b"}]}`)
	if status != 200 || !strings.Contains(string(body), `"scores":[-2,7]`) || !strings.Contains(string(body), `"score_type":"relevance_logit"`) {
		t.Fatalf("reranker: %d %s", status, body)
	}
	if calls.Load() != 10 {
		t.Fatalf("inference calls = %d, want 10", calls.Load())
	}
	for _, bad := range []struct {
		body   string
		status int
	}{
		{`{"binding":"domain_classifier","text":"query"}`, 400},
		{`{"recipe":"foreign","binding":"domain_classifier","text":"query"}`, 404},
		{`{"recipe":"private","binding":"domain_classifier","text":"query"}`, 404},
		{`{"recipe":"default","binding":"unloaded","text":"query"}`, 404},
		{`{"recipe":"default","binding":"rag.reranker","text":"query"}`, 404},
		{`{"recipe":"default","binding":"domain_classifier","text":"query","artifact":"arbitrary"}`, 400},
	} {
		badStatus, badBody := diagnosticRequest(t, server, "labels", bad.body)
		if badStatus != bad.status {
			t.Fatalf("invalid scope/input %s: %d %s", bad.body, badStatus, badBody)
		}
	}
	if calls.Load() != 10 {
		t.Fatal("invalid request executed a model")
	}
	response, err := server.Client().Get(server.URL + apiDiagnosticsPath + "/models?recipe=default")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	var inventory ModelDiagnosticInventory
	if err = json.NewDecoder(response.Body).Decode(&inventory); err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != 200 || len(inventory.Bindings) != 10 {
		t.Fatalf("inventory = %+v", inventory)
	}
	// A recipe need not have a public model alias to be diagnosed. Sharing
	// weights/name with default still resolves its own prepared consumer.
	diagnosticTestHandle(t, runtime, "private", "domain_classifier", "label_distribution.v1", binding.Capability{Labels: labels}, func(string) (tasks.LabelDistribution, error) {
		return tasks.LabelDistribution{Probabilities: []float32{.7, .3}}, nil
	})
	status, body = diagnosticRequest(t, server, "labels", `{"recipe":"private","binding":"domain_classifier","text":"query"}`)
	if status != 200 || !strings.Contains(string(body), `"recipe":"private"`) || !strings.Contains(string(body), `"probabilities":[0.7,0.3]`) {
		t.Fatalf("named recipe fell back to default: %d %s", status, body)
	}
}

func TestModelDiagnosticsHTTPUsesPreparedWindowGeometry(t *testing.T) {
	runtime := native.New(nil)
	cfg, service := preparedInventoryService(t, runtime)
	server := httptest.NewServer((&ClassificationAPIServer{config: cfg, classificationSvc: service}).setupRoutes())
	t.Cleanup(server.Close)
	check := func(input tasks.TextWindowsRequest) {
		if input.Text != "document" || input.Size != 5 || input.Overlap != 1 {
			t.Errorf("diagnostics changed scan policy: %+v", input)
		}
	}
	capability := binding.Capability{Window: &binding.WindowCapability{Size: 5, Overlap: 1}, Labels: []string{"safe", "unsafe"}}
	usage := &tasks.InputUsage{OriginalTokens: 8, ProcessedTokens: 8}
	diagnosticTestHandle(t, runtime, "default", "prompt_guard", "label_distribution.v1", capability, func(input tasks.TextWindowsRequest) (tasks.WindowedLabelDistribution, error) {
		check(input)
		return tasks.WindowedLabelDistribution{Windows: []tasks.LabelDistributionWindow{{Start: 0, End: 3, Probabilities: []float32{.9, .1}}, {Start: 2, End: 6, Probabilities: []float32{.1, .9}}}, ContentTokens: 6, Input: usage}, nil
	})
	diagnosticTestHandle(t, runtime, "default", "pii_classifier", "token_spans.v1", capability, func(input tasks.TextWindowsRequest) (tasks.WindowedTokenClassification, error) {
		check(input)
		return tasks.WindowedTokenClassification{Result: tasks.TokenClassificationResult{Input: usage}, Windows: [][2]int{{0, 3}, {2, 6}}, ContentTokens: 6}, nil
	})
	diagnosticTestHandle(t, runtime, "default", "classifier.risk", "label_scores.v1", capability, func(input tasks.TextWindowsRequest) (tasks.WindowedLabelScores, error) {
		check(input)
		return tasks.WindowedLabelScores{Windows: []tasks.LabelScoresWindow{{Start: 0, End: 3, Scores: []float32{.9, .8}}}, ContentTokens: 3, Input: usage}, nil
	})
	for _, tc := range []struct{ path, name string }{{"labels", "prompt_guard"}, {"tokens", "pii_classifier"}, {"label-scores", "classifier.risk"}} {
		status, body := diagnosticRequest(t, server, tc.path, fmt.Sprintf(`{"recipe":"default","binding":%q,"text":"document"}`, tc.name))
		if status != 200 || !strings.Contains(string(body), `"window":{"size":5,"overlap":1}`) || !strings.Contains(string(body), `"windows":`) {
			t.Fatalf("window policy %s: %d %s", tc.name, status, body)
		}
	}
}

func TestModelDiagnosticsHoldsLiveGenerationThroughInferenceAndUsesReplacement(t *testing.T) {
	oldRuntime, nextRuntime := native.New(nil), native.New(nil)
	cfg, oldService := preparedInventoryService(t, oldRuntime)
	_, nextService := preparedInventoryService(t, nextRuntime)
	entered, finish := make(chan struct{}), make(chan struct{})
	var leases atomic.Int32
	diagnosticTestHandle(t, oldRuntime, "default", "domain_classifier", "label_distribution.v1", binding.Capability{Labels: []string{"a", "b"}}, func(string) (tasks.LabelDistribution, error) {
		close(entered)
		<-finish
		if leases.Load() != 1 {
			t.Error("old generation released before inference finished")
		}
		return tasks.LabelDistribution{Probabilities: []float32{.9, .1}}, nil
	})
	diagnosticTestHandle(t, nextRuntime, "default", "domain_classifier", "label_distribution.v1", binding.Capability{Labels: []string{"a", "b"}}, func(string) (tasks.LabelDistribution, error) {
		return tasks.LabelDistribution{Probabilities: []float32{.1, .9}}, nil
	})
	registry := routerruntime.NewRegistry(cfg)
	registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{Config: cfg, ClassificationService: oldService, AcquireClassification: func() (func(), bool) {
		leases.Add(1)
		return func() { leases.Add(-1) }, true
	}})
	server := httptest.NewServer((&ClassificationAPIServer{config: cfg, runtimeRegistry: registry}).setupRoutes())
	t.Cleanup(server.Close)
	result := make(chan []byte, 1)
	go func() {
		_, body := diagnosticRequest(t, server, "labels", `{"recipe":"default","binding":"domain_classifier","text":"query"}`)
		result <- body
	}()
	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		close(finish)
		t.Fatal("old model did not start")
	}
	registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{Config: cfg, ClassificationService: nextService})
	close(finish)
	select {
	case body := <-result:
		if !strings.Contains(string(body), `"probabilities":[0.9,0.1]`) {
			t.Fatalf("mixed generation: %s", body)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("old inference did not finish")
	}
	if leases.Load() != 0 {
		t.Fatal("generation lease leaked")
	}
	status, body := diagnosticRequest(t, server, "labels", `{"recipe":"default","binding":"domain_classifier","text":"query"}`)
	if status != 200 || !strings.Contains(string(body), `"probabilities":[0.1,0.9]`) {
		t.Fatalf("replacement not used: %d %s", status, body)
	}
}

func TestModelDiagnosticsRerankBoundsBatchUsingLeasedConfig(t *testing.T) {
	for _, tc := range []struct {
		name                 string
		limit, count, status int
	}{{"configured", 2, 3, 400}, {"at limit", 2, 2, 200}, {"default", 0, 101, 400}} {
		t.Run(tc.name, func(t *testing.T) {
			runtime := native.New(nil)
			cfg, service := preparedInventoryService(t, runtime)
			cfg.API.BatchClassification.MaxBatchSize = tc.limit
			registry := routerruntime.NewRegistry(cfg)
			registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{Config: cfg, ClassificationService: service})
			stale := &config.RouterConfig{}
			stale.API.BatchClassification.MaxBatchSize = 1
			api := &ClassificationAPIServer{config: stale, runtimeRegistry: registry}
			calls := 0
			diagnosticTestHandle(t, runtime, "default", "rag.reranker", "relevance_scores.v1", binding.Capability{}, func(pairs []tasks.QueryDocument) (tasks.RelevanceScores, error) {
				calls++
				return tasks.RelevanceScores{Scores: make([]float32, len(pairs))}, nil
			})
			pairs := make([]ModelRerankPair, tc.count)
			for i := range pairs {
				pairs[i] = ModelRerankPair{Query: "q", Document: "d"}
			}
			body, _ := json.Marshal(ModelRerankDiagnosticRequest{ModelDiagnosticTarget: ModelDiagnosticTarget{Recipe: "default", Binding: "rag.reranker"}, Pairs: pairs})
			response := httptest.NewRecorder()
			api.handleModelDiagnosticRerank(response, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(string(body))))
			if response.Code != tc.status || (calls > 0) != (tc.status == 200) {
				t.Fatalf("status=%d calls=%d body=%s", response.Code, calls, response.Body.String())
			}
		})
	}
}

func TestModelDiagnosticsDeadlineRetainsGenerationUntilNativeReturn(t *testing.T) {
	runtime := native.New(nil)
	cfg, service := preparedInventoryService(t, runtime)
	registry := routerruntime.NewRegistry(cfg)
	var leases atomic.Int32
	registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{Config: cfg, ClassificationService: service, AcquireClassification: func() (func(), bool) { leases.Add(1); return func() { leases.Add(-1) }, true }})
	api := &ClassificationAPIServer{config: cfg, runtimeRegistry: registry}
	entered, expired, finish := make(chan struct{}), make(chan struct{}), make(chan struct{})
	diagnosticTestContextHandle(t, runtime, "default", "domain_classifier", "label_distribution.v1", binding.Capability{}, func(ctx context.Context, text string) (tasks.LabelDistribution, error) {
		deadline, ok := ctx.Deadline()
		if !ok || time.Until(deadline) > apiWriteTimeout {
			t.Error("API did not bound the diagnostic inference context")
		}
		if text == "blocked" {
			close(entered)
			<-ctx.Done()
			close(expired)
			<-finish
			if leases.Load() != 1 {
				t.Error("deadline released live generation while native inference was running")
			}
		}
		return tasks.LabelDistribution{Probabilities: []float32{1}}, nil
	})
	bounded := httptest.NewRecorder()
	api.handleModelDiagnosticLabels(bounded, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"recipe":"default","binding":"domain_classifier","text":"bound"}`)))
	if bounded.Code != 200 {
		t.Fatalf("bounded request: %d %s", bounded.Code, bounded.Body.String())
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	request := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"recipe":"default","binding":"domain_classifier","text":"blocked"}`)).WithContext(ctx)
	response := httptest.NewRecorder()
	done := make(chan struct{})
	go func() { api.handleModelDiagnosticLabels(response, request); close(done) }()
	select {
	case <-entered:
	case <-time.After(time.Second):
		close(finish)
		t.Fatal("inference did not begin")
	}
	select {
	case <-expired:
	case <-time.After(time.Second):
		close(finish)
		t.Fatal("inference never received deadline")
	}
	if leases.Load() != 1 {
		close(finish)
		t.Fatal("cancellation released the generation early")
	}

	select {
	case <-done:
	case <-time.After(time.Second):
		close(finish)
		t.Fatal("HTTP response did not respect request deadline")
	}
	if response.Code != 503 || leases.Load() != 1 {
		close(finish)
		t.Fatalf("deadline status=%d leases=%d body=%s", response.Code, leases.Load(), response.Body.String())
	}
	close(finish)
	until := time.Now().Add(time.Second)
	for leases.Load() != 0 && time.Now().Before(until) {
		time.Sleep(time.Millisecond)
	}
	if leases.Load() != 0 {
		t.Fatal("completed inference retained its generation")
	}
}
