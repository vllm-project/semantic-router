package classification

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestCategoryHTTPBackendPreservesNamedFullDistribution(t *testing.T) {
	mapping := &CategoryMapping{
		CategoryToIdx: map[string]int{"math": 0, "physics": 1, "other": 2},
		IdxToCategory: map[string]string{"0": "math", "1": "physics", "2": "other"},
	}
	server := httptest.NewServer(categoryDistributionHandler(t, "solve this"))
	defer server.Close()

	deadline := 1500
	backend, err := newCategoryHTTPBackend(&config.ExternalModelConfig{
		ModelEndpoint: config.ClassifierVLLMEndpoint{Address: "127.0.0.1", Port: 8080},
		ModelName:     "named-category-service",
	}, mapping, time.Duration(deadline)*time.Millisecond)
	if err != nil {
		t.Fatalf("newCategoryHTTPBackend: %v", err)
	}
	backend.(*categoryHTTPBackend).backend.(*HTTPClassifierInference).baseURL = server.URL

	result, err := backend.ClassifyWithProbabilities(context.Background(), "solve this")
	if err != nil {
		t.Fatalf("ClassifyWithProbabilities: %v", err)
	}
	want := []float32{0.50, 0.40, 0.10}
	assertProbabilities(t, result.Probabilities, want)
	if result.Class != 0 || result.Confidence != 0.5 {
		t.Fatalf("argmax result = %#v, want class 0 confidence .5", result)
	}
}

func TestCategoryHTTPBackendPreservesCallerCancellation(t *testing.T) {
	requestStarted := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(requestStarted)
		// The delayed response makes the test fail if the adapter replaces the
		// caller's context with context.Background(): the request would then
		// complete successfully instead of returning context.Canceled.
		time.Sleep(200 * time.Millisecond)
		_, _ = w.Write([]byte(`[{"label":"math","score":0.75},{"label":"other","score":0.25}]`))
	}))
	defer server.Close()

	mapping := &CategoryMapping{
		CategoryToIdx: map[string]int{"math": 0, "other": 1},
		IdxToCategory: map[string]string{"0": "math", "1": "other"},
	}
	backend, err := newCategoryHTTPBackend(&config.ExternalModelConfig{
		ModelEndpoint: config.ClassifierVLLMEndpoint{Address: "127.0.0.1", Port: 8080},
		ModelName:     "named-category-service",
	}, mapping, time.Second)
	if err != nil {
		t.Fatalf("newCategoryHTTPBackend: %v", err)
	}
	backend.(*categoryHTTPBackend).backend.(*HTTPClassifierInference).baseURL = server.URL

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errCh := make(chan error, 1)
	go func() {
		_, classifyErr := backend.ClassifyWithProbabilities(ctx, "cancel this")
		errCh <- classifyErr
	}()
	select {
	case <-requestStarted:
	case <-time.After(time.Second):
		t.Fatal("HTTP request did not reach the test server")
	}
	cancel()

	select {
	case err := <-errCh:
		if err == nil || !errors.Is(err, context.Canceled) {
			t.Fatalf("classification error = %v, want context.Canceled", err)
		}
	case <-time.After(time.Second):
		t.Fatal("classification did not stop after caller cancellation")
	}
}

func TestCategoryHTTPBackendDoesNotUseTop1FallbackOnDistributionError(t *testing.T) {
	backend := &categoryHTTPBackend{}
	if categoryProbabilityFallbackAllowed(backend) {
		t.Fatal("remote category backend must not retry a failed distribution request through Classify")
	}
	if !categoryProbabilityFallbackAllowed(&MockCategoryInference{}) {
		t.Fatal("legacy category inference should retain the top1 fallback")
	}
}

func TestCategoryHTTPBackendDistributionFailureMakesOneRequest(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		http.Error(w, "temporary classifier failure", http.StatusBadGateway)
	}))
	defer server.Close()

	mapping := &CategoryMapping{
		CategoryToIdx: map[string]int{"math": 0, "other": 1},
		IdxToCategory: map[string]string{"0": "math", "1": "other"},
	}
	backend, err := newCategoryHTTPBackend(&config.ExternalModelConfig{
		ModelEndpoint: config.ClassifierVLLMEndpoint{Address: "127.0.0.1", Port: 8080},
		ModelName:     "named-category-service",
	}, mapping, time.Second)
	if err != nil {
		t.Fatalf("newCategoryHTTPBackend: %v", err)
	}
	backend.(*categoryHTTPBackend).backend.(*HTTPClassifierInference).baseURL = server.URL

	classifier := &Classifier{
		Config: &config.RouterConfig{InlineModels: config.InlineModels{Classifier: config.Classifier{
			CategoryModel: config.CategoryModel{},
		}}},
		CategoryMapping:   mapping,
		categoryInference: backend,
	}
	results := &SignalResults{
		SignalConfidences: map[string]float64{},
		SignalErrors:      map[string]string{},
		Metrics:           &SignalMetricsCollection{},
	}
	classifier.evaluateDomainSignal(context.Background(), results, &sync.Mutex{}, "one request")
	if got := requests.Load(); got != 1 {
		t.Fatalf("remote requests = %d, want 1", got)
	}
	if results.SignalErrors[config.SignalTypeDomain] != "category_classification_failed" {
		t.Fatalf("signal error = %q, want category_classification_failed", results.SignalErrors[config.SignalTypeDomain])
	}
}

func categoryDistributionHandler(t *testing.T, input string) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/classify" || r.Method != http.MethodPost {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		var request map[string]string
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if request["inputs"] != input {
			t.Errorf("inputs = %q, want %q", request["inputs"], input)
		}
		// Deliberately return labels out of mapping order: alignment by name is
		// what prevents a remote top-score ordering from changing routing.
		_, _ = w.Write([]byte(`[{"label":"physics","score":0.40},{"label":"other","score":0.10},{"label":"math","score":0.50}]`))
	})
}

func assertProbabilities(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("probability length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("probability[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestCategoryBackendConstructionUsesSharedModelValidator(t *testing.T) {
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{Classifier: config.Classifier{CategoryModel: config.CategoryModel{
			CategoryMappingPath: "mapping",
			Backend: &config.RemoteClassifierBackend{
				Protocol: config.RemoteClassifierProtocolHTTPClassify,
				Model:    "named-category",
			},
		}}},
		ExternalModels: []config.ExternalModelConfig{{
			Name:          "named-category",
			ModelRole:     config.ModelRoleGuardrail,
			ModelName:     "wrong-role-service",
			ModelEndpoint: config.ClassifierVLLMEndpoint{Address: "127.0.0.1", Port: 8080},
		}},
	}
	builder := newClassifierOptionBuilder(cfg, nil)
	categoryMapping := &CategoryMapping{
		CategoryToIdx: map[string]int{"math": 0, "other": 1},
		IdxToCategory: map[string]string{"0": "math", "1": "other"},
	}
	if err := builder.addCategoryClassifier(categoryMapping); err == nil {
		t.Fatal("expected direct category construction to reject an external model with the wrong role")
	}
}
