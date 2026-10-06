package classification

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func TestSafetyBoundHTTPHeadRetainsIndependentScoresAndRecipeOwnership(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		_ = json.NewEncoder(w).Encode([]map[string]any{{"label": "b", "score": .9}, {"label": "a", "score": .8}})
	}))
	defer server.Close()
	cfg := &config.RouterConfig{}
	cfg.RoutingScope = "private"
	cfg.SafetyRules = []config.SafetyRule{{Name: "risk", Threshold: .5, Hazard: &config.SafetyHazardRule{Labels: []string{"a", "b"}, Categories: []string{"b"}, Threshold: .85}}}
	cfg.SafetyModels.Safety.ModelID = t.TempDir() // Native fallback must never be loaded.
	cfg.ExternalModels = []config.ExternalModelConfig{{Name: "bound", ModelRole: config.ModelRoleClassification, ModelName: "real-head", ModelEndpoint: endpointForTestServer(t, server)}}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"http": {Provider: "http", ExternalModel: "bound"}}
	cfg.ModelBindings = map[string]config.ModelBinding{
		"safety.risk":        {Deployment: "http", Adapter: config.RemoteClassifierProtocolHTTPClassify, Contract: config.RemoteClassifierContractLabelDistribution},
		"safety.risk.hazard": {Deployment: "http", Adapter: config.RemoteClassifierProtocolHTTPClassify, Contract: config.RemoteClassifierContractLabelScores},
	}
	cfg.SafetyRules[0].Labels = []string{"a", "b"}
	cfg.SafetyRules[0].UnsafeLabels = []string{"b"}
	builder := newClassifierOptionBuilder(cfg, nil)
	apply, buildErr := builder.buildSafetyClassifiersOption()
	if buildErr != nil {
		t.Fatal(buildErr)
	}
	classifier := &Classifier{Config: cfg}
	apply(classifier)
	defer classifier.Close()
	if err := classifier.initializeSafetyClassifiers(); err != nil {
		t.Fatal(err)
	}
	detector := classifier.safetyClassifiers["risk"]
	if _, err := detector.binary.Classify(context.Background(), "hello"); err == nil {
		t.Fatal("categorical head accepted independent scores")
	}
	got, err := detector.hazard.Classify(context.Background(), "hello")
	if err != nil {
		t.Fatal(err)
	}
	if got.Scores["a"] < .79 || got.Scores["b"] < .89 || selectedHazardScore(got.Scores, []string{"a", "b"}) > 1 {
		t.Fatalf("incorrect independent alignment/selection: %+v", got)
	}
	owned := detector.hazard.(*remoteSafetyScores)
	if _, err := owned.handle.Call(context.Background(), "other", "hello"); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("foreign recipe admitted: %v", err)
	}
	if calls.Load() != 2 {
		t.Fatalf("unexpected remote calls: %d", calls.Load())
	}
	if err := detector.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := detector.hazard.Classify(context.Background(), "hello"); !errors.Is(err, binding.ErrClosed) {
		t.Fatalf("closed head usable: %v", err)
	}
}
