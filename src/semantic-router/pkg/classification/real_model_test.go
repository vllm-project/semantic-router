package classification

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving/servingtest"
)

// vela1SpecialistsConfig is the default configuration with every module the
// published-model runner provisions on its Vela 1.0 specialist, at the
// threshold calibrated for that model, as decision_model Vela-1.0 resolves it.
// The Router defaults name Vela 2.0 0.3B, which the runner does not provide.
func vela1SpecialistsConfig() config.RouterConfig {
	cfg := config.DefaultGlobalConfig()
	vela1 := config.Vela1SystemModels()
	cfg.PromptGuard.ModelID = vela1.PromptGuard
	cfg.PromptGuard.Threshold = config.ModuleThresholdsOf(vela1.PromptGuard).PromptGuard
	cfg.CategoryModel.ModelID = vela1.DomainClassifier
	cfg.CategoryModel.Threshold = config.ModuleThresholdsOf(vela1.DomainClassifier).Domain
	cfg.PIIModel.ModelID = vela1.PIIClassifier
	cfg.PIIModel.Threshold = config.ModuleThresholdsOf(vela1.PIIClassifier).PII
	factCheck := &cfg.HallucinationMitigation.FactCheckModel
	factCheck.ModelID = vela1.FactCheckClassifier
	factCheck.Threshold = config.ModuleThresholdsOf(vela1.FactCheckClassifier).FactCheck
	cfg.FeedbackDetector.ModelID = vela1.FeedbackDetector
	cfg.FeedbackDetector.Threshold = config.ModuleThresholdsOf(vela1.FeedbackDetector).Feedback
	return cfg
}

// Real-model execution requires an explicit path. A populated local cache must
// not change core-test behavior. The published-model runner supplies every path
// and sets VLLM_SR_REQUIRE_MODEL_TESTS=1, making missing opt-in a failure too.
func requireRealModel(t *testing.T, override, defaultPath string) string {
	t.Helper()
	model := config.GetModelByPath(defaultPath)
	if model == nil || len(model.Revision) != 40 {
		t.Fatalf("real-model default %q must have an immutable registry revision", defaultPath)
	}
	path := os.Getenv(override)
	if path == "" {
		if os.Getenv("VLLM_SR_REQUIRE_MODEL_TESTS") == "1" {
			t.Fatalf("required real model %s needs explicit %s", model.RepoID, override)
		}
		t.Skipf("optional real model %s requires explicit %s", model.RepoID, override)
	}
	path, err := filepath.Abs(path)
	if err != nil {
		t.Fatalf("resolve %s: %v", override, err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("required real model %s at %q: %v", model.RepoID, path, err)
	}
	if !info.IsDir() {
		t.Fatalf("real model %s must be a directory: %q", model.RepoID, path)
	}
	// A partial download is a failure even in optional local runs.
	for _, name := range []string{"config.json", "tokenizer.json"} {
		if _, err := os.Stat(filepath.Join(path, name)); err != nil {
			t.Fatalf("real model %s is incomplete: %v", model.RepoID, err)
		}
	}
	// The published-model runner stamps each package with the revision it
	// provisioned, so a test that resolves another model's registry entry
	// fails here instead of running that model's policy on these weights.
	if stamp, err := os.ReadFile(filepath.Join(path, ".complete")); err == nil {
		if provisioned := strings.TrimSpace(string(stamp)); provisioned != model.Revision {
			t.Fatalf("real model %s registers revision %s, but %s holds revision %s", model.RepoID, model.Revision, path, provisioned)
		}
	}
	t.Logf("real model=%s registered_revision=%s path=%s", model.RepoID, model.Revision, path)
	return path
}

// managedRuntimeOptions serves a test's consumers from managed model_runtime
// processes, as the router does.
func managedRuntimeOptions(t *testing.T) RecipeRuntimeOptions {
	t.Helper()
	return RecipeRuntimeOptions{Runtime: servingtest.Managed(t)}
}

// managedModelRuntime is managedRuntimeOptions for a standalone consumer.
func managedModelRuntime(t *testing.T) *classifierModelRuntime {
	t.Helper()
	models, err := newClassifierModelRuntime(&config.RouterConfig{}, managedRuntimeOptions(t))
	if err != nil {
		t.Fatal(err)
	}
	return models
}

func assertRealModelCPU(t *testing.T, capability binding.Capability) {
	t.Helper()
	if capability.Provider != config.ModelRuntimeProvider || capability.Device != "cpu" {
		t.Fatalf("expected %s CPU execution, got %+v", config.ModelRuntimeProvider, capability)
	}
	t.Logf("prepared provider=%s device=%s precision=%s labels=%v", capability.Provider, capability.Device, capability.Precision, capability.Labels)
}

func assertRealModelDistribution(t *testing.T, probabilities []float32, classes int) {
	t.Helper()
	if classes < 2 || len(probabilities) != classes {
		t.Fatalf("incomplete distribution: got %v, want %d classes", probabilities, classes)
	}
	var total float64
	for _, value := range probabilities {
		p := float64(value)
		if math.IsNaN(p) || math.IsInf(p, 0) || p < 0 || p > 1 {
			t.Fatalf("invalid probability distribution: %v", probabilities)
		}
		total += p
	}
	if math.Abs(total-1) > 1e-3 {
		t.Fatalf("probabilities sum to %g, want 1: %v", total, probabilities)
	}
}
