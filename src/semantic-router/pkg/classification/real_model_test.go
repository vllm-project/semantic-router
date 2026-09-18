package classification

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

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
	t.Logf("real model=%s registered_revision=%s path=%s", model.RepoID, model.Revision, path)
	return path
}

func assertRealModelCPU(t *testing.T, capability binding.Capability) {
	t.Helper()
	provider, _ := config.DefaultModelExecution(true)
	if capability.Provider != provider || capability.Device != "cpu" {
		t.Fatalf("expected %s CPU execution, got %+v", provider, capability)
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
