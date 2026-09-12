package classification

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type safetyClosingHead struct{ closed atomic.Int32 }

func (s *safetyClosingHead) Classify(context.Context, string) (labelClassification, error) {
	return labelClassification{}, nil
}
func (s *safetyClosingHead) Close() error { s.closed.Add(1); return nil }

func TestSafetySharedModelOwnership(t *testing.T) {
	base := &safetyClosingHead{}
	shared := &sharedSafetyHead{labelClassifier: base}
	first := &safetyDetector{binary: shared}
	second := &safetyDetector{binary: shared}
	var workers sync.WaitGroup
	for range 10 {
		workers.Add(1)
		go func() { defer workers.Done(); _ = first.Close(); _ = second.Close() }()
	}
	workers.Wait()
	if base.closed.Load() != 1 {
		t.Fatalf("model closed %d times", base.closed.Load())
	}
}

func TestSafetyAdmissionRejectsBeforeNativeInference(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.SafetyModels.Safety.ModelID = t.TempDir()
	cfg.SafetyRules = []config.SafetyRule{{Name: "risk", Threshold: 0.5}}
	cfg.ModelAdmission = map[string]config.AdmissionConfig{
		"safety": {MaxConcurrency: 1, OnOverflow: "shed"},
	}
	option, err := newClassifierOptionBuilder(cfg, nil).buildSafetyClassifiersOption()
	if err != nil {
		t.Fatal(err)
	}
	c := &Classifier{Config: cfg}
	option(c)
	c.applyAdmissionGates()
	ticket, err := c.admissionRegistry.For("safety").Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer ticket()
	_, err = c.safetyClassifiers["risk"].binary.Classify(context.Background(), "unscored text")
	if !errors.Is(err, admission.ErrQueueFull) {
		t.Fatalf("safety ignored its full deployment gate: %v", err)
	}
}

func TestSafetyBuildDefersNativeInitializationAndSharesRules(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.SafetyModels = config.SafetyModelsConfig{Safety: config.SequenceHeadModelConfig{ModelID: t.TempDir(), UseCPU: true, MaxSequenceLength: 32768}}
	cfg.SafetyRules = []config.SafetyRule{{Name: "a", Threshold: 0.4}, {Name: "b", Threshold: 0.8}}
	builder := newClassifierOptionBuilder(cfg, nil)
	option, err := builder.buildSafetyClassifiersOption()
	if err != nil {
		t.Fatalf("build loaded missing weights: %v", err)
	}
	classifier := &Classifier{Config: cfg}
	option(classifier)
	a, b := classifier.safetyClassifiers["a"], classifier.safetyClassifiers["b"]
	if a.binary != b.binary {
		t.Fatal("identical local heads are loaded per rule")
	}
	if err := classifier.initializeSafetyClassifiers(); err == nil {
		t.Fatal("runtime initialization accepted missing weights")
	}
	if err := classifier.Close(); err != nil {
		t.Fatal(err)
	}
	if err := classifier.initializeSafetyClassifiers(); err == nil {
		t.Fatal("closed model reopened")
	}
}
