package classification

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBuildAdmissionRegistryUsesConfiguredGates(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.ModelAdmission = map[string]config.AdmissionConfig{
		"prompt_guard": {MaxConcurrency: 1, OnOverflow: "shed"},
	}
	registry := buildAdmissionRegistry(cfg)

	if _, ok := registry.For("prompt_guard").(*admission.Semaphore); !ok {
		t.Fatal("configured deployment must get a semaphore gate")
	}
	if _, ok := registry.For("pii_classifier").(admission.Noop); !ok {
		t.Fatal("unconfigured deployment must get Noop")
	}
}

func TestAdmitModelInferenceShedsWhenGateIsFull(t *testing.T) {
	gate := admission.NewSemaphore(1, 0, 0, admission.OverflowShed)
	ticket, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer ticket()

	_, err = admitModelInference(context.Background(), gate, "prompt_guard", func() (string, error) {
		t.Fatal("inference must not run when the gate sheds")
		return "", nil
	})
	if !errors.Is(err, admission.ErrQueueFull) {
		t.Fatalf("error = %v, want ErrQueueFull", err)
	}
}

func TestAdmitModelInferenceRunsWithNilGate(t *testing.T) {
	result, err := admitModelInference[string](nil, nil, "prompt_guard", func() (string, error) {
		return "ok", nil
	})
	if err != nil || result != "ok" {
		t.Fatalf("result = %q, err = %v", result, err)
	}
}

type stubSequenceBackend struct{ calls int }

func (s *stubSequenceBackend) Classify(ctx context.Context, text string) (SequenceClassificationResult, error) {
	s.calls++
	return SequenceClassificationResult{}, nil
}

func TestApplyAdmissionGatesWrapsJailbreakBackend(t *testing.T) {
	stub := &stubSequenceBackend{}
	cfg := &config.RouterConfig{}
	cfg.ModelAdmission = map[string]config.AdmissionConfig{
		"prompt_guard": {MaxConcurrency: 1, OnOverflow: "shed"},
	}
	classifier := &Classifier{Config: cfg, jailbreakInference: stub}
	classifier.applyAdmissionGates()

	wrapped, ok := classifier.jailbreakInference.(admittedSequenceClassifier)
	if !ok {
		t.Fatalf("jailbreak backend = %T, want admittedSequenceClassifier", classifier.jailbreakInference)
	}
	if _, err := wrapped.Classify(context.Background(), "text"); err != nil {
		t.Fatal(err)
	}
	if stub.calls != 1 {
		t.Fatalf("backend calls = %d, want 1", stub.calls)
	}

	ticket, err := wrapped.gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer ticket()
	if _, err := wrapped.Classify(context.Background(), "text"); !errors.Is(err, admission.ErrQueueFull) {
		t.Fatalf("error = %v, want ErrQueueFull", err)
	}
}
