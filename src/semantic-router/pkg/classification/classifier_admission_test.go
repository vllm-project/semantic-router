package classification

import (
	"context"
	"errors"
	"sync"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
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

func TestSharedAdmissionRegistryAcrossClassifiers(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.ModelAdmission = map[string]config.AdmissionConfig{
		"prompt_guard": {MaxConcurrency: 1, OnOverflow: "shed"},
	}
	shared := buildAdmissionRegistry(cfg)
	first := &Classifier{Config: cfg, jailbreakInference: &stubSequenceBackend{}, admissionRegistry: shared}
	second := &Classifier{Config: cfg, jailbreakInference: &stubSequenceBackend{}, admissionRegistry: shared}
	first.applyAdmissionGates()
	second.applyAdmissionGates()

	ticket, err := first.jailbreakInference.(admittedSequenceClassifier).gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer ticket()
	if _, err := second.jailbreakInference.Classify(context.Background(), "text"); !errors.Is(err, admission.ErrQueueFull) {
		t.Fatalf("error = %v, want ErrQueueFull through the shared gate", err)
	}
}

type failingCategoryInference struct{}

func (failingCategoryInference) Classify(context.Context, string) (candle_binding.ClassResult, error) {
	return candle_binding.ClassResult{}, admission.ErrQueueFull
}

func (failingCategoryInference) ClassifyWithProbabilities(context.Context, string) (candle_binding.ClassResultWithProbs, error) {
	return candle_binding.ClassResultWithProbs{}, admission.ErrQueueFull
}

func TestDomainInferenceErrorPopulatesSignalErrors(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.Categories = []config.Category{{CategoryMetadata: config.CategoryMetadata{Name: "business"}}, {CategoryMetadata: config.CategoryMetadata{Name: "law"}}}
	classifier := &Classifier{Config: cfg, categoryInference: failingCategoryInference{}}
	results := &SignalResults{Metrics: &SignalMetricsCollection{}, SignalErrors: make(map[string]string), SignalConfidences: make(map[string]float64)}
	var mu sync.Mutex

	classifier.evaluateDomainSignal(context.Background(), results, &mu, "text")

	for _, name := range []string{"business", "law"} {
		if results.SignalErrors["domain:"+name] != domainEvaluationFailedCode {
			t.Fatalf("SignalErrors = %#v, want %q for %q", results.SignalErrors, domainEvaluationFailedCode, name)
		}
	}
}

type countingCategoryInference struct{ classify, classifyWithProbs int }

func (c *countingCategoryInference) Classify(context.Context, string) (candle_binding.ClassResult, error) {
	c.classify++
	return candle_binding.ClassResult{}, nil
}

func (c *countingCategoryInference) ClassifyWithProbabilities(context.Context, string) (candle_binding.ClassResultWithProbs, error) {
	c.classifyWithProbs++
	return candle_binding.ClassResultWithProbs{}, nil
}

type countingGate struct {
	calls int
	err   error
}

func (g *countingGate) Acquire(context.Context) (admission.Ticket, error) {
	g.calls++
	return nil, g.err
}

func TestDomainSignalDoesNotRetryAdmissionErrors(t *testing.T) {
	for name, gateErr := range map[string]error{"shed": admission.ErrQueueFull, "canceled": context.Canceled} {
		t.Run(name, func(t *testing.T) {
			cfg := &config.RouterConfig{}
			cfg.Categories = []config.Category{{CategoryMetadata: config.CategoryMetadata{Name: "business"}}}
			backend := &countingCategoryInference{}
			gate := &countingGate{err: gateErr}
			classifier := &Classifier{Config: cfg, categoryInference: admittedCategoryInference{backend: backend, gate: gate, deployment: admissionDeploymentDomainClassifier}}
			results := &SignalResults{Metrics: &SignalMetricsCollection{}, SignalErrors: make(map[string]string), SignalConfidences: make(map[string]float64)}
			var mu sync.Mutex

			classifier.evaluateDomainSignal(context.Background(), results, &mu, "text")

			if gate.calls != 1 {
				t.Fatalf("gate acquisitions = %d, want 1", gate.calls)
			}
			if backend.classify != 0 || backend.classifyWithProbs != 0 {
				t.Fatalf("backend calls = %d/%d, want none", backend.classifyWithProbs, backend.classify)
			}
			if results.SignalErrors["domain:business"] != domainEvaluationFailedCode {
				t.Fatalf("SignalErrors = %#v, want %q", results.SignalErrors, domainEvaluationFailedCode)
			}
		})
	}
}

type noFallbackCategoryInference struct{ countingCategoryInference }

func (*noFallbackCategoryInference) fallbackToTop1OnProbabilityError() bool { return false }

func TestAdmittedCategoryInferenceForwardsFallbackPolicy(t *testing.T) {
	if categoryProbabilityFallbackAllowed(admittedCategoryInference{backend: &countingCategoryInference{}}) != true {
		t.Fatal("backend without a policy must keep the fallback")
	}
	if categoryProbabilityFallbackAllowed(admittedCategoryInference{backend: &noFallbackCategoryInference{}}) {
		t.Fatal("wrapper must forward the backend's no-fallback policy")
	}
}

type failingPIIInference struct{}

func (failingPIIInference) ClassifyTokens(context.Context, string) (candle_binding.TokenClassificationResult, error) {
	return candle_binding.TokenClassificationResult{}, admission.ErrQueueFull
}

func TestPIIInferenceErrorPopulatesSignalErrors(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.PIIRules = []config.PIIRule{{Name: "no_pii"}}
	classifier := &Classifier{Config: cfg, piiInference: failingPIIInference{}, PIIMapping: &PIIMapping{}}
	results := &SignalResults{Metrics: &SignalMetricsCollection{}, SignalErrors: make(map[string]string)}
	var mu sync.Mutex

	classifier.evaluatePIISignal(context.Background(), results, &mu, "text", nil)

	if results.SignalErrors["pii:no_pii"] != piiEvaluationFailedCode {
		t.Fatalf("SignalErrors = %#v, want %q", results.SignalErrors, piiEvaluationFailedCode)
	}
	if len(results.MatchedPIIRules) != 0 {
		t.Fatalf("MatchedPIIRules = %v, want none", results.MatchedPIIRules)
	}
}
