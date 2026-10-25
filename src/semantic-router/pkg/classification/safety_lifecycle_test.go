package classification

import (
	"context"
	"errors"
	"io"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type safetyClosingHead struct{ closed atomic.Int32 }

func (s *safetyClosingHead) Close() error { s.closed.Add(1); return nil }

func TestSafetyTypedBindingsSharePhysicalOwnershipAndAdmission(t *testing.T) {
	pool := binding.NewPool()
	physical := &safetyClosingHead{}
	gate := admission.NewSemaphore(1, 0, 0, admission.Overflow("shed"))
	calls := 0
	loads := 0
	task, registerErr := binding.Register(binding.NewRegistry(), config.RemoteClassifierContractLabelScores, func(string) error { return nil }, func(_ string, out tasks.LabelScores) error { return tasks.ValidateLabelScores(out.Scores) })
	if registerErr != nil {
		t.Fatal(registerErr)
	}
	prepare := func(name string) func(context.Context) (*binding.Resolved[string, tasks.LabelScores], error) {
		return func(ctx context.Context) (*binding.Resolved[string, tasks.LabelScores], error) {
			resource, err := pool.Acquire(ctx, binding.ResourceIdentity{Artifact: "same-artifact", Provider: "candle", Device: "cpu", Precision: "fp32"}, "one", gate, func(context.Context) (io.Closer, error) { loads++; return physical, nil })
			if err != nil {
				return nil, err
			}
			return task.Resolve(binding.Identity{Recipe: "default", Name: name, Deployment: "hazard", Contract: config.RemoteClassifierContractLabelScores, Adapter: "modernbert"}, binding.Capability{Contract: config.RemoteClassifierContractLabelScores, Provider: "candle", Device: "cpu", Precision: "fp32", Labels: []string{"a", "b"}}, resource, func(context.Context, io.Closer, string) (tasks.LabelScores, error) {
				calls++
				return tasks.LabelScores{Scores: []float32{.8, .9}}, nil
			})
		}
	}
	makeHead := func(name string) *ownedLabelTask[string, tasks.LabelScores] {
		return &ownedLabelTask[string, tasks.LabelScores]{labels: []string{"a", "b"}, recipe: "default", prepare: prepare(name), input: func(s string) string { return s }, convert: func(out tasks.LabelScores) (labelClassification, error) {
			scores, e := namedLabelScores([]string{"a", "b"}, out.Scores)
			return labelClassification{Scores: scores}, e
		}}
	}
	first, second := makeHead("first"), makeHead("second")
	for _, head := range []*ownedLabelTask[string, tasks.LabelScores]{first, second} {
		if err := head.Initialize(); err != nil {
			t.Fatal(err)
		}
	}
	if loads != 1 {
		t.Fatalf("loaded identical artifact %d times", loads)
	}
	release, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.Classify(context.Background(), "x"); !errors.Is(err, admission.ErrQueueFull) || calls != 0 {
		t.Fatalf("admission missed: %v calls=%d", err, calls)
	}
	release()
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	if physical.closed.Load() != 0 {
		t.Fatal("first task closed sibling physical resource")
	}
	if _, err := second.Classify(context.Background(), "x"); err != nil {
		t.Fatal(err)
	}
	if _, err := first.Classify(context.Background(), "x"); !errors.Is(err, binding.ErrClosed) {
		t.Fatalf("closed task remained usable: %v", err)
	}
	_ = second.Close()
	_ = second.Close()
	if physical.closed.Load() != 1 {
		t.Fatal("physical resource did not close once")
	}
}

func TestSafetyBuildDefersNativeInitializationWithDistinctConsumerHandles(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.SafetyModels = config.SafetyModelsConfig{Safety: config.SequenceHeadModelConfig{ModelID: t.TempDir(), UseCPU: true, MaxSequenceLength: 32768}}
	cfg.SafetyRules = []config.SafetyRule{{Name: "a", Threshold: .4}, {Name: "b", Threshold: .8}}
	builder := newClassifierOptionBuilder(cfg, nil)
	apply, err := builder.buildSafetyClassifiersOption()
	if err != nil {
		t.Fatalf("build loaded missing weights: %v", err)
	}
	classifier := &Classifier{Config: cfg}
	apply(classifier)
	a, b := classifier.safetyClassifiers["a"], classifier.safetyClassifiers["b"]
	if a.binary == b.binary {
		t.Fatal("consumers share a typed task identity")
	}
	if a.binaryKey != b.binaryKey {
		t.Fatal("same effective computation cannot share request results")
	}
	if err := classifier.initializeSafetyClassifiers(); err == nil {
		t.Fatal("initialization accepted missing weights")
	}
	if err := classifier.Close(); err != nil {
		t.Fatal(err)
	}
	if err := classifier.initializeSafetyClassifiers(); err == nil {
		t.Fatal("closed model reopened")
	}
}
