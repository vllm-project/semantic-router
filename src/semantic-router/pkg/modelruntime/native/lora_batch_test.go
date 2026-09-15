package native

import (
	"context"
	"errors"
	"io"
	"reflect"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type batchTestResource struct{ closes *atomic.Int32 }

func (r batchTestResource) Close() error { r.closes.Add(1); return nil }

func batchTestBinding[Output any](t *testing.T, pool *binding.Pool, name, contract string, closes *atomic.Int32, infer func(string) Output) *binding.Resolved[string, Output] {
	t.Helper()
	resource, err := pool.Acquire(context.Background(), binding.ResourceIdentity{Artifact: name, Revision: "fixture", Provider: "test", Device: "cpu", Precision: "fp32"}, "", nil, func(context.Context) (io.Closer, error) { return batchTestResource{closes}, nil })
	if err != nil {
		t.Fatal(err)
	}
	task, err := binding.Register(binding.NewRegistry(), contract, validateText, func(string, Output) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	handle, err := task.Resolve(binding.Identity{Recipe: "private", Name: name, Deployment: name, Adapter: "test", Contract: contract}, binding.Capability{Contract: contract, Provider: "test", Device: "cpu", Precision: "fp32"}, resource, func(_ context.Context, _ io.Closer, text string) (Output, error) { return infer(text), nil })
	if err != nil {
		t.Fatal(err)
	}
	return handle
}

func TestLoRABatchKeepsInputsAndIndependentResources(t *testing.T) {
	pool := binding.NewPool()
	var closes, calls atomic.Int32
	sequence := func(text string) tasks.LabelDistribution {
		calls.Add(1)
		if text == "one" {
			return tasks.LabelDistribution{Probabilities: []float32{.9, .1}}
		}
		return tasks.LabelDistribution{Probabilities: []float32{.2, .8}}
	}
	m := &LoRABatch{
		intent:   batchTestBinding(t, pool, "intent", "label_distribution.v1", &closes, sequence),
		security: batchTestBinding(t, pool, "security", "label_distribution.v1", &closes, sequence),
		pii: batchTestBinding(t, pool, "pii", "token_spans.v1", &closes, func(text string) tasks.TokenClassificationResult {
			calls.Add(1)
			available := true
			return tasks.TokenClassificationResult{ScoresAvailable: &available, Entities: []tasks.TokenEntity{{Text: text, Start: 0, End: len(text), EntityType: "O", Confidence: .8}}}
		}),
	}
	texts := []string{"one", "中文"}
	result, err := m.ClassifyBatch(context.Background(), "private", texts)
	if err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 6 || reflect.DeepEqual(result.Intent[0], result.Intent[1]) {
		t.Fatal("batch reused an aggregate result")
	}
	if result.PII[1].Entities[0].Text != "中文" || result.PII[1].Entities[0].End != len("中文") {
		t.Fatal("token bytes changed")
	}
	if _, err = m.ClassifyBatch(context.Background(), "foreign", texts); err == nil {
		t.Fatal("foreign recipe used prepared tasks")
	}
	if err = m.Close(); err != nil {
		t.Fatal(err)
	}
	if closes.Load() != 3 {
		t.Fatalf("closed %d resources, want three independent owners", closes.Load())
	}
	if _, err = m.ClassifyBatch(context.Background(), "private", texts); !errors.Is(err, binding.ErrClosed) {
		t.Fatalf("closed owner usable: %v", err)
	}
}
