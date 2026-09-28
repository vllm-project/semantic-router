package binding

import (
	"context"
	"errors"
	"fmt"
	"io"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func TestObservationPreservesActualInputAndImmutableCapabilities(t *testing.T) {
	var events []Event
	registry := NewRegistry(func(event Event) { events = append(events, event) })
	task, err := Register(registry, "test.distribution", func(string) error { return nil }, func(string, tasks.LabelDistribution) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	resource, err := NewPool().Acquire(context.Background(), testIdentity(), "", nil, func(context.Context) (io.Closer, error) { return &testModel{}, nil })
	if err != nil {
		t.Fatal(err)
	}
	id := Identity{Recipe: "r", Name: "domain", Deployment: "weights", Contract: "test.distribution", Adapter: "test"}
	capability := Capability{Contract: id.Contract, Provider: "test", Device: "cpu", Precision: "fp32", Limits: Limits{ModelTokens: 32768, TaskTokens: 512, DeploymentTokens: 128, Overflow: "truncate"}, Labels: []string{"safe", "unsafe"}, Embedding: &EmbeddingCapability{Dimension: 2, Modalities: []string{"text"}}}
	bound, err := task.Resolve(id, capability, resource, func(context.Context, io.Closer, string) (tasks.LabelDistribution, error) {
		return tasks.LabelDistribution{Probabilities: []float32{0.25, 0.75}, Input: &tasks.InputUsage{OriginalTokens: 200, ProcessedTokens: 128, Truncated: true}}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	capability.Labels[0] = "mutated"
	capability.Embedding.Modalities[0] = "mutated"
	published := bound.Capability()
	published.Labels[1] = "changed"
	published.Embedding.Modalities[0] = "changed"
	if got := bound.Capability(); got.Labels[0] != "safe" || got.Labels[1] != "unsafe" || got.Embedding.Modalities[0] != "text" {
		t.Fatalf("capability aliased caller memory: %+v", got)
	}
	bound.Ready()
	if _, err := bound.Call(context.Background(), "r", "text is not exposed by event"); err != nil {
		t.Fatal(err)
	}
	if err := bound.Close(); err != nil {
		t.Fatal(err)
	}
	if err := bound.Close(); err != nil {
		t.Fatal(err)
	}
	if len(events) != 4 || events[0].State != "resolved" || events[1].State != "ready" || events[2].State != "call" || events[3].State != "closed" {
		t.Fatalf("lifecycle events %+v", events)
	}
	event := events[2]
	if !event.Executed || event.Input == nil || event.Input.OriginalTokens != 200 || event.Input.ProcessedTokens != 128 || !event.Input.Truncated || event.Capability.Limits.EffectiveTokens() != 128 {
		t.Fatalf("lost actual input/limit facts %+v", event)
	}
}

func TestPartialSpansMustStillPassTaskValidation(t *testing.T) {
	for _, valid := range []bool{true, false} {
		t.Run(fmt.Sprint(valid), func(t *testing.T) {
			task, err := Register(NewRegistry(), "spans", func(string) error { return nil }, func(_ string, output tasks.TokenClassificationResult) error {
				if len(output.Entities) != 1 {
					return fmt.Errorf("test task requires one actual span")
				}
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			resource, err := NewPool().Acquire(context.Background(), testIdentity(), "", nil, func(context.Context) (io.Closer, error) { return &testModel{}, nil })
			if err != nil {
				t.Fatal(err)
			}
			id := Identity{Recipe: "r", Name: "spans", Deployment: "weights", Contract: "spans", Adapter: "test"}
			bound, err := task.Resolve(id, Capability{Contract: "spans", Provider: "test", Device: "cpu", Precision: "fp32"}, resource, func(context.Context, io.Closer, string) (tasks.TokenClassificationResult, error) {
				result := tasks.TokenClassificationResult{}
				if valid {
					result.Entities = []tasks.TokenEntity{{EntityType: "claim", Text: "text", End: 4}}
				}
				return result, tasks.ErrTokenSpansTruncated
			})
			if err != nil {
				t.Fatal(err)
			}
			defer bound.Close()
			_, err = bound.Call(context.Background(), "r", "text")
			if valid && !errors.Is(err, tasks.ErrTokenSpansTruncated) {
				t.Fatalf("valid partial span error lost: %v", err)
			}
			if !valid && (!errors.Is(err, ErrInvalidResult) || errors.Is(err, tasks.ErrTokenSpansTruncated)) {
				t.Fatalf("invalid result masqueraded as a valid partial scan: %v", err)
			}
		})
	}
}
