package binding

import (
	"context"
	"errors"
	"fmt"
	"io"
	"testing"
)

// This task is deliberately internal. It proves structured registration and
// binding without adding a routing signal, production selection API or model.
type selectionRequest struct {
	Messages   []struct{ Role, Content string }
	Candidates []string
}
type selectionResult struct {
	Selected []string
	Outcome  string
}

func TestStructuredTaskRegistrationBindingAndRecipeIsolation(t *testing.T) {
	registry := NewRegistry()
	task, err := Register(registry, "internal.selection.v1", func(input selectionRequest) error {
		if len(input.Messages) == 0 || len(input.Candidates) == 0 {
			return fmt.Errorf("messages and candidates required")
		}
		return nil
	}, func(input selectionRequest, result selectionResult) error {
		for _, selected := range result.Selected {
			found := false
			for _, candidate := range input.Candidates {
				if candidate == selected {
					found = true
				}
			}
			if !found {
				return fmt.Errorf("selected unknown candidate")
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	resolvedTask, err := Lookup[selectionRequest, selectionResult](registry, "internal.selection.v1")
	if err != nil || resolvedTask != task {
		t.Fatalf("lookup: %v", err)
	}
	if _, lookupErr := Lookup[string, string](registry, "internal.selection.v1"); !errors.Is(lookupErr, ErrCapability) {
		t.Fatalf("untyped lookup: %v", lookupErr)
	}
	resource, err := NewPool().Acquire(context.Background(), testIdentity(), "", nil, func(context.Context) (io.Closer, error) { return &testModel{}, nil })
	if err != nil {
		t.Fatal(err)
	}
	defer resource.Close()
	id := Identity{Recipe: "support", Name: "selection", Deployment: "internal", Contract: "internal.selection.v1", Adapter: "test-structured"}
	capability := Capability{Contract: id.Contract, Provider: "test", Device: "cpu", Precision: "fp32"}
	err = task.RegisterProvider("test", func(context.Context, Identity) (Prepared[selectionRequest, selectionResult], error) {
		return Prepared[selectionRequest, selectionResult]{Capability: capability, Resource: resource, Infer: func(_ context.Context, _ io.Closer, input selectionRequest) (selectionResult, error) {
			return selectionResult{Selected: []string{input.Candidates[0]}, Outcome: "selected"}, nil
		}}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	bound, err := task.Bind(context.Background(), "test", id)
	if err != nil {
		t.Fatal(err)
	}
	input := selectionRequest{Messages: []struct{ Role, Content string }{{"user", "help"}}, Candidates: []string{"answer"}}
	if _, callErr := bound.Call(context.Background(), "foreign", input); !errors.Is(callErr, ErrCapability) {
		t.Fatalf("foreign recipe got %v", callErr)
	}
	result, err := bound.Call(context.Background(), "support", input)
	if err != nil || len(result.Selected) != 1 || result.Selected[0] != "answer" {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	invalid, err := task.Resolve(id, capability, resource, func(context.Context, io.Closer, selectionRequest) (selectionResult, error) {
		return selectionResult{Selected: []string{"foreign-candidate"}}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := invalid.Call(context.Background(), "support", input); !errors.Is(err, ErrInvalidResult) {
		t.Fatalf("invalid result got %v", err)
	}
}

func TestTaskLimitsDoNotAdvertiseModelCapacityAsTaskSupport(t *testing.T) {
	limits := Limits{ModelTokens: 32768, TaskTokens: 512, DeploymentTokens: 256, Overflow: "reject"}
	if err := limits.Validate(); err != nil {
		t.Fatal(err)
	}
	if limits.EffectiveTokens() != 256 {
		t.Fatal("wrong effective budget")
	}
	if err := limits.CheckInput(257); !errors.Is(err, ErrInputLimit) {
		t.Fatalf("overflow=%v", err)
	}
	limits.DeploymentTokens = 1024
	if err := limits.Validate(); !errors.Is(err, ErrCapability) {
		t.Fatalf("unsupported task budget=%v", err)
	}
	limits.DeploymentTokens = 0
	if limits.EffectiveTokens() != 512 {
		t.Fatal("model capacity bypassed task restriction")
	}
}

func TestWindowLimitsKeepDocumentAndForwardCapacitySeparate(t *testing.T) {
	limits := Limits{ModelTokens: 32768, TaskTokens: 32768, DocumentTokens: 262144, DeploymentTokens: 65536, Overflow: "window"}
	if err := limits.Validate(); err != nil {
		t.Fatal(err)
	}
	if limits.ForwardTokens() != 32768 || limits.EffectiveTokens() != 65536 {
		t.Fatalf("mixed forward and document capacity: %+v", limits)
	}
	if err := limits.CheckInput(42000); err != nil {
		t.Fatal(err)
	}
	if err := limits.CheckInput(65537); !errors.Is(err, ErrInputLimit) {
		t.Fatalf("document overflow: %v", err)
	}
	for name, mutate := range map[string]func(*Limits){
		"single forward":         func(l *Limits) { l.Overflow = "reject" },
		"truncation":             func(l *Limits) { l.Overflow = "truncate" },
		"no document capability": func(l *Limits) { l.DocumentTokens = 0 },
		"insufficient scan":      func(l *Limits) { l.DocumentTokens = 32768 },
		"unknown forward":        func(l *Limits) { l.ModelTokens = 0; l.TaskTokens = 0 },
		"negative scan":          func(l *Limits) { l.DocumentTokens = -1 },
	} {
		t.Run(name, func(t *testing.T) {
			bad := limits
			mutate(&bad)
			if err := bad.Validate(); !errors.Is(err, ErrCapability) {
				t.Fatalf("unsupported capability accepted: %+v %v", bad, err)
			}
		})
	}
}
