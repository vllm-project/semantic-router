package binding

import (
	"context"
	"errors"
	"io"
	"testing"
)

func TestPreparedTaskLookupRequiresReadyTypedOwnedHandle(t *testing.T) {
	prepared := NewPreparedTasks()
	registry := NewRegistry(prepared.Observe)
	task, err := Register(registry, "labels.v1", func(string) error { return nil }, func(string, string) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	resource, err := NewPool().Acquire(context.Background(), ResourceIdentity{Artifact: "weights", Revision: "immutable", Provider: "test", Device: "cpu", Precision: "fp32"}, "", nil, func(context.Context) (io.Closer, error) { return &testModel{}, nil })
	if err != nil {
		t.Fatal(err)
	}
	handle, err := task.Resolve(Identity{Recipe: "one", Name: "domain", Deployment: "local", Contract: "labels.v1", Adapter: "test"}, Capability{Contract: "labels.v1", Provider: "test", Device: "cpu", Precision: "fp32", Labels: []string{"a", "b"}}, resource, func(_ context.Context, _ io.Closer, input string) (string, error) { return input, nil })
	if err != nil {
		t.Fatal(err)
	}
	defer handle.Close()
	if _, _, lookupErr := LookupPrepared[string, string](prepared, "one", "domain", "labels.v1"); !errors.Is(lookupErr, ErrNotPrepared) {
		t.Fatalf("unwarmed handle visible: %v", lookupErr)
	}
	handle.Ready()
	found, metadata, err := LookupPrepared[string, string](prepared, "one", "domain", "labels.v1")
	if err != nil || found != handle || metadata.Revision != "immutable" {
		t.Fatalf("lookup: %+v %v", metadata, err)
	}
	metadata.Capability.Labels[0] = "changed"
	_, again, _ := LookupPrepared[string, string](prepared, "one", "domain", "labels.v1")
	if again.Capability.Labels[0] != "a" {
		t.Fatal("lookup leaked mutable metadata")
	}
	if _, _, err := LookupPrepared[string, string](prepared, "two", "domain", "labels.v1"); !errors.Is(err, ErrNotPrepared) {
		t.Fatalf("foreign handle visible: %v", err)
	}
	if _, _, err := LookupPrepared[int, string](prepared, "one", "domain", "labels.v1"); !errors.Is(err, ErrNotPrepared) {
		t.Fatalf("wrong input type accepted: %v", err)
	}
	if _, err := found.Call(context.Background(), "two", "input"); !errors.Is(err, ErrCapability) {
		t.Fatalf("foreign inference accepted: %v", err)
	}
	if err := handle.Close(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := LookupPrepared[string, string](prepared, "one", "domain", "labels.v1"); !errors.Is(err, ErrNotPrepared) {
		t.Fatalf("retired handle visible: %v", err)
	}
}
