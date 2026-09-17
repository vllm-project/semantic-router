package binding

import (
	"context"
	"io"
	"sync"
	"testing"
)

func inventoryHandle(t *testing.T, pool *Pool, inventory *Inventory, recipe, device string) *Resolved[string, string] {
	t.Helper()
	task, err := Register(NewRegistry(inventory.Observe), "test.v1", func(string) error { return nil }, func(string, string) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	resource, err := pool.Acquire(context.Background(), ResourceIdentity{Artifact: "artifact", Provider: "test", Device: device, Precision: "fp32"}, "", nil, func(context.Context) (io.Closer, error) { return &testModel{}, nil })
	if err != nil {
		t.Fatal(err)
	}
	handle, err := task.Resolve(Identity{Recipe: recipe, Name: "domain", Deployment: "weights", Contract: "test.v1", Adapter: "test"}, Capability{Contract: "test.v1", Provider: "test", Device: device, Precision: "fp32", Labels: []string{"safe", "unsafe"}}, resource, func(_ context.Context, _ io.Closer, text string) (string, error) { return text, nil })
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handle.Close() })
	return handle
}

func TestInventoryTracksReadyHandlesAcrossSharedResourcesAndGenerations(t *testing.T) {
	pool := NewPool()
	current, candidate := NewInventory(), NewInventory()
	first := inventoryHandle(t, pool, current, "active", "cpu")
	second := inventoryHandle(t, pool, current, "active", "cpu")
	named := inventoryHandle(t, pool, current, "named", "cpu")
	failed := inventoryHandle(t, pool, candidate, "candidate", "cpu")
	if len(current.Snapshot()) != 0 {
		t.Fatal("resource load and task resolution must not report ready")
	}
	first.Ready()
	first.Ready()
	second.Ready()
	named.Ready()
	if got := current.Snapshot(); len(got) != 3 || got[0].Artifact != "artifact" || got[2].Identity.Recipe != "named" {
		t.Fatalf("lost separate handle/recipe inventory: %+v", got)
	}
	shared := current.Snapshot()
	if len(shared[0].ResourceID) != 64 || shared[0].ResourceID != shared[1].ResourceID || shared[1].ResourceID != shared[2].ResourceID {
		t.Fatalf("shared resource identity differs between consumer handles: %+v", shared)
	}
	other := inventoryHandle(t, pool, candidate, "other-device", "cuda:0")
	other.Ready()
	if candidate.Snapshot()[0].ResourceID == shared[0].ResourceID {
		t.Fatal("incompatible execution devices were merged")
	}
	_ = other.Close()
	_ = failed.Close() // A candidate that fails before warmup is never ready.
	if len(candidate.Snapshot()) != 0 || len(current.Snapshot()) != 3 {
		t.Fatal("candidate failure changed the active generation")
	}
	_ = first.Close()
	first.Ready() // A late readiness call cannot revive a retired handle.
	if len(current.Snapshot()) != 2 {
		t.Fatal("closing one shared handle removed a sibling or revived itself")
	}
	copy := current.Snapshot()
	copy[0].Capability.Labels[0] = "changed"
	if current.Snapshot()[0].Capability.Labels[0] != "safe" {
		t.Fatal("inventory snapshot aliases prepared metadata")
	}
	_ = second.Close()
	_ = named.Close()
	if len(current.Snapshot()) != 0 {
		t.Fatal("retired handles remain in inventory")
	}
}

func TestInventoryConcurrentReadyCloseAndSnapshot(t *testing.T) {
	inventory := NewInventory()
	handle := inventoryHandle(t, NewPool(), inventory, "active", "cpu")
	var workers sync.WaitGroup
	workers.Add(3)
	go func() { defer workers.Done(); handle.Ready() }()
	go func() { defer workers.Done(); _ = handle.Close() }()
	go func() { defer workers.Done(); _ = inventory.Snapshot() }()
	workers.Wait()
	if len(inventory.Snapshot()) != 0 {
		t.Fatal("closed handle remains ready after concurrent publication")
	}
}

func TestInventoryDoesNotExposeExternalConnectionIdentity(t *testing.T) {
	inventory := NewInventory()
	handle := inventoryHandle(t, NewPool(), inventory, "external", "external")
	handle.Ready()
	if got := inventory.Snapshot(); len(got) != 1 || got[0].Artifact != "" {
		t.Fatalf("external connector identity exposed: %+v", got)
	}
}
