package routerruntime

import (
	"sync/atomic"
	"testing"
)

type leasedReplayRuntime struct {
	releases atomic.Int32
}

func (*leasedReplayRuntime) HandleReplayRequest(string, string) (ReplayResponse, bool) {
	return ReplayResponse{StatusCode: 200, Body: []byte(`{"ok":true}`)}, true
}

func (runtime *leasedReplayRuntime) AcquireLease() (func(), bool) {
	return func() { runtime.releases.Add(1) }, true
}

func TestRegistryPublishesAndLeasesReplayRuntime(t *testing.T) {
	runtime := &leasedReplayRuntime{}
	registry := NewRegistry(nil)
	registry.PublishRouterRuntimeSnapshot(RouterRuntimeSnapshot{ReplayRuntime: runtime})

	current, release := registry.AcquireReplayRuntime()
	if current != runtime {
		t.Fatalf("replay runtime = %T, want published runtime", current)
	}
	if runtime.releases.Load() != 0 {
		t.Fatal("replay lease released before request completion")
	}
	release()
	if runtime.releases.Load() != 1 {
		t.Fatalf("replay lease releases = %d, want 1", runtime.releases.Load())
	}
}
