package routerruntime

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
)

type nativeGenerationRouter struct{ name string }

func (r *nativeGenerationRouter) RouteSystemOne(context.Context, string, json.RawMessage, systemone.Invoke) (int, []byte, error) {
	return 200, []byte(r.name), nil
}

func TestSystemOneAcquireKeepsSnapshotAndRouterTogether(t *testing.T) {
	oldSnapshot, nextSnapshot := &configsnapshot.Snapshot{}, &configsnapshot.Snapshot{}
	oldRouter, nextRouter := &nativeGenerationRouter{"old"}, &nativeGenerationRouter{"next"}
	started, allow := make(chan struct{}), make(chan struct{})
	gen := &generation{}
	registry := &Registry{}
	registry.PublishRouterRuntimeSnapshot(RouterRuntimeSnapshot{
		ConfigSnapshot: oldSnapshot, NativeRouter: oldRouter,
		AcquireClassification: func() (func(), bool) {
			close(started)
			<-allow
			return gen.acquire()
		},
	})
	type acquisition struct {
		snapshot *configsnapshot.Snapshot
		router   systemone.Router
		release  func()
		ok       bool
	}
	acquired := make(chan acquisition, 1)
	go func() {
		snapshot, router, release, ok := registry.AcquireSystemOne()
		acquired <- acquisition{snapshot, router, release, ok}
	}()
	<-started
	published := make(chan struct{})
	go func() {
		registry.PublishRouterRuntimeSnapshot(RouterRuntimeSnapshot{
			ConfigSnapshot: nextSnapshot, NativeRouter: nextRouter,
			AcquireClassification: (&generation{}).acquire,
		})
		close(published)
	}()
	select {
	case <-published:
		t.Fatal("generation changed before native request acquired its lease")
	case <-time.After(50 * time.Millisecond):
	}
	close(allow)
	got := <-acquired
	if !got.ok || got.snapshot != oldSnapshot || got.router != oldRouter {
		t.Fatal("native request mixed generations")
	}
	<-published
	retired := make(chan struct{})
	go func() { gen.retireAndClose(); close(retired) }()
	select {
	case <-retired:
		t.Fatal("old generation closed underneath its native request")
	case <-time.After(50 * time.Millisecond):
	}
	got.release()
	select {
	case <-retired:
	case <-time.After(time.Second):
		t.Fatal("native lease was not released")
	}
	snapshot, router, release, ok := registry.AcquireSystemOne()
	if !ok || snapshot != nextSnapshot || router != nextRouter {
		t.Fatal("new request did not acquire the current generation")
	}
	release()
}

func TestSystemOneAcquireRejectsRetiredGeneration(t *testing.T) {
	gen := &generation{}
	registry := &Registry{}
	registry.PublishRouterRuntimeSnapshot(RouterRuntimeSnapshot{
		ConfigSnapshot: &configsnapshot.Snapshot{}, NativeRouter: &nativeGenerationRouter{"retired"},
		AcquireClassification: gen.acquire,
	})
	gen.retireAndClose()
	if _, _, _, ok := registry.AcquireSystemOne(); ok {
		t.Fatal("acquired an already retired native runtime")
	}
}
