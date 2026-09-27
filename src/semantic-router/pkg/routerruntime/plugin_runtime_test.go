package routerruntime

import (
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestPluginRuntimeRetainsPublishedGeneration(t *testing.T) {
	var held atomic.Int32
	cfg := &config.RouterConfig{}
	registry := NewRegistry(nil)
	registry.PublishRouterRuntimeSnapshot(RouterRuntimeSnapshot{Config: cfg, AcquireClassification: func() (func(), bool) { held.Add(1); return func() { held.Add(-1) }, true }})
	actual, _, release, ok := registry.AcquirePluginRuntime()
	if !ok || actual != cfg || held.Load() != 1 {
		t.Fatalf("generation lease missing: acquired=%v held=%d", ok, held.Load())
	}
	registry.PublishRouterRuntimeSnapshot(RouterRuntimeSnapshot{Config: &config.RouterConfig{}})
	if held.Load() != 1 {
		t.Fatal("publication released an in-flight plugin operation")
	}
	release()
	if held.Load() != 0 {
		t.Fatal("plugin operation leaked its generation lease")
	}
	registry.PublishRouterRuntimeSnapshot(RouterRuntimeSnapshot{AcquireClassification: func() (func(), bool) { return nil, false }})
	if _, _, release, ok := registry.AcquirePluginRuntime(); ok {
		release()
		t.Fatal("retired generation was acquired")
	}
}
