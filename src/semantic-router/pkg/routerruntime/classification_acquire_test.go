package routerruntime

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// generation mirrors the retirement contract the ext_proc router generation
// implements: a acquire registers a reference, and retirement waits for every
// outstanding reference before closing.
type generation struct {
	mu      sync.Mutex
	refs    sync.WaitGroup
	retired atomic.Bool
	closed  atomic.Bool
}

func (g *generation) acquire() (func(), bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.retired.Load() {
		return nil, false
	}
	g.refs.Add(1)
	return g.refs.Done, true
}

func (g *generation) retireAndClose() {
	g.mu.Lock()
	g.retired.Store(true)
	g.mu.Unlock()
	g.refs.Wait()
	g.closed.Store(true)
}

// TestRetirementWaitsForClassificationAcquire proves the classification API path
// keeps a retired generation alive for the duration of one call. Without the
// acquire the registry hands out a bare pointer and retirement closes the
// service while the caller still holds it.
func TestRetirementWaitsForClassificationAcquire(t *testing.T) {
	gen := &generation{}
	registry := &Registry{}
	registry.PublishRouterRuntimeSnapshot(RouterRuntimeSnapshot{
		ClassificationService: &services.ClassificationService{},
		AcquireClassification: gen.acquire,
	})

	service, release, ok := registry.AcquireClassificationService()
	if !ok || service == nil {
		t.Fatalf("expected a live classification service, got ok=%v service=%v", ok, service)
	}

	retired := make(chan struct{})
	go func() {
		gen.retireAndClose()
		close(retired)
	}()

	// Retirement must not complete while the reference is still held.
	select {
	case <-retired:
		t.Fatal("generation closed while a classification reference was still held")
	case <-time.After(100 * time.Millisecond):
	}
	if gen.closed.Load() {
		t.Fatal("classification service closed underneath an in-flight call")
	}

	release()

	select {
	case <-retired:
	case <-time.After(2 * time.Second):
		t.Fatal("retirement did not complete after the reference was released")
	}
}

// TestAcquireDeclinedAfterRetirement keeps a caller that loses the race from
// using a generation that is already closing.
func TestAcquireDeclinedAfterRetirement(t *testing.T) {
	gen := &generation{}
	registry := &Registry{}
	registry.PublishRouterRuntimeSnapshot(RouterRuntimeSnapshot{
		ClassificationService: &services.ClassificationService{},
		AcquireClassification: gen.acquire,
	})
	gen.retireAndClose()

	if _, _, ok := registry.AcquireClassificationService(); ok {
		t.Fatal("acquire succeeded against a retired generation")
	}
}
