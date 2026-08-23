package extproc

import (
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// ManagedRouterLease keeps an immutable router generation alive until Release
// is called. Release is idempotent and safe to call from deferred cleanup.
type ManagedRouterLease struct {
	Router *OpenAIRouter
	Pin    ManagedRouterGenerationPin

	releaseOnce sync.Once
	release     func()
}

// Release relinquishes the generation reference held by the lease.
func (lease *ManagedRouterLease) Release() {
	if lease == nil {
		return
	}
	lease.releaseOnce.Do(func() {
		if lease.release != nil {
			lease.release()
		}
	})
}

type managedRouterNamespace struct {
	reference accesspublisher.NamespacePublication

	mu          sync.Mutex
	removed     bool
	active      *managedRouterGeneration
	generations map[string]*managedRouterGeneration
}

type managedRouterGeneration struct {
	identity accesspublisher.RuntimePublicationIdentity
	pin      ManagedRouterGenerationPin
	router   *OpenAIRouter
	snapshot *routingsnapshot.Snapshot

	mu      sync.Mutex
	refs    uint64
	retired bool
	closing bool
	onClose func(*managedRouterGeneration, error)
}

func (set *managedRouterNamespace) remove() []*managedRouterGeneration {
	set.mu.Lock()
	defer set.mu.Unlock()
	if set.removed {
		return nil
	}
	set.removed = true
	result := make([]*managedRouterGeneration, 0, len(set.generations))
	for publicationID, generation := range set.generations {
		result = append(result, generation)
		delete(set.generations, publicationID)
	}
	set.active = nil
	return result
}

func newManagedRouterGeneration(
	identity accesspublisher.RuntimePublicationIdentity,
	pin ManagedRouterGenerationPin,
	router *OpenAIRouter,
	snapshot *routingsnapshot.Snapshot,
	onClose func(*managedRouterGeneration, error),
) *managedRouterGeneration {
	return &managedRouterGeneration{
		identity: identity,
		pin:      pin,
		router:   router,
		snapshot: snapshot,
		onClose:  onClose,
	}
}

func (generation *managedRouterGeneration) acquire() bool {
	generation.mu.Lock()
	defer generation.mu.Unlock()
	if generation.retired {
		return false
	}
	generation.refs++
	return true
}

func (generation *managedRouterGeneration) release() {
	generation.mu.Lock()
	if generation.refs == 0 {
		generation.mu.Unlock()
		return
	}
	generation.refs--
	closeNow := generation.retired && generation.refs == 0 && !generation.closing
	if closeNow {
		generation.closing = true
	}
	generation.mu.Unlock()
	if closeNow {
		go generation.close()
	}
}

func (generation *managedRouterGeneration) retire() {
	generation.mu.Lock()
	generation.retired = true
	closeNow := generation.refs == 0 && !generation.closing
	if closeNow {
		generation.closing = true
	}
	generation.mu.Unlock()
	if closeNow {
		go generation.close()
	}
}

// snapshotReadable reports whether this generation can still back a dispatch
// that was admitted against it. Active generations are readable without a
// request reference; retired generations remain readable only while an
// already-admitted router lease is draining.
func (generation *managedRouterGeneration) snapshotReadable() bool {
	if generation == nil {
		return false
	}
	generation.mu.Lock()
	defer generation.mu.Unlock()
	return !generation.retired || generation.refs > 0
}

func (generation *managedRouterGeneration) close() {
	err := generation.router.Close()
	if generation.onClose != nil {
		generation.onClose(generation, err)
	}
}

func (registry *ManagedRouterRegistry) generationClosed(generation *managedRouterGeneration, err error) {
	registry.releaseRoutingSnapshot(generation)
	if err != nil {
		registry.closeMu.Lock()
		registry.closeErrors = append(registry.closeErrors, err)
		registry.closeMu.Unlock()
	}
	registry.generationWait.Done()
}
