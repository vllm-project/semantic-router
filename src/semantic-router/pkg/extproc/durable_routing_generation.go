package extproc

import (
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// DurableRoutingLease keeps an immutable router generation alive until Release
// is called. Release is idempotent and safe to call from deferred cleanup.
type DurableRoutingLease struct {
	Router *OpenAIRouter
	Pin    DurableRoutingGenerationPin

	releaseOnce sync.Once
	release     func()
}

// Release relinquishes the generation reference held by the lease.
func (lease *DurableRoutingLease) Release() {
	if lease == nil {
		return
	}
	lease.releaseOnce.Do(func() {
		if lease.release != nil {
			lease.release()
		}
	})
}

type durableRoutingNamespace struct {
	reference accesspublisher.NamespacePublication

	mu          sync.Mutex
	removed     bool
	active      *durableRoutingGeneration
	generations map[string]*durableRoutingGeneration
}

type durableRoutingGeneration struct {
	identity accesspublisher.RuntimePublicationIdentity
	pin      DurableRoutingGenerationPin
	runtime  *durableRoutingRuntime
	snapshot *routingsnapshot.Snapshot

	mu      sync.Mutex
	refs    uint64
	retired bool
	closing bool
	onClose func(*durableRoutingGeneration, error)
}

// durableRoutingRuntime owns the expensive classifier and routing runtime that
// may be shared by exact publication generations with identical executable
// routing semantics. Publication pins and snapshots remain generation-local.
type durableRoutingRuntime struct {
	router         *OpenAIRouter
	semanticDigest string
	runtimeEpoch   uint64

	mu             sync.Mutex
	generationRefs uint64
	closed         bool
}

func (set *durableRoutingNamespace) remove() []*durableRoutingGeneration {
	set.mu.Lock()
	defer set.mu.Unlock()
	if set.removed {
		return nil
	}
	set.removed = true
	result := make([]*durableRoutingGeneration, 0, len(set.generations))
	for publicationID, generation := range set.generations {
		result = append(result, generation)
		delete(set.generations, publicationID)
	}
	set.active = nil
	return result
}

func newDurableRoutingGeneration(
	identity accesspublisher.RuntimePublicationIdentity,
	pin DurableRoutingGenerationPin,
	runtime *durableRoutingRuntime,
	snapshot *routingsnapshot.Snapshot,
	onClose func(*durableRoutingGeneration, error),
) (*durableRoutingGeneration, error) {
	if runtime == nil || runtime.router == nil || snapshot == nil ||
		runtime.semanticDigest != snapshot.SemanticDigest || runtime.runtimeEpoch != identity.RuntimeEpoch ||
		!runtime.retain() {
		return nil, ErrDurableRoutingPublicationCorrupt
	}
	return &durableRoutingGeneration{
		identity: identity,
		pin:      pin,
		runtime:  runtime,
		snapshot: snapshot,
		onClose:  onClose,
	}, nil
}

func newDurableRoutingRuntime(
	router *OpenAIRouter,
	semanticDigest string,
	runtimeEpoch uint64,
) *durableRoutingRuntime {
	return &durableRoutingRuntime{
		router: router, semanticDigest: semanticDigest, runtimeEpoch: runtimeEpoch,
	}
}

func (runtime *durableRoutingRuntime) retain() bool {
	if runtime == nil {
		return false
	}
	runtime.mu.Lock()
	defer runtime.mu.Unlock()
	if runtime.closed {
		return false
	}
	runtime.generationRefs++
	return true
}

func (runtime *durableRoutingRuntime) matches(semanticDigest string, runtimeEpoch uint64) bool {
	if runtime == nil || semanticDigest == "" || runtimeEpoch == 0 {
		return false
	}
	runtime.mu.Lock()
	defer runtime.mu.Unlock()
	return !runtime.closed && runtime.router != nil &&
		runtime.semanticDigest == semanticDigest && runtime.runtimeEpoch == runtimeEpoch
}

func (runtime *durableRoutingRuntime) release() error {
	if runtime == nil {
		return nil
	}
	runtime.mu.Lock()
	if runtime.generationRefs == 0 {
		runtime.mu.Unlock()
		return ErrDurableRoutingPublicationCorrupt
	}
	runtime.generationRefs--
	closeNow := runtime.generationRefs == 0
	if closeNow {
		runtime.closed = true
	}
	router := runtime.router
	runtime.mu.Unlock()
	if !closeNow || router == nil {
		return nil
	}
	return router.Close()
}

func (generation *durableRoutingGeneration) acquire() bool {
	generation.mu.Lock()
	defer generation.mu.Unlock()
	if generation.retired {
		return false
	}
	generation.refs++
	return true
}

func (generation *durableRoutingGeneration) release() {
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

func (generation *durableRoutingGeneration) retire() {
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
func (generation *durableRoutingGeneration) snapshotReadable() bool {
	if generation == nil {
		return false
	}
	generation.mu.Lock()
	defer generation.mu.Unlock()
	return !generation.retired || generation.refs > 0
}

func (generation *durableRoutingGeneration) close() {
	err := generation.runtime.release()
	if generation.onClose != nil {
		generation.onClose(generation, err)
	}
}

func (registry *DurableRoutingRegistry) generationClosed(generation *durableRoutingGeneration, err error) {
	registry.releaseRoutingSnapshot(generation)
	if err != nil {
		registry.closeMu.Lock()
		registry.closeErrors = append(registry.closeErrors, err)
		registry.closeMu.Unlock()
	}
	registry.generationWait.Done()
}
