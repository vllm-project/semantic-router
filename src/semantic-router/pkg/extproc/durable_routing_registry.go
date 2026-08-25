package extproc

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/publicationreplica"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/runtimecapabilities"
)

var (
	ErrDurableRoutingRegistryClosed     = errors.New("durable routing registry is closed")
	ErrDurableRoutingUnavailable        = errors.New("durable routing generation is unavailable")
	ErrDurableRoutingPinMismatch        = errors.New("durable routing generation pin does not match")
	ErrDurableRoutingStaleGeneration    = errors.New("durable routing generation is stale")
	ErrDurableRoutingPublicationCorrupt = errors.New("durable routing publication is corrupt")
)

// DurableRoutingGenerationPin is the complete request-time identity of one
// immutable namespace router generation. Callers must supply every field; the
// registry never falls back to a default namespace or a newer generation.
type DurableRoutingGenerationPin struct {
	NamespaceID      string
	QuotaPartition   string
	PublicationID    string
	RuntimeEpoch     uint64
	SnapshotRevision int64
	RoutingDigest    string
}

func (pin DurableRoutingGenerationPin) validate() error {
	reference := accesspublisher.NamespacePublication{
		NamespaceID: pin.NamespaceID, QuotaPartition: pin.QuotaPartition,
	}
	if err := reference.Validate(); err != nil {
		return fmt.Errorf("%w: %w", ErrDurableRoutingPinMismatch, err)
	}
	if strings.TrimSpace(pin.PublicationID) == "" || pin.RuntimeEpoch == 0 || pin.SnapshotRevision <= 0 ||
		!durableRoutingDigestValid(pin.RoutingDigest) {
		return fmt.Errorf("%w: publication, epoch, revision, and digest are required", ErrDurableRoutingPinMismatch)
	}
	return nil
}

// DurableRoutingRegistryOptions contains process-owned bootstrap state. The
// bootstrap configuration and dependencies are borrowed as immutable values;
// every warmed generation receives a newly compiled RouterConfig.
type DurableRoutingRegistryOptions struct {
	BootstrapConfig *config.RouterConfig
	Dependencies    RuntimeDependencies
}

type durableRoutingBuilder func(*config.RouterConfig, RuntimeDependencies) (*OpenAIRouter, error)

type routingSnapshotKey struct {
	namespaceID string
	revision    int64
}

type retainedRoutingSnapshot struct {
	digest      string
	snapshot    *routingsnapshot.Snapshot
	generations map[*durableRoutingGeneration]struct{}
}

// DurableRoutingRegistry owns the immutable OpenAIRouter generations loaded by
// publicationreplica.Manager. Namespace state is isolated so warming one
// namespace never serializes generation work for another namespace.
type DurableRoutingRegistry struct {
	bootstrap    *config.RouterConfig
	dependencies RuntimeDependencies
	build        durableRoutingBuilder

	closed            atomic.Bool
	mu                sync.Mutex
	sets              map[string]*durableRoutingNamespace
	retainedSnapshots map[routingSnapshotKey]*retainedRoutingSnapshot

	generationWait sync.WaitGroup
	closeMu        sync.Mutex
	closeErrors    []error
	closeDone      chan struct{}
	finalCloseErr  error
}

var (
	_ publicationreplica.SnapshotLifecycle = (*DurableRoutingRegistry)(nil)
	_ backendinvoker.RoutingSnapshotSource = (*DurableRoutingRegistry)(nil)
)

// NewDurableRoutingRegistry creates an empty namespace generation registry.
func NewDurableRoutingRegistry(options DurableRoutingRegistryOptions) (*DurableRoutingRegistry, error) {
	return newDurableRoutingRegistry(options, buildOpenAIRouterFromConfigWithDependencies)
}

func newDurableRoutingRegistry(
	options DurableRoutingRegistryOptions,
	build durableRoutingBuilder,
) (*DurableRoutingRegistry, error) {
	if options.BootstrapConfig == nil {
		return nil, fmt.Errorf("durable routing bootstrap configuration is required")
	}
	capabilities, err := runtimecapabilities.Derive(options.BootstrapConfig)
	if err != nil {
		return nil, err
	}
	if !capabilities.DurableRouting {
		return nil, fmt.Errorf("durable routing registry requires durable Management")
	}
	if build == nil {
		return nil, fmt.Errorf("durable routing builder is required")
	}
	if err := options.Dependencies.validate(options.BootstrapConfig); err != nil {
		return nil, err
	}
	// Isolate top-level mutations while retaining the bootstrap's explicitly
	// immutable nested service state. CompileDurableRoutingSnapshot exports and
	// rebuilds that state for every generation.
	bootstrap := *options.BootstrapConfig
	return &DurableRoutingRegistry{
		bootstrap: &bootstrap, dependencies: options.Dependencies, build: build,
		sets:              make(map[string]*durableRoutingNamespace),
		retainedSnapshots: make(map[routingSnapshotKey]*retainedRoutingSnapshot),
		closeDone:         make(chan struct{}),
	}, nil
}

// Warm strictly verifies and builds one publication candidate without making
// it visible to request admission.
func (registry *DurableRoutingRegistry) Warm(
	ctx context.Context,
	publication accesspublisher.LoadedRoutingPublication,
) error {
	if registry == nil || registry.closed.Load() {
		return ErrDurableRoutingRegistryClosed
	}
	if err := contextError(ctx); err != nil {
		return err
	}
	pin, compiled, snapshot, err := registry.compilePublication(publication, false)
	if err != nil {
		return err
	}
	reference := accesspublisher.NamespacePublication{
		NamespaceID: pin.NamespaceID, QuotaPartition: pin.QuotaPartition,
	}
	set, err := registry.namespace(reference, true)
	if err != nil {
		return err
	}

	set.mu.Lock()
	if set.removed {
		set.mu.Unlock()
		return ErrDurableRoutingUnavailable
	}
	if registry.closed.Load() {
		set.mu.Unlock()
		return ErrDurableRoutingRegistryClosed
	}
	if existing := set.generations[pin.PublicationID]; existing != nil {
		matchErr := matchingDurableRoutingGeneration(existing, publication.Identity, pin)
		set.mu.Unlock()
		return matchErr
	}
	if newest := newestDurableRoutingGeneration(set); newest != nil {
		switch compareDurableRoutingGenerationOrder(publication.Identity, newest.identity) {
		case -1:
			set.mu.Unlock()
			return fmt.Errorf("%w: %s precedes %s", ErrDurableRoutingStaleGeneration,
				publication.Identity.PublicationID, newest.identity.PublicationID)
		case 0:
			set.mu.Unlock()
			return fmt.Errorf("%w: revision already names another publication", ErrDurableRoutingPublicationCorrupt)
		}
	}

	runtime, builtRouter, err := registry.buildWarmRuntime(set, compiled, snapshot, publication.Identity.RuntimeEpoch)
	if err != nil {
		set.mu.Unlock()
		return err
	}
	if err := contextError(ctx); err != nil || registry.closed.Load() {
		set.mu.Unlock()
		if builtRouter != nil {
			_ = builtRouter.Close()
		}
		if err != nil {
			return err
		}
		return ErrDurableRoutingRegistryClosed
	}
	registry.mu.Lock()
	if registry.closed.Load() {
		registry.mu.Unlock()
		set.mu.Unlock()
		if builtRouter != nil {
			_ = builtRouter.Close()
		}
		return ErrDurableRoutingRegistryClosed
	}
	generation, generationErr := newDurableRoutingGeneration(
		publication.Identity, pin, runtime, snapshot, registry.generationClosed,
	)
	if generationErr != nil {
		registry.mu.Unlock()
		set.mu.Unlock()
		if builtRouter != nil {
			_ = builtRouter.Close()
		}
		return fmt.Errorf("%w: retain durable routing runtime", generationErr)
	}
	registry.generationWait.Add(1)
	set.generations[pin.PublicationID] = generation
	registry.mu.Unlock()
	set.mu.Unlock()
	return nil
}

func (registry *DurableRoutingRegistry) buildWarmRuntime(
	set *durableRoutingNamespace,
	compiled *config.RouterConfig,
	snapshot *routingsnapshot.Snapshot,
	runtimeEpoch uint64,
) (*durableRoutingRuntime, *OpenAIRouter, error) {
	runtime := reusableDurableRoutingRuntime(set, snapshot.SemanticDigest, runtimeEpoch)
	if runtime != nil {
		return runtime, nil, nil
	}
	router, err := registry.build(compiled, registry.dependencies)
	if err != nil {
		if router != nil {
			_ = router.Close()
		}
		return nil, nil, fmt.Errorf("build durable routing generation: %w", err)
	}
	if router == nil || router.Config != compiled || router.Config.DocumentHash != snapshot.SemanticDigest {
		if router != nil {
			_ = router.Close()
		}
		return nil, nil, fmt.Errorf("%w: router builder returned a different configuration", ErrDurableRoutingPublicationCorrupt)
	}
	return newDurableRoutingRuntime(router, snapshot.SemanticDigest, runtimeEpoch), router, nil
}

// Activate atomically selects an already verified generation for one
// namespace. The method is independently idempotent and therefore first warms
// the supplied value if a replica is recovering without an in-process cache.
func (registry *DurableRoutingRegistry) Activate(
	ctx context.Context,
	publication accesspublisher.LoadedRoutingPublication,
) error {
	if !publication.Identity.Activated() {
		return fmt.Errorf("%w: publication is not active", ErrDurableRoutingPinMismatch)
	}
	if err := registry.Warm(ctx, publication); err != nil {
		return err
	}
	pin, _, _, activateErr := registry.compilePublication(publication, true)
	if activateErr != nil {
		return activateErr
	}
	set, activateErr := registry.namespace(accesspublisher.NamespacePublication{
		NamespaceID: pin.NamespaceID, QuotaPartition: pin.QuotaPartition,
	}, false)
	if activateErr != nil {
		return activateErr
	}

	var retired []*durableRoutingGeneration
	set.mu.Lock()
	if set.removed {
		set.mu.Unlock()
		return ErrDurableRoutingUnavailable
	}
	if registry.closed.Load() {
		set.mu.Unlock()
		return ErrDurableRoutingRegistryClosed
	}
	generation := set.generations[pin.PublicationID]
	if generation == nil {
		set.mu.Unlock()
		return ErrDurableRoutingUnavailable
	}
	if err := matchingDurableRoutingGeneration(generation, publication.Identity, pin); err != nil {
		set.mu.Unlock()
		return err
	}
	if set.active == generation {
		registry.mu.Lock()
		err := registry.retainRoutingSnapshotLocked(generation)
		registry.mu.Unlock()
		set.mu.Unlock()
		return err
	}
	if set.active != nil {
		switch compareDurableRoutingGenerationOrder(publication.Identity, set.active.identity) {
		case -1:
			set.mu.Unlock()
			return fmt.Errorf("%w: active publication %s is newer", ErrDurableRoutingStaleGeneration,
				set.active.identity.PublicationID)
		case 0:
			set.mu.Unlock()
			return fmt.Errorf("%w: active revision names another publication", ErrDurableRoutingPublicationCorrupt)
		}
	}
	registry.mu.Lock()
	activateErr = registry.retainRoutingSnapshotLocked(generation)
	registry.mu.Unlock()
	if activateErr != nil {
		set.mu.Unlock()
		return activateErr
	}
	set.active = generation
	for publicationID, candidate := range set.generations {
		if candidate == generation {
			continue
		}
		delete(set.generations, publicationID)
		retired = append(retired, candidate)
	}
	set.mu.Unlock()
	for _, generation := range retired {
		generation.retire()
	}
	return nil
}

// Snapshot returns the exact immutable routing value named by namespace and
// revision. It never substitutes the active revision. A retired generation
// remains available only while an already-admitted router lease is draining.
// The returned value is a process-owned immutable Go object, so its lifetime
// is independent of the native resources released when the generation closes.
func (registry *DurableRoutingRegistry) Snapshot(
	ctx context.Context,
	pin routingcontext.Generation,
) (*routingsnapshot.Snapshot, error) {
	if err := contextError(ctx); err != nil {
		return nil, err
	}
	if registry == nil {
		return nil, ErrDurableRoutingRegistryClosed
	}
	if pin.Validate() != nil || strings.ContainsRune(pin.NamespaceID, 0) {
		return nil, fmt.Errorf("%w: complete routing generation is required", ErrDurableRoutingPinMismatch)
	}
	key := routingSnapshotKey{namespaceID: pin.NamespaceID, revision: pin.SnapshotRevision}
	registry.mu.Lock()
	retained := registry.retainedSnapshots[key]
	if retained == nil {
		closed := registry.closed.Load()
		registry.mu.Unlock()
		if closed {
			return nil, ErrDurableRoutingRegistryClosed
		}
		return nil, ErrDurableRoutingUnavailable
	}
	if retained.snapshot == nil || retained.snapshot.NamespaceID != pin.NamespaceID ||
		retained.snapshot.Revision != pin.SnapshotRevision || retained.snapshot.Digest != retained.digest {
		registry.mu.Unlock()
		return nil, ErrDurableRoutingPublicationCorrupt
	}
	matchedGeneration := false
	readable := false
	for generation := range retained.generations {
		if generation.pin.NamespaceID != pin.NamespaceID || generation.pin.QuotaPartition != pin.QuotaPartition ||
			generation.pin.PublicationID != pin.PublicationID || generation.pin.RuntimeEpoch != pin.RuntimeEpoch ||
			generation.pin.SnapshotRevision != pin.SnapshotRevision || generation.pin.RoutingDigest != pin.RoutingDigest {
			continue
		}
		matchedGeneration = true
		if generation.snapshotReadable() {
			readable = true
			break
		}
	}
	snapshot := retained.snapshot
	closed := registry.closed.Load()
	registry.mu.Unlock()
	if !matchedGeneration {
		return nil, ErrDurableRoutingPinMismatch
	}
	if !readable {
		if closed {
			return nil, ErrDurableRoutingRegistryClosed
		}
		return nil, ErrDurableRoutingUnavailable
	}
	return snapshot, nil
}

// retainRoutingSnapshotLocked makes one activated generation discoverable by
// backend dispatch. registry.mu and the namespace lock must both be held so
// activation and exact snapshot visibility share one linearization point.
func (registry *DurableRoutingRegistry) retainRoutingSnapshotLocked(
	generation *durableRoutingGeneration,
) error {
	if registry.closed.Load() {
		return ErrDurableRoutingRegistryClosed
	}
	if generation == nil || generation.snapshot == nil ||
		generation.snapshot.NamespaceID != generation.pin.NamespaceID ||
		generation.snapshot.Revision != generation.pin.SnapshotRevision ||
		!durableRoutingDigestValid(generation.snapshot.Digest) {
		return fmt.Errorf("%w: generation snapshot identity is invalid", ErrDurableRoutingPublicationCorrupt)
	}
	key := routingSnapshotKey{
		namespaceID: generation.pin.NamespaceID,
		revision:    generation.pin.SnapshotRevision,
	}
	retained := registry.retainedSnapshots[key]
	if retained == nil {
		registry.retainedSnapshots[key] = &retainedRoutingSnapshot{
			digest: generation.snapshot.Digest, snapshot: generation.snapshot,
			generations: map[*durableRoutingGeneration]struct{}{generation: {}},
		}
		return nil
	}
	if retained.snapshot == nil || retained.digest != generation.snapshot.Digest ||
		retained.snapshot.NamespaceID != key.namespaceID || retained.snapshot.Revision != key.revision ||
		retained.snapshot.Digest != retained.digest {
		return fmt.Errorf(
			"%w: namespace revision names different routing snapshots",
			ErrDurableRoutingPublicationCorrupt,
		)
	}
	retained.generations[generation] = struct{}{}
	return nil
}

func (registry *DurableRoutingRegistry) releaseRoutingSnapshot(generation *durableRoutingGeneration) {
	if registry == nil || generation == nil {
		return
	}
	key := routingSnapshotKey{
		namespaceID: generation.pin.NamespaceID,
		revision:    generation.pin.SnapshotRevision,
	}
	registry.mu.Lock()
	retained := registry.retainedSnapshots[key]
	if retained != nil {
		delete(retained.generations, generation)
		if len(retained.generations) == 0 {
			delete(registry.retainedSnapshots, key)
		}
	}
	registry.mu.Unlock()
}

// Acquire returns the exactly pinned active generation. Any missing or stale
// field fails closed instead of selecting the namespace's current generation.
func (registry *DurableRoutingRegistry) Acquire(pin DurableRoutingGenerationPin) (*DurableRoutingLease, error) {
	if registry == nil || registry.closed.Load() {
		return nil, ErrDurableRoutingRegistryClosed
	}
	if err := pin.validate(); err != nil {
		return nil, err
	}
	set, err := registry.namespace(accesspublisher.NamespacePublication{
		NamespaceID: pin.NamespaceID, QuotaPartition: pin.QuotaPartition,
	}, false)
	if err != nil {
		return nil, err
	}
	set.mu.Lock()
	generation := set.active
	removed := set.removed
	set.mu.Unlock()
	if removed || generation == nil {
		return nil, ErrDurableRoutingUnavailable
	}
	if generation.pin != pin {
		return nil, ErrDurableRoutingPinMismatch
	}
	if !generation.acquire() {
		return nil, ErrDurableRoutingUnavailable
	}
	return &DurableRoutingLease{
		Router: generation.runtime.router, Pin: generation.pin, release: generation.release,
	}, nil
}

// Remove retires every generation in exactly one namespace partition. It is
// idempotent after successful removal and never removes a namespace through a
// mismatched quota partition.
func (registry *DurableRoutingRegistry) Remove(
	ctx context.Context,
	reference accesspublisher.NamespacePublication,
) error {
	if err := contextError(ctx); err != nil {
		return err
	}
	if err := reference.Validate(); err != nil {
		return err
	}
	if registry == nil || registry.closed.Load() {
		return ErrDurableRoutingRegistryClosed
	}
	registry.mu.Lock()
	set := registry.sets[reference.NamespaceID]
	if set != nil && set.reference.QuotaPartition != reference.QuotaPartition {
		registry.mu.Unlock()
		return ErrDurableRoutingPinMismatch
	}
	delete(registry.sets, reference.NamespaceID)
	registry.mu.Unlock()
	if set == nil {
		return nil
	}
	retired := set.remove()
	for _, generation := range retired {
		generation.retire()
	}
	return nil
}

// Close stops admission, retires every namespace, waits for all leases to
// drain, and only then closes each OpenAIRouter generation.
func (registry *DurableRoutingRegistry) Close() error {
	if registry == nil {
		return nil
	}
	if !registry.closed.CompareAndSwap(false, true) {
		<-registry.closeDone
		registry.closeMu.Lock()
		defer registry.closeMu.Unlock()
		return registry.finalCloseErr
	}
	registry.mu.Lock()
	sets := make([]*durableRoutingNamespace, 0, len(registry.sets))
	for _, set := range registry.sets {
		sets = append(sets, set)
	}
	registry.sets = make(map[string]*durableRoutingNamespace)
	registry.mu.Unlock()

	var generations []*durableRoutingGeneration
	for _, set := range sets {
		generations = append(generations, set.remove()...)
	}
	for _, generation := range generations {
		generation.retire()
	}
	registry.generationWait.Wait()
	registry.closeMu.Lock()
	registry.finalCloseErr = errors.Join(registry.closeErrors...)
	result := registry.finalCloseErr
	close(registry.closeDone)
	registry.closeMu.Unlock()
	return result
}

func (registry *DurableRoutingRegistry) namespace(
	reference accesspublisher.NamespacePublication,
	create bool,
) (*durableRoutingNamespace, error) {
	if err := reference.Validate(); err != nil {
		return nil, err
	}
	if registry.closed.Load() {
		return nil, ErrDurableRoutingRegistryClosed
	}
	registry.mu.Lock()
	defer registry.mu.Unlock()
	if registry.closed.Load() {
		return nil, ErrDurableRoutingRegistryClosed
	}
	set := registry.sets[reference.NamespaceID]
	if set == nil {
		if !create {
			return nil, ErrDurableRoutingUnavailable
		}
		set = &durableRoutingNamespace{
			reference: reference, generations: make(map[string]*durableRoutingGeneration),
		}
		registry.sets[reference.NamespaceID] = set
		return set, nil
	}
	if set.reference.QuotaPartition != reference.QuotaPartition {
		return nil, ErrDurableRoutingPinMismatch
	}
	return set, nil
}
