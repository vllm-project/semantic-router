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
)

var (
	ErrManagedRouterRegistryClosed     = errors.New("managed router registry is closed")
	ErrManagedRouterUnavailable        = errors.New("managed router generation is unavailable")
	ErrManagedRouterPinMismatch        = errors.New("managed router generation pin does not match")
	ErrManagedRouterStaleGeneration    = errors.New("managed router generation is stale")
	ErrManagedRouterPublicationCorrupt = errors.New("managed router publication is corrupt")
)

// ManagedRouterGenerationPin is the complete request-time identity of one
// immutable namespace router generation. Callers must supply every field; the
// registry never falls back to a default namespace or a newer generation.
type ManagedRouterGenerationPin struct {
	NamespaceID      string
	QuotaPartition   string
	PublicationID    string
	RuntimeEpoch     uint64
	SnapshotRevision int64
	RoutingDigest    string
}

func (pin ManagedRouterGenerationPin) validate() error {
	reference := accesspublisher.NamespacePublication{
		NamespaceID: pin.NamespaceID, QuotaPartition: pin.QuotaPartition,
	}
	if err := reference.Validate(); err != nil {
		return fmt.Errorf("%w: %w", ErrManagedRouterPinMismatch, err)
	}
	if strings.TrimSpace(pin.PublicationID) == "" || pin.RuntimeEpoch == 0 || pin.SnapshotRevision <= 0 ||
		!managedRouterDigestValid(pin.RoutingDigest) {
		return fmt.Errorf("%w: publication, epoch, revision, and digest are required", ErrManagedRouterPinMismatch)
	}
	return nil
}

// ManagedRouterRegistryOptions contains process-owned bootstrap state. The
// bootstrap configuration and dependencies are borrowed as immutable values;
// every warmed generation receives a newly compiled RouterConfig.
type ManagedRouterRegistryOptions struct {
	BootstrapConfig *config.RouterConfig
	Dependencies    RuntimeDependencies
}

type managedRouterBuilder func(*config.RouterConfig, RuntimeDependencies) (*OpenAIRouter, error)

type managedRoutingSnapshotKey struct {
	namespaceID string
	revision    int64
}

type managedRetainedRoutingSnapshot struct {
	digest      string
	snapshot    *routingsnapshot.Snapshot
	generations map[*managedRouterGeneration]struct{}
}

// ManagedRouterRegistry owns the immutable OpenAIRouter generations loaded by
// publicationreplica.Manager. Namespace state is isolated so warming one
// namespace never serializes generation work for another namespace.
type ManagedRouterRegistry struct {
	bootstrap    *config.RouterConfig
	dependencies RuntimeDependencies
	build        managedRouterBuilder

	closed            atomic.Bool
	mu                sync.Mutex
	sets              map[string]*managedRouterNamespace
	retainedSnapshots map[managedRoutingSnapshotKey]*managedRetainedRoutingSnapshot

	generationWait sync.WaitGroup
	closeMu        sync.Mutex
	closeErrors    []error
	closeDone      chan struct{}
	finalCloseErr  error
}

var (
	_ publicationreplica.SnapshotLifecycle = (*ManagedRouterRegistry)(nil)
	_ backendinvoker.RoutingSnapshotSource = (*ManagedRouterRegistry)(nil)
)

// NewManagedRouterRegistry creates an empty namespace generation registry.
func NewManagedRouterRegistry(options ManagedRouterRegistryOptions) (*ManagedRouterRegistry, error) {
	return newManagedRouterRegistry(options, buildOpenAIRouterFromConfigWithDependencies)
}

func newManagedRouterRegistry(
	options ManagedRouterRegistryOptions,
	build managedRouterBuilder,
) (*ManagedRouterRegistry, error) {
	if options.BootstrapConfig == nil {
		return nil, fmt.Errorf("managed router bootstrap configuration is required")
	}
	if options.BootstrapConfig.ControlPlane.Mode != config.ControlPlaneModeManaged {
		return nil, fmt.Errorf("managed router registry requires managed control-plane mode")
	}
	if build == nil {
		return nil, fmt.Errorf("managed router builder is required")
	}
	if err := options.Dependencies.validate(options.BootstrapConfig); err != nil {
		return nil, err
	}
	// Isolate top-level mutations while retaining the bootstrap's explicitly
	// immutable nested service state. CompileManagedRoutingSnapshot exports and
	// rebuilds that state for every generation.
	bootstrap := *options.BootstrapConfig
	return &ManagedRouterRegistry{
		bootstrap: &bootstrap, dependencies: options.Dependencies, build: build,
		sets:              make(map[string]*managedRouterNamespace),
		retainedSnapshots: make(map[managedRoutingSnapshotKey]*managedRetainedRoutingSnapshot),
		closeDone:         make(chan struct{}),
	}, nil
}

// Warm strictly verifies and builds one publication candidate without making
// it visible to request admission.
func (registry *ManagedRouterRegistry) Warm(
	ctx context.Context,
	publication accesspublisher.LoadedRoutingPublication,
) error {
	if registry == nil || registry.closed.Load() {
		return ErrManagedRouterRegistryClosed
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
		return ErrManagedRouterUnavailable
	}
	if registry.closed.Load() {
		set.mu.Unlock()
		return ErrManagedRouterRegistryClosed
	}
	if existing := set.generations[pin.PublicationID]; existing != nil {
		err := matchingManagedGeneration(existing, publication.Identity, pin)
		set.mu.Unlock()
		return err
	}
	if newest := newestManagedGeneration(set); newest != nil {
		switch compareManagedGenerationOrder(publication.Identity, newest.identity) {
		case -1:
			set.mu.Unlock()
			return fmt.Errorf("%w: %s precedes %s", ErrManagedRouterStaleGeneration,
				publication.Identity.PublicationID, newest.identity.PublicationID)
		case 0:
			set.mu.Unlock()
			return fmt.Errorf("%w: revision already names another publication", ErrManagedRouterPublicationCorrupt)
		}
	}

	router, buildErr := registry.build(compiled, registry.dependencies)
	if buildErr != nil {
		set.mu.Unlock()
		if router != nil {
			_ = router.Close()
		}
		return fmt.Errorf("build managed router generation: %w", buildErr)
	}
	if router == nil || router.Config != compiled || router.Config.DocumentHash != snapshot.Digest {
		set.mu.Unlock()
		if router != nil {
			_ = router.Close()
		}
		return fmt.Errorf("%w: router builder returned a different configuration", ErrManagedRouterPublicationCorrupt)
	}
	if err := contextError(ctx); err != nil || registry.closed.Load() {
		set.mu.Unlock()
		_ = router.Close()
		if err != nil {
			return err
		}
		return ErrManagedRouterRegistryClosed
	}
	registry.mu.Lock()
	if registry.closed.Load() {
		registry.mu.Unlock()
		set.mu.Unlock()
		_ = router.Close()
		return ErrManagedRouterRegistryClosed
	}
	registry.generationWait.Add(1)
	set.generations[pin.PublicationID] = newManagedRouterGeneration(
		publication.Identity, pin, router, snapshot, registry.generationClosed,
	)
	registry.mu.Unlock()
	set.mu.Unlock()
	return nil
}

// Activate atomically selects an already verified generation for one
// namespace. The method is independently idempotent and therefore first warms
// the supplied value if a replica is recovering without an in-process cache.
func (registry *ManagedRouterRegistry) Activate(
	ctx context.Context,
	publication accesspublisher.LoadedRoutingPublication,
) error {
	if !publication.Identity.Activated() {
		return fmt.Errorf("%w: publication is not active", ErrManagedRouterPinMismatch)
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

	var retired []*managedRouterGeneration
	set.mu.Lock()
	if set.removed {
		set.mu.Unlock()
		return ErrManagedRouterUnavailable
	}
	if registry.closed.Load() {
		set.mu.Unlock()
		return ErrManagedRouterRegistryClosed
	}
	generation := set.generations[pin.PublicationID]
	if generation == nil {
		set.mu.Unlock()
		return ErrManagedRouterUnavailable
	}
	if err := matchingManagedGeneration(generation, publication.Identity, pin); err != nil {
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
		switch compareManagedGenerationOrder(publication.Identity, set.active.identity) {
		case -1:
			set.mu.Unlock()
			return fmt.Errorf("%w: active publication %s is newer", ErrManagedRouterStaleGeneration,
				set.active.identity.PublicationID)
		case 0:
			set.mu.Unlock()
			return fmt.Errorf("%w: active revision names another publication", ErrManagedRouterPublicationCorrupt)
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
func (registry *ManagedRouterRegistry) Snapshot(
	ctx context.Context,
	pin routingcontext.Generation,
) (*routingsnapshot.Snapshot, error) {
	if err := contextError(ctx); err != nil {
		return nil, err
	}
	if registry == nil {
		return nil, ErrManagedRouterRegistryClosed
	}
	if pin.Validate() != nil || strings.ContainsRune(pin.NamespaceID, 0) {
		return nil, fmt.Errorf("%w: complete routing generation is required", ErrManagedRouterPinMismatch)
	}
	key := managedRoutingSnapshotKey{namespaceID: pin.NamespaceID, revision: pin.SnapshotRevision}
	registry.mu.Lock()
	retained := registry.retainedSnapshots[key]
	if retained == nil {
		closed := registry.closed.Load()
		registry.mu.Unlock()
		if closed {
			return nil, ErrManagedRouterRegistryClosed
		}
		return nil, ErrManagedRouterUnavailable
	}
	if retained.snapshot == nil || retained.snapshot.NamespaceID != pin.NamespaceID ||
		retained.snapshot.Revision != pin.SnapshotRevision || retained.snapshot.Digest != retained.digest {
		registry.mu.Unlock()
		return nil, ErrManagedRouterPublicationCorrupt
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
		return nil, ErrManagedRouterPinMismatch
	}
	if !readable {
		if closed {
			return nil, ErrManagedRouterRegistryClosed
		}
		return nil, ErrManagedRouterUnavailable
	}
	return snapshot, nil
}

// retainRoutingSnapshotLocked makes one activated generation discoverable by
// backend dispatch. registry.mu and the namespace lock must both be held so
// activation and exact snapshot visibility share one linearization point.
func (registry *ManagedRouterRegistry) retainRoutingSnapshotLocked(
	generation *managedRouterGeneration,
) error {
	if registry.closed.Load() {
		return ErrManagedRouterRegistryClosed
	}
	if generation == nil || generation.snapshot == nil ||
		generation.snapshot.NamespaceID != generation.pin.NamespaceID ||
		generation.snapshot.Revision != generation.pin.SnapshotRevision ||
		!managedRouterDigestValid(generation.snapshot.Digest) {
		return fmt.Errorf("%w: generation snapshot identity is invalid", ErrManagedRouterPublicationCorrupt)
	}
	key := managedRoutingSnapshotKey{
		namespaceID: generation.pin.NamespaceID,
		revision:    generation.pin.SnapshotRevision,
	}
	retained := registry.retainedSnapshots[key]
	if retained == nil {
		registry.retainedSnapshots[key] = &managedRetainedRoutingSnapshot{
			digest: generation.snapshot.Digest, snapshot: generation.snapshot,
			generations: map[*managedRouterGeneration]struct{}{generation: {}},
		}
		return nil
	}
	if retained.snapshot == nil || retained.digest != generation.snapshot.Digest ||
		retained.snapshot.NamespaceID != key.namespaceID || retained.snapshot.Revision != key.revision ||
		retained.snapshot.Digest != retained.digest {
		return fmt.Errorf(
			"%w: namespace revision names different routing snapshots",
			ErrManagedRouterPublicationCorrupt,
		)
	}
	retained.generations[generation] = struct{}{}
	return nil
}

func (registry *ManagedRouterRegistry) releaseRoutingSnapshot(generation *managedRouterGeneration) {
	if registry == nil || generation == nil {
		return
	}
	key := managedRoutingSnapshotKey{
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
func (registry *ManagedRouterRegistry) Acquire(pin ManagedRouterGenerationPin) (*ManagedRouterLease, error) {
	if registry == nil || registry.closed.Load() {
		return nil, ErrManagedRouterRegistryClosed
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
		return nil, ErrManagedRouterUnavailable
	}
	if generation.pin != pin {
		return nil, ErrManagedRouterPinMismatch
	}
	if !generation.acquire() {
		return nil, ErrManagedRouterUnavailable
	}
	return &ManagedRouterLease{
		Router: generation.router, Pin: generation.pin, release: generation.release,
	}, nil
}

// Remove retires every generation in exactly one namespace partition. It is
// idempotent after successful removal and never removes a namespace through a
// mismatched quota partition.
func (registry *ManagedRouterRegistry) Remove(
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
		return ErrManagedRouterRegistryClosed
	}
	registry.mu.Lock()
	set := registry.sets[reference.NamespaceID]
	if set != nil && set.reference.QuotaPartition != reference.QuotaPartition {
		registry.mu.Unlock()
		return ErrManagedRouterPinMismatch
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
func (registry *ManagedRouterRegistry) Close() error {
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
	sets := make([]*managedRouterNamespace, 0, len(registry.sets))
	for _, set := range registry.sets {
		sets = append(sets, set)
	}
	registry.sets = make(map[string]*managedRouterNamespace)
	registry.mu.Unlock()

	var generations []*managedRouterGeneration
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

func (registry *ManagedRouterRegistry) namespace(
	reference accesspublisher.NamespacePublication,
	create bool,
) (*managedRouterNamespace, error) {
	if err := reference.Validate(); err != nil {
		return nil, err
	}
	if registry.closed.Load() {
		return nil, ErrManagedRouterRegistryClosed
	}
	registry.mu.Lock()
	defer registry.mu.Unlock()
	if registry.closed.Load() {
		return nil, ErrManagedRouterRegistryClosed
	}
	set := registry.sets[reference.NamespaceID]
	if set == nil {
		if !create {
			return nil, ErrManagedRouterUnavailable
		}
		set = &managedRouterNamespace{
			reference: reference, generations: make(map[string]*managedRouterGeneration),
		}
		registry.sets[reference.NamespaceID] = set
		return set, nil
	}
	if set.reference.QuotaPartition != reference.QuotaPartition {
		return nil, ErrManagedRouterPinMismatch
	}
	return set, nil
}
