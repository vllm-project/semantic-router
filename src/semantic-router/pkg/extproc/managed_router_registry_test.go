package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func TestManagedRouterRegistryWarmsActivatesAndPinsExactly(t *testing.T) {
	registry, builder := newManagedRouterRegistryForTest(t)
	publication := managedRouterPublication(t, "namespace-a", "partition-a", 1, 7, "a")
	active := managedRouterPublicationState(publication, accesspublisher.PublicationStateActive)

	if err := registry.Warm(context.Background(), publication); err != nil {
		t.Fatalf("Warm() error = %v", err)
	}
	if err := registry.Warm(context.Background(), publication); err != nil {
		t.Fatalf("idempotent Warm() error = %v", err)
	}
	pin := managedRouterPin(active)
	if _, err := registry.Acquire(pin); !errors.Is(err, ErrManagedRouterUnavailable) {
		t.Fatalf("Acquire() before activation error = %v", err)
	}
	if err := registry.Activate(context.Background(), active); err != nil {
		t.Fatalf("Activate() error = %v", err)
	}
	if err := registry.Activate(context.Background(), active); err != nil {
		t.Fatalf("idempotent Activate() error = %v", err)
	}
	if got := builder.calls.Load(); got != 1 {
		t.Fatalf("router builds = %d, want 1", got)
	}

	lease, err := registry.Acquire(pin)
	if err != nil {
		t.Fatalf("Acquire() error = %v", err)
	}
	if lease.Router == nil || lease.Router.Config == nil || lease.Router.Config.DocumentHash != publication.Snapshot.Digest || lease.Pin != pin {
		t.Fatalf("lease = %+v", lease)
	}
	wrong := pin
	wrong.RoutingDigest = strings.Repeat("f", 64)
	if _, err := registry.Acquire(wrong); !errors.Is(err, ErrManagedRouterPinMismatch) {
		t.Fatalf("Acquire() mismatched digest error = %v", err)
	}
	wrong = pin
	wrong.QuotaPartition = "partition-b"
	if _, err := registry.Acquire(wrong); !errors.Is(err, ErrManagedRouterPinMismatch) {
		t.Fatalf("Acquire() mismatched partition error = %v", err)
	}
	lease.Release()
	lease.Release()
	if err := registry.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}

func TestManagedRouterRegistryPinsRoutingDocumentDigestSeparatelyFromSnapshotDigest(t *testing.T) {
	registry, _ := newManagedRouterRegistryForTest(t)
	publication := managedRouterPublicationState(
		managedRouterPublication(t, "namespace-a", "partition-a", 1, 7, "a"),
		accesspublisher.PublicationStateActive,
	)
	if publication.Identity.RoutingDigest == publication.Snapshot.Digest {
		t.Fatal("compiled routing document and nested snapshot unexpectedly share a digest")
	}
	if err := registry.Activate(context.Background(), publication); err != nil {
		t.Fatal(err)
	}
	resolver := backendinvoker.SnapshotPlanResolver{Source: registry}
	capability := backendinvoker.DispatchCapability{
		NamespaceID: publication.Identity.NamespaceID, QuotaPartition: publication.Identity.QuotaPartition,
		PublicationID: publication.Identity.PublicationID, RuntimeEpoch: publication.Identity.RuntimeEpoch,
		RoutingRevision: publication.Snapshot.Revision, RoutingDigest: publication.Identity.RoutingDigest,
		Candidates: []backendinvoker.DispatchCandidate{{
			DispatchID: "dispatch-chat", DispatchType: "primary", DispatchPlanDigest: strings.Repeat("a", 64),
			ModelID: "model-chat", ModelRevision: 1,
		}},
	}
	plans, err := resolver.ResolvePlans(context.Background(), capability)
	if err != nil || len(plans.Candidates) != 1 || plans.Candidates[0].ModelID != "model-chat" {
		t.Fatalf("ResolvePlans() = %+v, %v", plans, err)
	}
	for name, mutate := range map[string]func(*backendinvoker.DispatchCapability){
		"routing digest": func(value *backendinvoker.DispatchCapability) { value.RoutingDigest = strings.Repeat("f", 64) },
		"publication":    func(value *backendinvoker.DispatchCapability) { value.PublicationID = "other-publication" },
		"runtime epoch":  func(value *backendinvoker.DispatchCapability) { value.RuntimeEpoch++ },
		"partition":      func(value *backendinvoker.DispatchCapability) { value.QuotaPartition = "other-partition" },
	} {
		t.Run(name, func(t *testing.T) {
			wrong := capability
			mutate(&wrong)
			if _, err := resolver.ResolvePlans(context.Background(), wrong); err == nil {
				t.Fatal("ResolvePlans() accepted a pin outside the active generation")
			}
		})
	}
}

func TestManagedRouterRegistryRetiresOnlyAfterLeasesDrain(t *testing.T) {
	registry, builder := newManagedRouterRegistryForTest(t)
	first := managedRouterPublicationState(
		managedRouterPublication(t, "namespace-a", "partition-a", 1, 7, "a"),
		accesspublisher.PublicationStateActive,
	)
	second := managedRouterPublicationState(
		managedRouterPublication(t, "namespace-a", "partition-a", 2, 7, "b"),
		accesspublisher.PublicationStateActive,
	)
	if err := registry.Activate(context.Background(), first); err != nil {
		t.Fatal(err)
	}
	firstLease, testManagedRouterRegistryRetiresOnlyAfterLeasesDrainErr := registry.Acquire(managedRouterPin(first))
	if testManagedRouterRegistryRetiresOnlyAfterLeasesDrainErr != nil {
		t.Fatal(testManagedRouterRegistryRetiresOnlyAfterLeasesDrainErr)
	}
	if err := registry.Activate(context.Background(), second); err != nil {
		t.Fatal(err)
	}
	if got := builder.closed(first.Snapshot.Digest); got != 0 {
		t.Fatalf("retired generation close calls = %d while leased", got)
	}
	firstLease.Release()
	firstLease.Release()
	waitManagedRouterRegistry(t, "retired router close", func() bool {
		return builder.closed(first.Snapshot.Digest) == 1
	})

	secondLease, testManagedRouterRegistryRetiresOnlyAfterLeasesDrainErr := registry.Acquire(managedRouterPin(second))
	if testManagedRouterRegistryRetiresOnlyAfterLeasesDrainErr != nil {
		t.Fatal(testManagedRouterRegistryRetiresOnlyAfterLeasesDrainErr)
	}
	closed := make(chan error, 1)
	go func() { closed <- registry.Close() }()
	select {
	case err := <-closed:
		t.Fatalf("Close() returned before active lease drained: %v", err)
	case <-time.After(25 * time.Millisecond):
	}
	waitManagedRouterRegistry(t, "registry close admission stop", registry.closed.Load)
	closingSnapshot, testManagedRouterRegistryRetiresOnlyAfterLeasesDrainErr := registry.Snapshot(context.Background(), managedRoutingSnapshotPin(second))
	if testManagedRouterRegistryRetiresOnlyAfterLeasesDrainErr != nil || closingSnapshot.Digest != second.Snapshot.Digest {
		t.Fatalf("Snapshot() while Close drains an admitted request = (%+v, %v)", closingSnapshot, testManagedRouterRegistryRetiresOnlyAfterLeasesDrainErr)
	}
	secondLease.Release()
	select {
	case err := <-closed:
		if err != nil {
			t.Fatalf("Close() error = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Close() did not finish after lease release")
	}
	if got := builder.closed(second.Snapshot.Digest); got != 1 {
		t.Fatalf("active generation close calls = %d, want 1", got)
	}
	if _, err := registry.Acquire(managedRouterPin(second)); !errors.Is(err, ErrManagedRouterRegistryClosed) {
		t.Fatalf("Acquire() after close error = %v", err)
	}
	if _, err := registry.Snapshot(context.Background(), managedRoutingSnapshotPin(second)); !errors.Is(err, ErrManagedRouterRegistryClosed) {
		t.Fatalf("Snapshot() after close error = %v", err)
	}
	if err := registry.Close(); err != nil {
		t.Fatalf("idempotent Close() error = %v", err)
	}
}

func TestManagedRouterRegistrySnapshotRetainsRolledGenerationUntilLeaseDrain(t *testing.T) {
	registry, builder := newManagedRouterRegistryForTest(t)
	first := managedRouterPublicationState(
		managedRouterPublication(t, "namespace-a", "partition-a", 1, 7, "a"),
		accesspublisher.PublicationStateActive,
	)
	secondCandidate := managedRouterPublication(t, "namespace-a", "partition-a", 2, 7, "b")
	second := managedRouterPublicationState(secondCandidate, accesspublisher.PublicationStateActive)
	if err := registry.Activate(context.Background(), first); err != nil {
		t.Fatal(err)
	}
	if err := registry.Warm(context.Background(), secondCandidate); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.Snapshot(context.Background(), managedRoutingSnapshotPin(second)); !errors.Is(err, ErrManagedRouterUnavailable) {
		t.Fatalf("Snapshot() exposed a warmed generation: %v", err)
	}
	firstLease, testManagedRouterRegistrySnapshotRetainsRolledGenerationUntilLeaseDrainErr := registry.Acquire(managedRouterPin(first))
	if testManagedRouterRegistrySnapshotRetainsRolledGenerationUntilLeaseDrainErr != nil {
		t.Fatal(testManagedRouterRegistrySnapshotRetainsRolledGenerationUntilLeaseDrainErr)
	}
	held, testManagedRouterRegistrySnapshotRetainsRolledGenerationUntilLeaseDrainErr := registry.Snapshot(context.Background(), managedRoutingSnapshotPin(first))
	if testManagedRouterRegistrySnapshotRetainsRolledGenerationUntilLeaseDrainErr != nil || held.Digest != first.Snapshot.Digest {
		t.Fatalf("Snapshot() active generation = (%+v, %v)", held, testManagedRouterRegistrySnapshotRetainsRolledGenerationUntilLeaseDrainErr)
	}
	first.Snapshot.Models[0].Name = "caller-mutated"
	model, found := held.Model("model-chat")
	if !found || model.Name != "local/chat-a" {
		t.Fatalf("registry retained caller-owned snapshot state: %+v", model)
	}
	if err := registry.Activate(context.Background(), second); err != nil {
		t.Fatal(err)
	}
	retired, testManagedRouterRegistrySnapshotRetainsRolledGenerationUntilLeaseDrainErr := registry.Snapshot(context.Background(), managedRoutingSnapshotPin(first))
	if testManagedRouterRegistrySnapshotRetainsRolledGenerationUntilLeaseDrainErr != nil || retired != held {
		t.Fatalf("Snapshot() retired leased generation = (%+v, %v)", retired, testManagedRouterRegistrySnapshotRetainsRolledGenerationUntilLeaseDrainErr)
	}
	current, testManagedRouterRegistrySnapshotRetainsRolledGenerationUntilLeaseDrainErr := registry.Snapshot(context.Background(), managedRoutingSnapshotPin(second))
	if testManagedRouterRegistrySnapshotRetainsRolledGenerationUntilLeaseDrainErr != nil || current.Digest != second.Snapshot.Digest {
		t.Fatalf("Snapshot() current generation = (%+v, %v)", current, testManagedRouterRegistrySnapshotRetainsRolledGenerationUntilLeaseDrainErr)
	}
	wrongNamespace := managedRoutingSnapshotPin(first)
	wrongNamespace.NamespaceID = "namespace-b"
	if _, err := registry.Snapshot(context.Background(), wrongNamespace); !errors.Is(err, ErrManagedRouterUnavailable) {
		t.Fatalf("Snapshot() crossed namespaces: %v", err)
	}
	wrongRevision := managedRoutingSnapshotPin(first)
	wrongRevision.SnapshotRevision = 99
	if _, err := registry.Snapshot(context.Background(), wrongRevision); !errors.Is(err, ErrManagedRouterUnavailable) {
		t.Fatalf("Snapshot() fell back across revisions: %v", err)
	}
	firstLease.Release()
	if _, err := registry.Snapshot(context.Background(), managedRoutingSnapshotPin(first)); !errors.Is(err, ErrManagedRouterUnavailable) {
		t.Fatalf("Snapshot() remained readable after its lease drained: %v", err)
	}
	waitManagedRouterRegistry(t, "retired router close", func() bool {
		return builder.closed(first.Snapshot.Digest) == 1
	})
	model, found = held.Model("model-chat")
	if !found || model.Name != "local/chat-a" {
		t.Fatalf("returned immutable snapshot became unsafe after generation close: %+v", model)
	}
	if err := registry.Close(); err != nil {
		t.Fatal(err)
	}
	registry.mu.Lock()
	retainedCount := len(registry.retainedSnapshots)
	registry.mu.Unlock()
	if retainedCount != 0 {
		t.Fatalf("retained snapshot entries after Close = %d", retainedCount)
	}
}

func TestManagedRouterRegistrySnapshotRejectsRevisionDigestCollision(t *testing.T) {
	registry, _ := newManagedRouterRegistryForTest(t)
	first := managedRouterPublicationState(
		managedRouterPublication(t, "namespace-a", "partition-a", 1, 7, "a"),
		accesspublisher.PublicationStateActive,
	)
	collision := managedRouterPublicationState(
		managedRouterPublication(t, "namespace-a", "partition-a", 1, 8, "b"),
		accesspublisher.PublicationStateActive,
	)
	if first.Snapshot.Digest == collision.Snapshot.Digest {
		t.Fatal("collision fixture unexpectedly has the same digest")
	}
	if err := registry.Activate(context.Background(), first); err != nil {
		t.Fatal(err)
	}
	lease, testManagedRouterRegistrySnapshotRejectsRevisionDigestCollisionErr := registry.Acquire(managedRouterPin(first))
	if testManagedRouterRegistrySnapshotRejectsRevisionDigestCollisionErr != nil {
		t.Fatal(testManagedRouterRegistrySnapshotRejectsRevisionDigestCollisionErr)
	}
	if err := registry.Activate(context.Background(), collision); !errors.Is(err, ErrManagedRouterPublicationCorrupt) {
		t.Fatalf("Activate() revision/digest collision error = %v", err)
	}
	snapshot, testManagedRouterRegistrySnapshotRejectsRevisionDigestCollisionErr := registry.Snapshot(context.Background(), managedRoutingSnapshotPin(first))
	if testManagedRouterRegistrySnapshotRejectsRevisionDigestCollisionErr != nil || snapshot.Digest != first.Snapshot.Digest {
		t.Fatalf("collision replaced exact snapshot = (%+v, %v)", snapshot, testManagedRouterRegistrySnapshotRejectsRevisionDigestCollisionErr)
	}
	lease.Release()
	if err := registry.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestManagedRouterRegistryIsolatesNamespacesAndRemovesExactly(t *testing.T) {
	registry, builder := newManagedRouterRegistryForTest(t)
	first := managedRouterPublicationState(
		managedRouterPublication(t, "namespace-a", "partition-a", 1, 7, "a"),
		accesspublisher.PublicationStateActive,
	)
	second := managedRouterPublicationState(
		managedRouterPublication(t, "namespace-b", "partition-b", 1, 7, "b"),
		accesspublisher.PublicationStateActive,
	)
	if err := registry.Activate(context.Background(), first); err != nil {
		t.Fatal(err)
	}
	if err := registry.Activate(context.Background(), second); err != nil {
		t.Fatal(err)
	}
	firstLease, testManagedRouterRegistryIsolatesNamespacesAndRemovesExactlyErr := registry.Acquire(managedRouterPin(first))
	if testManagedRouterRegistryIsolatesNamespacesAndRemovesExactlyErr != nil {
		t.Fatal(testManagedRouterRegistryIsolatesNamespacesAndRemovesExactlyErr)
	}
	secondLease, testManagedRouterRegistryIsolatesNamespacesAndRemovesExactlyErr := registry.Acquire(managedRouterPin(second))
	if testManagedRouterRegistryIsolatesNamespacesAndRemovesExactlyErr != nil {
		t.Fatal(testManagedRouterRegistryIsolatesNamespacesAndRemovesExactlyErr)
	}
	if firstLease.Router == secondLease.Router {
		t.Fatal("namespaces shared a router generation")
	}
	firstSnapshot, testManagedRouterRegistryIsolatesNamespacesAndRemovesExactlyErr := registry.Snapshot(context.Background(), managedRoutingSnapshotPin(first))
	if testManagedRouterRegistryIsolatesNamespacesAndRemovesExactlyErr != nil || firstSnapshot.Digest != first.Snapshot.Digest {
		t.Fatalf("Snapshot() namespace-a = (%+v, %v)", firstSnapshot, testManagedRouterRegistryIsolatesNamespacesAndRemovesExactlyErr)
	}
	secondSnapshot, testManagedRouterRegistryIsolatesNamespacesAndRemovesExactlyErr := registry.Snapshot(context.Background(), managedRoutingSnapshotPin(second))
	if testManagedRouterRegistryIsolatesNamespacesAndRemovesExactlyErr != nil || secondSnapshot.Digest != second.Snapshot.Digest {
		t.Fatalf("Snapshot() namespace-b = (%+v, %v)", secondSnapshot, testManagedRouterRegistryIsolatesNamespacesAndRemovesExactlyErr)
	}
	secondLease.Release()

	wrongReference := accesspublisher.NamespacePublication{NamespaceID: "namespace-a", QuotaPartition: "partition-b"}
	if err := registry.Remove(context.Background(), wrongReference); !errors.Is(err, ErrManagedRouterPinMismatch) {
		t.Fatalf("Remove() mismatched partition error = %v", err)
	}
	probe, testManagedRouterRegistryIsolatesNamespacesAndRemovesExactlyErr := registry.Acquire(managedRouterPin(first))
	if testManagedRouterRegistryIsolatesNamespacesAndRemovesExactlyErr != nil {
		t.Fatalf("mismatched removal affected namespace: %v", testManagedRouterRegistryIsolatesNamespacesAndRemovesExactlyErr)
	}
	probe.Release()
	reference := accesspublisher.NamespacePublication{NamespaceID: "namespace-a", QuotaPartition: "partition-a"}
	if err := registry.Remove(context.Background(), reference); err != nil {
		t.Fatalf("Remove() error = %v", err)
	}
	if err := registry.Remove(context.Background(), reference); err != nil {
		t.Fatalf("idempotent Remove() error = %v", err)
	}
	if _, err := registry.Acquire(managedRouterPin(first)); !errors.Is(err, ErrManagedRouterUnavailable) {
		t.Fatalf("Acquire() removed namespace error = %v", err)
	}
	if got := builder.closed(first.Snapshot.Digest); got != 0 {
		t.Fatalf("removed generation close calls = %d while leased", got)
	}
	removedSnapshot, testManagedRouterRegistryIsolatesNamespacesAndRemovesExactlyErr := registry.Snapshot(context.Background(), managedRoutingSnapshotPin(first))
	if testManagedRouterRegistryIsolatesNamespacesAndRemovesExactlyErr != nil || removedSnapshot.Digest != first.Snapshot.Digest {
		t.Fatalf("Snapshot() removed but leased generation = (%+v, %v)", removedSnapshot, testManagedRouterRegistryIsolatesNamespacesAndRemovesExactlyErr)
	}
	firstLease.Release()
	if _, err := registry.Snapshot(context.Background(), managedRoutingSnapshotPin(first)); !errors.Is(err, ErrManagedRouterUnavailable) {
		t.Fatalf("Snapshot() removed generation after lease drain = %v", err)
	}
	waitManagedRouterRegistry(t, "removed router close", func() bool {
		return builder.closed(first.Snapshot.Digest) == 1
	})
	remaining, testManagedRouterRegistryIsolatesNamespacesAndRemovesExactlyErr := registry.Acquire(managedRouterPin(second))
	if testManagedRouterRegistryIsolatesNamespacesAndRemovesExactlyErr != nil {
		t.Fatalf("removing namespace-a affected namespace-b: %v", testManagedRouterRegistryIsolatesNamespacesAndRemovesExactlyErr)
	}
	remaining.Release()
	if err := registry.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestManagedRouterRegistryRejectsCorruptConflictingAndStalePublications(t *testing.T) {
	registry, builder := newManagedRouterRegistryForTest(t)
	first := managedRouterPublication(t, "namespace-a", "partition-a", 1, 7, "a")

	corrupt := first
	corrupt.Snapshot.Digest = strings.Repeat("0", 64)
	if err := registry.Warm(context.Background(), corrupt); !errors.Is(err, ErrManagedRouterPublicationCorrupt) {
		t.Fatalf("Warm() corrupt snapshot error = %v", err)
	}
	mismatched := first
	mismatched.Identity.RoutingDigest = strings.Repeat("f", 64)
	if err := registry.Warm(context.Background(), mismatched); !errors.Is(err, ErrManagedRouterPublicationCorrupt) {
		t.Fatalf("Warm() mismatched document error = %v", err)
	}
	if got := builder.calls.Load(); got != 0 {
		t.Fatalf("invalid publications built %d routers", got)
	}

	activeFirst := managedRouterPublicationState(first, accesspublisher.PublicationStateActive)
	if err := registry.Activate(context.Background(), activeFirst); err != nil {
		t.Fatal(err)
	}
	conflict := managedRouterPublication(t, "namespace-a", "partition-a", 1, 7, "conflict")
	if err := registry.Warm(context.Background(), conflict); !errors.Is(err, ErrManagedRouterPublicationCorrupt) {
		t.Fatalf("Warm() conflicting revision error = %v", err)
	}
	second := managedRouterPublicationState(
		managedRouterPublication(t, "namespace-a", "partition-a", 2, 7, "b"),
		accesspublisher.PublicationStateActive,
	)
	if err := registry.Activate(context.Background(), second); err != nil {
		t.Fatal(err)
	}
	if err := registry.Warm(context.Background(), first); !errors.Is(err, ErrManagedRouterStaleGeneration) {
		t.Fatalf("Warm() stale publication error = %v", err)
	}
	if err := registry.Activate(context.Background(), activeFirst); !errors.Is(err, ErrManagedRouterStaleGeneration) {
		t.Fatalf("Activate() stale publication error = %v", err)
	}
	if err := registry.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestManagedRouterRegistryConcurrentAcquireAndActivation(t *testing.T) {
	registry, _ := newManagedRouterRegistryForTest(t)
	first := managedRouterPublicationState(
		managedRouterPublication(t, "namespace-a", "partition-a", 1, 7, "a"),
		accesspublisher.PublicationStateActive,
	)
	second := managedRouterPublicationState(
		managedRouterPublication(t, "namespace-a", "partition-a", 2, 7, "b"),
		accesspublisher.PublicationStateActive,
	)
	if err := registry.Activate(context.Background(), first); err != nil {
		t.Fatal(err)
	}

	var workers sync.WaitGroup
	workers.Add(16)
	for index := 0; index < 16; index++ {
		go func() {
			defer workers.Done()
			for attempt := 0; attempt < 100; attempt++ {
				lease, err := registry.Acquire(managedRouterPin(first))
				if err == nil {
					lease.Release()
					continue
				}
				if !errors.Is(err, ErrManagedRouterPinMismatch) && !errors.Is(err, ErrManagedRouterUnavailable) {
					t.Errorf("concurrent Acquire() error = %v", err)
					return
				}
			}
		}()
	}
	if err := registry.Activate(context.Background(), second); err != nil {
		t.Fatal(err)
	}
	workers.Wait()
	lease, err := registry.Acquire(managedRouterPin(second))
	if err != nil {
		t.Fatalf("Acquire() new generation error = %v", err)
	}
	lease.Release()
	if err := registry.Close(); err != nil {
		t.Fatal(err)
	}
}

type managedRouterTestBuilder struct {
	calls atomic.Int64
	mu    sync.Mutex
	close map[string]*atomic.Int64
}

func (builder *managedRouterTestBuilder) build(
	cfg *config.RouterConfig,
	_ RuntimeDependencies,
) (*OpenAIRouter, error) {
	builder.calls.Add(1)
	counter := &atomic.Int64{}
	builder.mu.Lock()
	builder.close[cfg.DocumentHash] = counter
	builder.mu.Unlock()
	return &OpenAIRouter{
		Config: cfg,
		lookupTableCancel: func() {
			counter.Add(1)
		},
	}, nil
}

func (builder *managedRouterTestBuilder) closed(digest string) int64 {
	builder.mu.Lock()
	counter := builder.close[digest]
	builder.mu.Unlock()
	if counter == nil {
		return 0
	}
	return counter.Load()
}

func newManagedRouterRegistryForTest(t *testing.T) (*ManagedRouterRegistry, *managedRouterTestBuilder) {
	t.Helper()
	base, err := config.ParseYAMLBytes([]byte(managedRouterBootstrapYAML))
	if err != nil {
		t.Fatalf("parse managed bootstrap: %v", err)
	}
	base.SkipExternalAssetValidation = true
	builder := &managedRouterTestBuilder{close: make(map[string]*atomic.Int64)}
	registry, err := newManagedRouterRegistry(ManagedRouterRegistryOptions{
		BootstrapConfig: base,
		Dependencies: RuntimeDependencies{
			InferenceAccess:      &fakeInferenceAccess{},
			DispatchCapabilities: dispatchCapabilityRuntimeStub{metered: true},
			OutcomeFeedback:      &outcomeRuntimeStub{},
			OutcomeProjection:    outcomeProjectionRuntimeStub{},
			ResponseTerminals:    backendinvoker.NewLocalResponseTerminalStore(),
			ProtocolCodecs:       protocolcodec.NewBuiltinRegistry(),
		},
	}, builder.build)
	if err != nil {
		t.Fatalf("NewManagedRouterRegistry() error = %v", err)
	}
	t.Cleanup(func() { _ = registry.Close() })
	return registry, builder
}

func managedRouterPublication(
	t *testing.T,
	namespaceID string,
	partition string,
	revision uint64,
	epoch uint64,
	variant string,
) accesspublisher.LoadedRoutingPublication {
	t.Helper()
	if revision > ^uint64(0)>>1 {
		t.Fatal("test revision exceeds int64")
	}
	now := time.Date(2026, 8, 22, 10, 0, 0, 0, time.UTC)
	namespace := accesscontrol.Namespace{
		ID: accesscontrol.NamespaceID(namespaceID), Name: namespaceID,
		QuotaPartitionID: accesscontrol.QuotaPartitionID(partition), BillingCurrency: "USD",
		Status: accesscontrol.NamespaceStatusActive, Revision: accesscontrol.Revision(revision),
		RuntimeEpoch: epoch, CreatedAt: now, UpdatedAt: now,
	}
	inputPrice := "0.25"
	bundle := routingsnapshot.Bundle{
		NamespaceID: namespaceID, Revision: int64(revision), Currency: "USD",
		Models: []routingsnapshot.Model{{
			ID: "model-chat", Revision: 1,
			CatalogRevision: "sha256:" + strings.Repeat("a", 64), Name: "local/chat-" + variant,
			Capabilities: []string{"chat"},
			Execution: routingsnapshot.ModelExecution{
				MaxRetries: 1, RequestTimeout: "30s", StreamTimeout: "60s",
			},
			Pricing: routingsnapshot.ModelPricing{InputCostPerMillionTokens: &inputPrice},
			Backends: []routingsnapshot.Backend{{
				ID: "backend-chat", ProviderID: "provider-local", WireFormat: "openai.chat.v1",
				Origin: "https://models.example", ProviderModelID: "chat-" + variant,
				Connection: routingsnapshot.BackendConnection{Path: "/v1/chat/completions"}, Weight: "1",
			}},
		}},
		Recipes: []routingsnapshot.Recipe{{
			ID: "recipe-chat", Revision: 1, Name: "Chat",
			Decisions: []routingsnapshot.Decision{{ID: "decision-chat", Name: "Chat", DispatchCardinality: routingsnapshot.DispatchCardinalitySingle}},
			Document: json.RawMessage(`{
  "signals": {},
  "projections": {},
  "decisions": [{"name":"Chat","rules":{}}]
}`),
		}},
		Entrypoints: []routingsnapshot.Entrypoint{{
			ID: "entrypoint-chat", Revision: 1, Name: "Chat", Aliases: []string{"vllm-sr/chat"},
			Rules: []routingsnapshot.EntrypointRule{{
				ID: "rule-chat", Name: "Chat", RecipeID: "recipe-chat", RecipeRevision: 1,
				Assignments: map[string]routingsnapshot.AssignmentSet{
					"decision-chat": {Models: []routingsnapshot.Assignment{{ModelID: "model-chat", ModelRevision: 1, Weight: "1"}}},
				},
			}},
		}},
	}
	publication, err := accesspublisher.Compile(accesspublisher.DesiredState{
		Namespace: namespace, Revision: revision, RevisionTime: now.Add(time.Duration(revision) * time.Millisecond),
		Routing: bundle,
	})
	if err != nil {
		t.Fatalf("compile publication: %v", err)
	}
	identity := accesspublisher.RuntimePublicationIdentity{
		PublicationID: publication.ID, NamespaceID: publication.NamespaceID,
		QuotaPartition: publication.QuotaPartition, DesiredRevision: publication.DesiredRevision,
		RuntimeEpoch: publication.RuntimeEpoch, PublicationDigest: publication.Digest,
		ManifestDigest: publication.Manifest.Digest, RoutingDigest: publication.Routing.Digest,
		State: accesspublisher.PublicationStateValidated,
	}
	return accesspublisher.LoadedRoutingPublication{
		Identity: identity, Manifest: publication.Manifest, Routing: publication.Routing,
		Snapshot: publication.Routing.Snapshot,
	}
}

func managedRouterPublicationState(
	publication accesspublisher.LoadedRoutingPublication,
	state accesspublisher.PublicationRuntimeState,
) accesspublisher.LoadedRoutingPublication {
	publication.Identity.State = state
	return publication
}

func managedRouterPin(publication accesspublisher.LoadedRoutingPublication) ManagedRouterGenerationPin {
	return ManagedRouterGenerationPin{
		NamespaceID: publication.Identity.NamespaceID, QuotaPartition: publication.Identity.QuotaPartition,
		PublicationID: publication.Identity.PublicationID, RuntimeEpoch: publication.Identity.RuntimeEpoch,
		SnapshotRevision: publication.Snapshot.Revision, RoutingDigest: publication.Identity.RoutingDigest,
	}
}

func managedRoutingSnapshotPin(publication accesspublisher.LoadedRoutingPublication) routingcontext.Generation {
	return routingcontext.Generation{
		NamespaceID: publication.Identity.NamespaceID, QuotaPartition: publication.Identity.QuotaPartition,
		PublicationID: publication.Identity.PublicationID, RuntimeEpoch: publication.Identity.RuntimeEpoch,
		SnapshotRevision: publication.Snapshot.Revision,
		RoutingDigest:    publication.Identity.RoutingDigest,
	}
}

func waitManagedRouterRegistry(t *testing.T, description string, condition func() bool) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if condition() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", description)
}

const managedRouterBootstrapYAML = `
version: v0.4
global:
  control_plane:
    mode: managed
    provider_catalog:
      replica_id_env: VLLM_SR_REPLICA_ID
      rollout_groups:
        - {plane: control, id: management}
        - {plane: data, id: router}
      required_rollout_groups:
        - {plane: control, id: management}
        - {plane: data, id: router}
  stores:
    access:
      type: postgres
      postgres:
        dsn_env: VLLM_SR_ACCESS_DATABASE_URL
    access_runtime:
      type: redis
      redis:
        url_file: /run/secrets/access-redis-url
  services:
    agent:
      public_inference_endpoint: http://public-inference.internal/v1/chat/completions
    access:
      enabled: true
      credentials:
        api_key_hmac_keyring_file: /run/secrets/api-key-peppers
        delegation_hmac_keyring_file: /run/secrets/delegation-peppers
      tenant_context:
        signing_key_file: /run/secrets/tenant-context-keys
    backend_credentials:
      provider_kek_keyring_file: /run/secrets/provider-keks
    backend_egress:
      policy_file: /etc/vllm-sr/backend-egress-policy.yaml
    management_api:
      tls:
        certificate_file: /run/secrets/management-cert
        private_key_file: /run/secrets/management-key
      auth:
        mode: router
        token_signing_keyring_file: /run/secrets/management-signing-keys
        service_account_hmac_keyring_file: /run/secrets/management-peppers
        invitation_hmac_keyring_file: /run/secrets/invitation-peppers
        control_plane_hmac_keyring_file: /run/secrets/control-plane-hmac-keys
        response_kek_keyring_file: /run/secrets/response-keks
        bootstrap:
          token_file: /run/secrets/bootstrap-token
          disable_after_first_cluster_admin: true
`

func TestManagedRouterRegistryConstructorRejectsInvalidBootstrap(t *testing.T) {
	if _, err := NewManagedRouterRegistry(ManagedRouterRegistryOptions{}); err == nil {
		t.Fatal("NewManagedRouterRegistry accepted a nil bootstrap")
	}
	standalone := config.DefaultGlobalConfig()
	if _, err := NewManagedRouterRegistry(ManagedRouterRegistryOptions{BootstrapConfig: &standalone}); err == nil {
		t.Fatal("NewManagedRouterRegistry accepted standalone mode")
	}
	base, err := config.ParseYAMLBytes([]byte(managedRouterBootstrapYAML))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewManagedRouterRegistry(ManagedRouterRegistryOptions{BootstrapConfig: base}); err == nil ||
		!strings.Contains(err.Error(), "required") {
		t.Fatalf("missing runtime dependency error = %v", err)
	}
}

func ExampleManagedRouterGenerationPin() {
	pin := ManagedRouterGenerationPin{SnapshotRevision: 12}
	fmt.Println(pin.SnapshotRevision)
	// Output: 12
}
