package publicationreplica

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
)

func TestManagerCoordinatesFirstNonrestrictiveAndRestrictivePublications(t *testing.T) {
	reference := accesspublisher.NamespacePublication{NamespaceID: "namespace-a", QuotaPartition: "partition-a"}
	first := runtimeIdentity(reference, 1, false)
	second := runtimeIdentity(reference, 2, false)
	third := runtimeIdentity(reference, 3, true)
	store := newFakeStore(reference)
	store.setCandidate(first)
	snapshots := newFakeSnapshots()
	manager, cancel := runTestManager(t, store, snapshots, "replica-a")
	defer cancel()

	waitFor(t, "first publication routing acknowledgement", func() bool { return store.routingAcked(first.PublicationID) })
	if store.barrierAcked(first.PublicationID) {
		t.Fatal("nonrestrictive first publication acknowledged barriers")
	}
	if manager.Ready() == nil {
		t.Fatal("replica became ready before the first publication activated")
	}
	if snapshots.wasActivated(first.PublicationID) {
		t.Fatal("candidate was activated before its gate changed")
	}
	store.activate(first)
	waitFor(t, "first active publication", func() bool {
		identity, ready := manager.Current(reference.NamespaceID)
		return ready && identity.SameGeneration(first) && identity.State == accesspublisher.PublicationStateActive
	})

	store.setCandidate(second)
	waitFor(t, "nonrestrictive publication acknowledgement", func() bool { return store.routingAcked(second.PublicationID) })
	if snapshots.wasActivated(second.PublicationID) {
		t.Fatal("nonrestrictive candidate was activated before its gate changed")
	}
	store.activate(second)
	waitFor(t, "second active publication", func() bool {
		identity, ready := manager.Current(reference.NamespaceID)
		return ready && identity.SameGeneration(second)
	})

	store.setCandidate(third)
	waitFor(t, "restrictive publication acknowledgements", func() bool {
		return store.barrierAcked(third.PublicationID) && store.routingAcked(third.PublicationID)
	})
	if snapshots.wasActivated(third.PublicationID) {
		t.Fatal("restrictive candidate was activated before its gate changed")
	}
	store.activate(third)
	waitFor(t, "third active publication", func() bool {
		identity, ready := manager.Current(reference.NamespaceID)
		return ready && identity.SameGeneration(third)
	})
}

func TestManagerFailsClosedUntilDiscoveredNamespaceHasAnActivePublication(t *testing.T) {
	reference := accesspublisher.NamespacePublication{NamespaceID: "namespace-unpublished", QuotaPartition: "partition-unpublished"}
	store := newFakeStore(reference)
	manager, cancel := runTestManager(t, store, newFakeSnapshots(), "replica-unpublished")
	defer cancel()

	waitFor(t, "unpublished namespace fail closed", func() bool {
		status := manager.Status()
		return !status.Ready && len(status.Namespaces) == 1 &&
			status.Namespaces[0].Reason == "publication_unavailable"
	})
	if _, ready := manager.Current(reference.NamespaceID); ready {
		t.Fatal("Current returned a generation before the namespace had an active publication")
	}
}

func TestManagerFailsClosedOnCorruptCandidate(t *testing.T) {
	reference := accesspublisher.NamespacePublication{NamespaceID: "namespace-corrupt", QuotaPartition: "partition-corrupt"}
	active := runtimeIdentity(reference, 1, false)
	corrupt := runtimeIdentity(reference, 2, false)
	store := newFakeStore(reference)
	store.activate(active)
	snapshots := newFakeSnapshots()
	manager, cancel := runTestManager(t, store, snapshots, "replica-corrupt")
	defer cancel()
	waitFor(t, "initial readiness", func() bool { return manager.Ready() == nil })

	store.mu.Lock()
	store.loadErrors[corrupt.PublicationID] = fmt.Errorf("%w: test corruption", accesspublisher.ErrStagedCorrupt)
	store.mu.Unlock()
	store.setCandidate(corrupt)
	waitFor(t, "corrupt candidate fail closed", func() bool {
		status := manager.Status()
		return !status.Ready && len(status.Namespaces) == 1 && status.Namespaces[0].Reason == "publication_corrupt"
	})
	if _, ready := manager.Current(reference.NamespaceID); ready {
		t.Fatal("Current returned a generation while a corrupt candidate was observed")
	}
	if store.routingAcked(corrupt.PublicationID) {
		t.Fatal("corrupt candidate was acknowledged")
	}
}

func TestManagerRetriesActivationRaceWithoutSwitchingStaleGeneration(t *testing.T) {
	reference := accesspublisher.NamespacePublication{NamespaceID: "namespace-race", QuotaPartition: "partition-race"}
	first := runtimeIdentity(reference, 1, false)
	second := runtimeIdentity(reference, 2, false)
	store := newFakeStore(reference)
	store.activate(first)
	snapshots := newFakeSnapshots()
	var once sync.Once
	snapshots.warmHook = func(publication accesspublisher.LoadedRoutingPublication) {
		if publication.Identity.SameGeneration(first) {
			once.Do(func() { store.activate(second) })
		}
	}
	manager, cancel := runTestManager(t, store, snapshots, "replica-race")
	defer cancel()
	waitFor(t, "new active publication after race", func() bool {
		identity, ready := manager.Current(reference.NamespaceID)
		return ready && identity.SameGeneration(second)
	})
	if snapshots.wasActivated(first.PublicationID) {
		t.Fatal("stale generation was switched after the active gate raced ahead")
	}
}

func TestManagerPollingRecoversWhenPublicationNotificationsAreLost(t *testing.T) {
	reference := accesspublisher.NamespacePublication{NamespaceID: "namespace-lost-notify", QuotaPartition: "partition-lost-notify"}
	first := runtimeIdentity(reference, 1, false)
	second := runtimeIdentity(reference, 2, false)
	store := newFakeStore(reference)
	store.activate(first)
	snapshots := newFakeSnapshots()
	manager, cancel := runTestManager(t, store, snapshots, "replica-lost-notify")
	defer cancel()
	waitFor(t, "initial publication without notification", func() bool { return manager.Ready() == nil })

	// Deliberately mutate the durable store without sending a wake-up. Periodic
	// polling, rather than the lossy notification channel, remains authoritative.
	store.setCandidate(second)
	waitFor(t, "candidate acknowledgement after lost notification", func() bool {
		return store.routingAcked(second.PublicationID)
	})
	store.activate(second)
	waitFor(t, "activation after lost notification", func() bool {
		identity, ready := manager.Current(reference.NamespaceID)
		return ready && identity.SameGeneration(second)
	})
}

func TestManagerFailsClosedWhenLeaseRenewalIsLost(t *testing.T) {
	reference := accesspublisher.NamespacePublication{NamespaceID: "namespace-lease", QuotaPartition: "partition-lease"}
	active := runtimeIdentity(reference, 1, false)
	store := newFakeStore(reference)
	store.activate(active)
	snapshots := newFakeSnapshots()
	manager, cancel := runTestManager(t, store, snapshots, "replica-lease")
	defer cancel()
	waitFor(t, "initial lease", func() bool { return manager.Ready() == nil })

	store.mu.Lock()
	store.registerErr = errors.New("lease backend unavailable")
	store.mu.Unlock()
	waitFor(t, "lease failure", func() bool {
		status := manager.Status()
		return !status.Ready && len(status.Namespaces) == 1 && status.Namespaces[0].Reason == "lease_unavailable"
	})
	if _, ready := manager.Current(reference.NamespaceID); ready {
		t.Fatal("Current returned a generation after lease loss")
	}
}

func TestManagerFailsClosedWhenFleetLeaseRenewalIsLost(t *testing.T) {
	reference := accesspublisher.NamespacePublication{NamespaceID: "namespace-fleet-lease", QuotaPartition: "partition-fleet-lease"}
	active := runtimeIdentity(reference, 1, false)
	store := newFakeStore(reference)
	store.activate(active)
	snapshots := newFakeSnapshots()
	manager, cancel := runTestManager(t, store, snapshots, "replica-fleet-lease")
	defer cancel()
	waitFor(t, "initial fleet lease", func() bool { return manager.Ready() == nil })

	store.mu.Lock()
	store.fleetRegisterErr = errors.New("fleet lease backend unavailable")
	store.mu.Unlock()
	waitFor(t, "fleet lease failure", func() bool {
		status := manager.Status()
		return !status.Ready && status.Reason == "fleet_lease_unavailable"
	})
	if _, ready := manager.Current(reference.NamespaceID); ready {
		t.Fatal("Current returned a generation after fleet lease loss")
	}
}

func TestManagerCanEstablishFleetLeaseBeforeRun(t *testing.T) {
	reference := accesspublisher.NamespacePublication{NamespaceID: "namespace-startup", QuotaPartition: "partition-startup"}
	manager, err := New(Options{
		Store: newFakeStore(reference), Snapshots: newFakeSnapshots(), ReplicaID: "replica-startup",
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := manager.EnsureFleetLease(context.Background()); err != nil {
		t.Fatalf("EnsureFleetLease() = %v", err)
	}
	if manager.Status().Reason != "not_running" {
		t.Fatalf("pre-run status = %+v", manager.Status())
	}
}

func TestManagerDynamicallyAddsAndRemovesNamespaces(t *testing.T) {
	firstReference := accesspublisher.NamespacePublication{NamespaceID: "namespace-dynamic-a", QuotaPartition: "partition-dynamic-a"}
	secondReference := accesspublisher.NamespacePublication{NamespaceID: "namespace-dynamic-b", QuotaPartition: "partition-dynamic-b"}
	first := runtimeIdentity(firstReference, 1, false)
	second := runtimeIdentity(secondReference, 1, false)
	store := newFakeStore(firstReference)
	store.activate(first)
	snapshots := newFakeSnapshots()
	manager, cancel := runTestManager(t, store, snapshots, "replica-dynamic")
	defer cancel()
	waitFor(t, "first dynamic namespace", func() bool { return manager.Ready() == nil })

	store.addNamespace(secondReference, second)
	waitFor(t, "second dynamic namespace", func() bool {
		identity, ready := manager.Current(secondReference.NamespaceID)
		return ready && identity.SameGeneration(second)
	})
	store.removeNamespace(firstReference.NamespaceID)
	waitFor(t, "removed namespace lifecycle", func() bool {
		_, exists := manager.Current(firstReference.NamespaceID)
		return !exists && snapshots.wasRemoved(firstReference.NamespaceID)
	})
	if manager.Ready() != nil {
		t.Fatalf("remaining namespace did not stay ready: %+v", manager.Status())
	}
}

func runTestManager(
	t *testing.T,
	store *fakeStore,
	snapshots *fakeSnapshots,
	replicaID string,
) (*Manager, context.CancelFunc) {
	t.Helper()
	manager, err := New(Options{
		Store: store, Snapshots: snapshots, ReplicaID: replicaID,
		DiscoveryInterval: 10 * time.Millisecond, PollInterval: 10 * time.Millisecond, RenewInterval: 20 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- manager.Run(ctx) }()
	t.Cleanup(func() {
		cancel()
		select {
		case err := <-done:
			if err != nil {
				t.Errorf("Manager.Run() = %v", err)
			}
		case <-time.After(time.Second):
			t.Error("Manager.Run() did not stop")
		}
	})
	return manager, cancel
}

func waitFor(t *testing.T, description string, condition func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if condition() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", description)
}

func runtimeIdentity(
	reference accesspublisher.NamespacePublication,
	revision uint64,
	restrictive bool,
) accesspublisher.RuntimePublicationIdentity {
	character := byte('a' + revision%20)
	digest := string(make([]byte, 64))
	bytes := []byte(digest)
	for index := range bytes {
		bytes[index] = character
	}
	digest = string(bytes)
	return accesspublisher.RuntimePublicationIdentity{
		PublicationID: fmt.Sprintf("pub-%s-%d", reference.NamespaceID, revision), NamespaceID: reference.NamespaceID,
		QuotaPartition: reference.QuotaPartition, DesiredRevision: revision, RuntimeEpoch: 9,
		PublicationDigest: digest, ManifestDigest: digest, RoutingDigest: digest,
		State: accesspublisher.PublicationStateValidated, Restrictive: restrictive,
	}
}

type fakeStore struct {
	mu sync.Mutex

	references       []accesspublisher.NamespacePublication
	heads            map[string]accesspublisher.PublicationHeads
	publications     map[string]accesspublisher.LoadedRoutingPublication
	loadErrors       map[string]error
	routingAcks      map[string]bool
	barrierAcks      map[string]bool
	registerErr      error
	fleetRegisterErr error
	lease            time.Duration
	notifications    chan struct{}
}

func (s *fakeStore) PublicationNotifications(context.Context) <-chan struct{} {
	return s.notifications
}

func (s *fakeStore) RegisterFleetReplica(_ context.Context, _ string) (time.Time, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.fleetRegisterErr != nil {
		return time.Time{}, s.fleetRegisterErr
	}
	return time.Now().Add(s.lease), nil
}

func newFakeStore(reference accesspublisher.NamespacePublication) *fakeStore {
	return &fakeStore{
		references: []accesspublisher.NamespacePublication{reference},
		heads: map[string]accesspublisher.PublicationHeads{
			reference.NamespaceID: {Namespace: reference},
		},
		publications: make(map[string]accesspublisher.LoadedRoutingPublication),
		loadErrors:   make(map[string]error), routingAcks: make(map[string]bool), barrierAcks: make(map[string]bool),
		lease: 200 * time.Millisecond, notifications: make(chan struct{}, 1),
	}
}

func (s *fakeStore) ListPublicationNamespaces(context.Context) ([]accesspublisher.NamespacePublication, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]accesspublisher.NamespacePublication(nil), s.references...), nil
}

func (s *fakeStore) ReadPublicationHeads(
	_ context.Context,
	reference accesspublisher.NamespacePublication,
) (accesspublisher.PublicationHeads, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return cloneHeads(s.heads[reference.NamespaceID]), nil
}

func (s *fakeStore) LoadRoutingPublication(
	_ context.Context,
	identity accesspublisher.RuntimePublicationIdentity,
) (accesspublisher.LoadedRoutingPublication, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.loadErrors[identity.PublicationID]; err != nil {
		return accesspublisher.LoadedRoutingPublication{}, err
	}
	publication := s.publications[identity.PublicationID]
	publication.Identity = identity
	return publication, nil
}

func (s *fakeStore) RegisterReplica(
	_ context.Context,
	namespaceID, _ string,
	registration accesspublisher.ReplicaRegistration,
) (time.Time, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.registerErr != nil {
		return time.Time{}, s.registerErr
	}
	heads := s.heads[namespaceID]
	activeID := ""
	epoch := registration.RuntimeEpoch
	if heads.Active != nil {
		activeID = heads.Active.PublicationID
		epoch = heads.Active.RuntimeEpoch
	} else if heads.Candidate != nil {
		epoch = heads.Candidate.RuntimeEpoch
	}
	if registration.RuntimeEpoch != epoch || registration.AccessPublication != activeID || registration.RoutingPublication != activeID {
		return time.Time{}, accesspublisher.ErrConflict
	}
	return time.Now().Add(s.lease), nil
}

func (s *fakeStore) AcknowledgeBarriers(
	_ context.Context, _, _, _, publicationID, _ string,
) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.barrierAcks[publicationID] = true
	return nil
}

func (s *fakeStore) AcknowledgeRouting(
	_ context.Context, _, _, _, publicationID, _ string,
) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.routingAcks[publicationID] = true
	return nil
}

func (s *fakeStore) setCandidate(identity accesspublisher.RuntimePublicationIdentity) {
	s.mu.Lock()
	defer s.mu.Unlock()
	heads := s.heads[identity.NamespaceID]
	copy := identity
	heads.Candidate = &copy
	s.heads[identity.NamespaceID] = heads
	s.publications[identity.PublicationID] = accesspublisher.LoadedRoutingPublication{Identity: identity}
}

func (s *fakeStore) activate(identity accesspublisher.RuntimePublicationIdentity) {
	s.mu.Lock()
	defer s.mu.Unlock()
	heads := s.heads[identity.NamespaceID]
	copy := identity
	copy.State = accesspublisher.PublicationStateActive
	heads.Active = &copy
	heads.Candidate = nil
	s.heads[identity.NamespaceID] = heads
	s.publications[identity.PublicationID] = accesspublisher.LoadedRoutingPublication{Identity: copy}
}

func (s *fakeStore) addNamespace(
	reference accesspublisher.NamespacePublication,
	identity accesspublisher.RuntimePublicationIdentity,
) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.references = append(s.references, reference)
	copy := identity
	copy.State = accesspublisher.PublicationStateActive
	s.heads[reference.NamespaceID] = accesspublisher.PublicationHeads{Namespace: reference, Active: &copy}
	s.publications[identity.PublicationID] = accesspublisher.LoadedRoutingPublication{Identity: copy}
}

func (s *fakeStore) removeNamespace(namespaceID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	kept := s.references[:0]
	for _, reference := range s.references {
		if reference.NamespaceID != namespaceID {
			kept = append(kept, reference)
		}
	}
	s.references = kept
	delete(s.heads, namespaceID)
}

func (s *fakeStore) routingAcked(publicationID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.routingAcks[publicationID]
}

func (s *fakeStore) barrierAcked(publicationID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.barrierAcks[publicationID]
}

func cloneHeads(input accesspublisher.PublicationHeads) accesspublisher.PublicationHeads {
	result := input
	if input.Active != nil {
		copy := *input.Active
		result.Active = &copy
	}
	if input.Candidate != nil {
		copy := *input.Candidate
		result.Candidate = &copy
	}
	return result
}

type fakeSnapshots struct {
	mu        sync.Mutex
	warmed    []string
	activated []string
	removed   []string
	warmHook  func(accesspublisher.LoadedRoutingPublication)
}

func newFakeSnapshots() *fakeSnapshots { return &fakeSnapshots{} }

func (s *fakeSnapshots) Warm(_ context.Context, publication accesspublisher.LoadedRoutingPublication) error {
	s.mu.Lock()
	s.warmed = append(s.warmed, publication.Identity.PublicationID)
	hook := s.warmHook
	s.mu.Unlock()
	if hook != nil {
		hook(publication)
	}
	return nil
}

func (s *fakeSnapshots) Activate(_ context.Context, publication accesspublisher.LoadedRoutingPublication) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.activated = append(s.activated, publication.Identity.PublicationID)
	return nil
}

func (s *fakeSnapshots) Remove(_ context.Context, reference accesspublisher.NamespacePublication) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.removed = append(s.removed, reference.NamespaceID)
	return nil
}

func (s *fakeSnapshots) wasActivated(publicationID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, activated := range s.activated {
		if activated == publicationID {
			return true
		}
	}
	return false
}

func (s *fakeSnapshots) wasRemoved(namespaceID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, removed := range s.removed {
		if removed == namespaceID {
			return true
		}
	}
	return false
}
