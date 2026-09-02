package evaluationplane

import (
	"context"
	"errors"
	"testing"
	"time"
)

func subscriberRegistryCounts(service *Service) (int, int, int) {
	registry := service.activity.eventSubscribers
	registry.mu.Lock()
	defer registry.mu.Unlock()
	return registry.subscriberCount, len(registry.byRun), len(registry.byOwner)
}

func newSubscriberPeerService(t *testing.T, first *Service) *Service {
	t.Helper()
	peer, err := NewService(Options{
		DataDir: first.store.Root(), PythonPath: "python3", ConfigPath: first.registrySource.configPath,
		DeploymentsDir: first.registrySource.deploymentsDir, RouterAPIURL: first.registrySource.routerAPIURL, EnvoyURL: first.registrySource.envoyURL,
		CodeRevision: first.codeRevision, MaxConcurrent: first.activity.workerCapacity, Process: &controlledProcess{},
	})
	if err != nil {
		t.Fatalf("open peer evaluation service: %v", err)
	}
	t.Cleanup(func() { _ = peer.Close() })
	return peer
}

func requireSubscriptionClosed(t *testing.T, events <-chan Event, description string) {
	t.Helper()
	select {
	case _, open := <-events:
		if open {
			t.Fatalf("%s delivered an event instead of closing", description)
		}
	case <-time.After(time.Second):
		t.Fatalf("%s remained open", description)
	}
}

func TestSubscribeEnforcesPerRunAndGlobalBounds(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRunAs(t.Context(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	unsubscribers := make([]func(), 0, maxSubscribersPerRun)
	for range maxSubscribersPerRun {
		_, unsubscribe, subscribeErr := service.SubscribeAs(SystemActor(), run.ID)
		if subscribeErr != nil {
			t.Fatalf("Subscribe below per-run limit: %v", subscribeErr)
		}
		unsubscribers = append(unsubscribers, unsubscribe)
	}
	if _, _, err := service.SubscribeAs(SystemActor(), run.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("Subscribe above per-run limit error=%v, want ErrConflict", err)
	}
	for _, unsubscribe := range unsubscribers {
		unsubscribe()
		unsubscribe() // Idempotent cleanup must not underflow the global bound.
	}
	if count, _, _ := subscriberRegistryCounts(service); count != 0 {
		t.Fatalf("subscriber count after cleanup=%d, want 0", count)
	}

	registry := service.activity.eventSubscribers
	registry.mu.Lock()
	registry.subscriberCount = maxSubscribersGlobal
	registry.mu.Unlock()
	if _, _, err := service.SubscribeAs(SystemActor(), run.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("Subscribe above global limit error=%v, want ErrConflict", err)
	}
	registry.mu.Lock()
	registry.subscriberCount = 0
	registry.mu.Unlock()
}

func TestSubscribeEnforcesPrincipalFairnessAcrossRuns(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	ownerA := testLifecycleActor(t, "subscriber-owner-a", false)
	ownerB := testLifecycleActor(t, "subscriber-owner-b", false)
	create := func(owner Actor, name string) Run {
		t.Helper()
		request := validCreateRequest()
		request.ClientRequestID = newTestClientRequestID()
		request.Name = name
		run, err := service.CreateRunAs(t.Context(), owner, request)
		if err != nil {
			t.Fatalf("create %s: %v", name, err)
		}
		return run
	}
	a1 := create(ownerA, "subscriber owner A first run")
	a2 := create(ownerA, "subscriber owner A second run")
	a3 := create(ownerA, "subscriber owner A overflow run")
	b1 := create(ownerB, "subscriber owner B run")
	unsubscribers := make([]func(), 0, maxSubscribersPerPrincipal)
	for _, run := range []Run{a1, a2} {
		for range maxSubscribersPerRun {
			_, unsubscribe, err := service.SubscribeAs(ownerA, run.ID)
			if err != nil {
				t.Fatalf("owner A subscription below principal bound: %v", err)
			}
			unsubscribers = append(unsubscribers, unsubscribe)
		}
	}
	if _, unsubscribe, err := service.SubscribeAs(ownerA, a3.ID); !errors.Is(err, ErrConflict) || unsubscribe != nil {
		t.Fatalf("owner A principal overflow unsubscribe=%v err=%v", unsubscribe != nil, err)
	}
	_, unsubscribeB, err := service.SubscribeAs(ownerB, b1.ID)
	if err != nil {
		t.Fatalf("owner A subscriber saturation starved owner B: %v", err)
	}
	unsubscribeB()
	for _, unsubscribe := range unsubscribers {
		unsubscribe()
	}
	registry := service.activity.eventSubscribers
	registry.mu.Lock()
	defer registry.mu.Unlock()
	if registry.subscriberCount != 0 || len(registry.byPrincipal) != 0 {
		t.Fatalf("principal subscriber accounting leaked count=%d principals=%d", registry.subscriberCount, len(registry.byPrincipal))
	}
}

func TestSubscribersAreSharedAcrossServicesForDeleteAndBroadcast(t *testing.T) {
	owner, _ := newTestService(t, &controlledProcess{}, 1)
	t.Cleanup(func() { _ = owner.Close() })
	run, createErr := owner.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("create shared subscriber run: %v", createErr)
	}
	peer := newSubscriberPeerService(t, owner)
	t.Cleanup(func() { _ = peer.Close() })

	peerEvents, _, peerSubscribeErr := peer.SubscribeAs(SystemActor(), run.ID)
	if peerSubscribeErr != nil {
		t.Fatalf("subscribe through peer Service: %v", peerSubscribeErr)
	}
	run = stageRunningTestRun(t, owner, run)
	if err := owner.recordWorkerEvent(run.ID, WorkerEvent{Type: "progress", Message: "cross-service progress"}); err != nil {
		t.Fatalf("record owner worker event: %v", err)
	}
	select {
	case event, open := <-peerEvents:
		if !open || event.RunID != run.ID || event.Message != publicWorkerEventMessage("progress") {
			t.Fatalf("peer live event=%+v open=%v", event, open)
		}
	case <-time.After(time.Second):
		t.Fatal("owner worker event did not reach peer subscriber")
	}

	completedAt := time.Now().UTC().Truncate(time.Microsecond)
	run.Status, run.CompletedAt, run.Progress.Message = StatusCancelled, &completedAt, "Run cancelled"
	if err := owner.store.updateRunFixture(run); err != nil {
		t.Fatalf("make cross-service deletion run terminal: %v", err)
	}
	ownerEvents, _, ownerSubscribeErr := owner.SubscribeAs(SystemActor(), run.ID)
	if ownerSubscribeErr != nil {
		t.Fatalf("subscribe through owner Service: %v", ownerSubscribeErr)
	}
	if err := peer.DeleteRunAs(SystemActor(), run.ID); err != nil {
		t.Fatalf("delete through peer Service: %v", err)
	}
	requireSubscriptionClosed(t, peerEvents, "peer subscription after peer deletion")
	requireSubscriptionClosed(t, ownerEvents, "owner subscription after peer deletion")
	if count, runs, owners := subscriberRegistryCounts(owner); count != 0 || runs != 0 || owners != 0 {
		t.Fatalf("shared registry after deletion count=%d runs=%d owners=%d", count, runs, owners)
	}
}

func TestSubscriberValidationAndRegistrationAreAtomicWithDeleteClose(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	t.Cleanup(func() { _ = service.Close() })
	registry := service.activity.eventSubscribers
	runID := newTestClientRequestID()
	validated := make(chan struct{})
	releaseValidation := make(chan struct{})
	type subscribeResult struct {
		events <-chan Event
		err    error
	}
	result := make(chan subscribeResult, 1)
	go func() {
		events, _, err := registry.subscribe(service, runID, SystemActor().principalDigest, func() error {
			close(validated)
			<-releaseValidation
			return nil
		})
		result <- subscribeResult{events: events, err: err}
	}()
	<-validated
	attempted := make(chan struct{})
	closed := make(chan struct{})
	go func() {
		close(attempted)
		registry.closeRun(runID)
		close(closed)
	}()
	<-attempted
	select {
	case <-closed:
		t.Fatal("delete close crossed the validation-to-registration critical section")
	default:
	}
	close(releaseValidation)
	subscribed := <-result
	if subscribed.err != nil {
		t.Fatalf("register validated subscriber: %v", subscribed.err)
	}
	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("delete close did not resume after subscriber registration")
	}
	requireSubscriptionClosed(t, subscribed.events, "subscriber registered during delete race")
	if count, runs, owners := subscriberRegistryCounts(service); count != 0 || runs != 0 || owners != 0 {
		t.Fatalf("registry after validation race count=%d runs=%d owners=%d", count, runs, owners)
	}
}

func TestServiceCloseOnlyClosesItsSubscribersAndLastCloseCleansRegistry(t *testing.T) {
	first, _ := newTestService(t, &controlledProcess{}, 1)
	run, err := first.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("create close cleanup run: %v", err)
	}
	second := newSubscriberPeerService(t, first)
	firstEvents, _, err := first.SubscribeAs(SystemActor(), run.ID)
	if err != nil {
		t.Fatalf("subscribe through first Service: %v", err)
	}
	secondEvents, _, err := second.SubscribeAs(SystemActor(), run.ID)
	if err != nil {
		t.Fatalf("subscribe through second Service: %v", err)
	}
	if err := first.Close(); err != nil {
		t.Fatalf("close first Service: %v", err)
	}
	requireSubscriptionClosed(t, firstEvents, "closed Service subscription")
	select {
	case _, open := <-secondEvents:
		if !open {
			t.Fatal("closing one Service closed a peer Service subscription")
		}
	default:
	}
	if count, runs, owners := subscriberRegistryCounts(second); count != 1 || runs != 1 || owners != 1 {
		t.Fatalf("registry before last close count=%d runs=%d owners=%d", count, runs, owners)
	}
	if err := second.Close(); err != nil {
		t.Fatalf("close last Service: %v", err)
	}
	requireSubscriptionClosed(t, secondEvents, "last Service subscription")
	if count, runs, owners := subscriberRegistryCounts(second); count != 0 || runs != 0 || owners != 0 {
		t.Fatalf("registry after last close count=%d runs=%d owners=%d", count, runs, owners)
	}
}

func TestSubscriberRegistriesAreIsolatedByRoot(t *testing.T) {
	first, _ := newTestService(t, &controlledProcess{}, 1)
	second, _ := newTestService(t, &controlledProcess{}, 1)
	t.Cleanup(func() { _ = first.Close() })
	t.Cleanup(func() { _ = second.Close() })
	request := validCreateRequest()
	if _, err := first.CreateRunAs(context.Background(), SystemActor(), request); err != nil {
		t.Fatalf("create first-root run: %v", err)
	}
	if _, err := second.CreateRunAs(context.Background(), SystemActor(), request); err != nil {
		t.Fatalf("create second-root run: %v", err)
	}
	firstEvents, _, err := first.SubscribeAs(SystemActor(), request.ClientRequestID)
	if err != nil {
		t.Fatalf("subscribe first root: %v", err)
	}
	secondEvents, _, err := second.SubscribeAs(SystemActor(), request.ClientRequestID)
	if err != nil {
		t.Fatalf("subscribe second root: %v", err)
	}
	first.activity.eventSubscribers.broadcast(Event{RunID: request.ClientRequestID, Message: "first root only"})
	select {
	case event := <-firstEvents:
		if event.Message != "first root only" {
			t.Fatalf("first-root event=%+v", event)
		}
	case <-time.After(time.Second):
		t.Fatal("first-root subscriber missed its event")
	}
	select {
	case event := <-secondEvents:
		t.Fatalf("first-root event leaked to second root: %+v", event)
	default:
	}
}
