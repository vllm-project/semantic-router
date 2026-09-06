package evaluationplane

import (
	"fmt"
	"sync"
)

const eventSubscriberBufferSize = 256

// eventSubscriberRegistry is process-local delivery state shared by every
// Service opened on one durable evaluation root. Durable event replay remains
// owned by Store; this registry only coordinates live delivery and channel
// lifecycle across those Service instances.
//
// Lock order for registration is lifecycle.mu -> Service.mu ->
// eventSubscriberRegistry.mu. Registry methods never acquire a Service or
// Store lock. subscribe keeps the registry lock
// across validation and registration so a concurrent successful deletion,
// which closes the run through the same registry, cannot leave an orphaned
// subscriber between GetRun and insertion.
type eventSubscriberRegistry struct {
	mu              sync.Mutex
	byRun           map[string]map[*eventSubscription]struct{}
	byOwner         map[*Service]map[*eventSubscription]struct{}
	byPrincipal     map[string]map[*eventSubscription]struct{}
	subscriberCount int
}

type eventSubscription struct {
	owner           *Service
	runID           string
	principalDigest string
	events          chan Event
}

func newEventSubscriberRegistry() *eventSubscriberRegistry {
	return &eventSubscriberRegistry{
		byRun:       make(map[string]map[*eventSubscription]struct{}),
		byOwner:     make(map[*Service]map[*eventSubscription]struct{}),
		byPrincipal: make(map[string]map[*eventSubscription]struct{}),
	}
}

func (registry *eventSubscriberRegistry) subscribe(
	owner *Service,
	runID string,
	principalDigest string,
	validate func() error,
) (<-chan Event, func(), error) {
	registry.mu.Lock()
	defer registry.mu.Unlock()
	if err := validate(); err != nil {
		return nil, nil, err
	}
	if len(registry.byRun[runID]) >= maxSubscribersPerRun ||
		len(registry.byPrincipal[principalDigest]) >= maxSubscribersPerPrincipal ||
		registry.subscriberCount >= maxSubscribersGlobal {
		return nil, nil, fmt.Errorf("%w: evaluation event subscriber capacity is exhausted", ErrConflict)
	}

	subscription := &eventSubscription{
		owner:           owner,
		runID:           runID,
		principalDigest: principalDigest,
		events:          make(chan Event, eventSubscriberBufferSize),
	}
	if registry.byRun[runID] == nil {
		registry.byRun[runID] = make(map[*eventSubscription]struct{})
	}
	registry.byRun[runID][subscription] = struct{}{}
	if registry.byOwner[owner] == nil {
		registry.byOwner[owner] = make(map[*eventSubscription]struct{})
	}
	registry.byOwner[owner][subscription] = struct{}{}
	if registry.byPrincipal[principalDigest] == nil {
		registry.byPrincipal[principalDigest] = make(map[*eventSubscription]struct{})
	}
	registry.byPrincipal[principalDigest][subscription] = struct{}{}
	registry.subscriberCount++

	unsubscribe := func() {
		registry.mu.Lock()
		defer registry.mu.Unlock()
		registry.removeLocked(subscription, false)
	}
	return subscription.events, unsubscribe, nil
}

func (registry *eventSubscriberRegistry) broadcast(event Event) {
	registry.mu.Lock()
	defer registry.mu.Unlock()
	for subscription := range registry.byRun[event.RunID] {
		select {
		case subscription.events <- event:
		default:
			registry.removeLocked(subscription, true)
		}
	}
}

func (registry *eventSubscriberRegistry) closeRun(runID string) {
	registry.mu.Lock()
	defer registry.mu.Unlock()
	for subscription := range registry.byRun[runID] {
		registry.removeLocked(subscription, true)
	}
}

func (registry *eventSubscriberRegistry) closeOwner(owner *Service) {
	registry.mu.Lock()
	defer registry.mu.Unlock()
	for subscription := range registry.byOwner[owner] {
		registry.removeLocked(subscription, true)
	}
}

func (registry *eventSubscriberRegistry) removeLocked(subscription *eventSubscription, closeEvents bool) {
	subscriptions := registry.byRun[subscription.runID]
	if _, exists := subscriptions[subscription]; !exists {
		return
	}
	delete(subscriptions, subscription)
	if len(subscriptions) == 0 {
		delete(registry.byRun, subscription.runID)
	}
	ownerSubscriptions := registry.byOwner[subscription.owner]
	delete(ownerSubscriptions, subscription)
	if len(ownerSubscriptions) == 0 {
		delete(registry.byOwner, subscription.owner)
	}
	principalSubscriptions := registry.byPrincipal[subscription.principalDigest]
	delete(principalSubscriptions, subscription)
	if len(principalSubscriptions) == 0 {
		delete(registry.byPrincipal, subscription.principalDigest)
	}
	registry.subscriberCount--
	if closeEvents {
		close(subscription.events)
	}
}
