package evaluationplane

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// evaluationRootCoordinator is the only process-local coordination object for
// one canonical evaluation data root. Lifecycle transactions, the run index,
// live event delivery, worker ownership, and the process ownership lock all
// share this identity so opening another Store or Service cannot create a
// second consistency domain for the same durable facts.
type evaluationRootCoordinator struct {
	root             string
	runIndex         *runMetadataIndex
	eventSubscribers *eventSubscriberRegistry
	ownershipLock    *evaluationStoreOwnershipLock
	serviceRefs      int
	openMu           sync.Mutex
	openCond         *sync.Cond
	startupOwner     *evaluationStoreOwnership
	ready            bool
	capacityMu       sync.Mutex
	workerSlots      chan struct{}
	workerCapacity   int
	capacityLoaded   bool
	evidenceReads    chan struct{}
	releaseBlocked   error

	mu                          sync.Mutex
	evidenceMu                  sync.RWMutex
	runNamespaceMu              sync.Mutex
	pendingRunPublications      map[string]pendingNamespacePublication
	campaignNamespaceMu         sync.Mutex
	pendingCampaignPublications map[string]pendingNamespacePublication
	lifecycleResourceMu         sync.Mutex
	pendingLifecycle            map[lifecycleResourceRef]pendingLifecycleMutation
	pendingCollection           *pendingLifecycleCollectionProjection
	activityMu                  sync.Mutex
	activeRuns                  map[string]context.CancelFunc
	controlledPairLaunchMu      sync.Mutex
	controlledPairLaunches      map[string]chan struct{}
	policy                      lifecycleStorePolicy
	policyLoaded                bool
	loaded                      bool
	sequence                    uint64
	activeCount                 uint64
	headDigest                  string
	checkpointDigest            string
	checkpointSequence          uint64
	checkpointSegmentStart      uint64
	checkpointSegmentRoot       string
	checkpointCleanup           bool
	checkpointDurabilityPending bool
	pendingLifecycleBindings    map[lifecycleResourceRef]lifecycleAuditRecord
	checkpointBindings          map[lifecycleResourceRef]lifecycleAuditRecord
	bytes                       int64
	records                     map[string]lifecycleAuditRecord
	creationBindings            map[string]lifecycleAuditRecord
	notFoundDenials             map[string]time.Time
}

// pendingNamespacePublication binds a visible-but-not-yet-durable namespace
// entry to the principal and exact immutable object that published it. A
// different idempotent-looking request, including an administrator request,
// must never be able to fsync and adopt another actor's undecided publication.
type pendingNamespacePublication struct {
	actorDigest    string
	identityDigest string
}

type evaluationStoreOwnership struct {
	coordinator *evaluationRootCoordinator
}

var evaluationRootCoordinators = struct {
	sync.Mutex
	byRoot map[string]*evaluationRootCoordinator
}{byRoot: make(map[string]*evaluationRootCoordinator)}

func newEvaluationRootCoordinator(root string) *evaluationRootCoordinator {
	coordinator := &evaluationRootCoordinator{
		root:                        root,
		runIndex:                    newRunMetadataIndex(),
		eventSubscribers:            newEventSubscriberRegistry(),
		evidenceReads:               make(chan struct{}, maxConcurrentEvidenceReads),
		pendingRunPublications:      make(map[string]pendingNamespacePublication),
		pendingCampaignPublications: make(map[string]pendingNamespacePublication),
		pendingLifecycle:            make(map[lifecycleResourceRef]pendingLifecycleMutation),
		activeRuns:                  make(map[string]context.CancelFunc),
		controlledPairLaunches:      make(map[string]chan struct{}),
		pendingLifecycleBindings:    make(map[lifecycleResourceRef]lifecycleAuditRecord),
		records:                     make(map[string]lifecycleAuditRecord),
		creationBindings:            make(map[string]lifecycleAuditRecord),
		notFoundDenials:             make(map[string]time.Time),
	}
	coordinator.openCond = sync.NewCond(&coordinator.openMu)
	return coordinator
}

func acquireEvaluationStoreOwnership(root string) (*evaluationStoreOwnership, error) {
	evaluationRootCoordinators.Lock()
	defer evaluationRootCoordinators.Unlock()
	coordinator := evaluationRootCoordinators.byRoot[root]
	created := false
	if coordinator == nil {
		coordinator = newEvaluationRootCoordinator(root)
		evaluationRootCoordinators.byRoot[root] = coordinator
		created = true
	}
	if coordinator.releaseBlocked != nil {
		return nil, fmt.Errorf(
			"%w: evaluation root release is blocked by live process state: %w",
			ErrConflict, coordinator.releaseBlocked,
		)
	}
	startupAuthority := coordinator.serviceRefs == 0
	if startupAuthority {
		ownershipLock, err := openEvaluationStoreOwnershipLock(root)
		if err != nil {
			if created {
				delete(evaluationRootCoordinators.byRoot, root)
			}
			return nil, err
		}
		coordinator.ownershipLock = ownershipLock
	}
	coordinator.serviceRefs++
	ownership := &evaluationStoreOwnership{coordinator: coordinator}
	if startupAuthority {
		coordinator.openMu.Lock()
		coordinator.startupOwner = ownership
		coordinator.openMu.Unlock()
	}
	return ownership, nil
}

func (ownership *evaluationStoreOwnership) initialize(open func(startupAuthority bool) error) error {
	if ownership == nil || ownership.coordinator == nil {
		return fmt.Errorf("evaluation store ownership is invalid")
	}
	coordinator := ownership.coordinator
	coordinator.openMu.Lock()
	for !coordinator.ready && coordinator.startupOwner != nil && coordinator.startupOwner != ownership {
		coordinator.openCond.Wait()
	}
	if coordinator.ready {
		coordinator.openMu.Unlock()
		return open(false)
	}
	if coordinator.startupOwner == nil {
		coordinator.startupOwner = ownership
	}
	coordinator.openMu.Unlock()

	err := open(true)
	coordinator.openMu.Lock()
	if coordinator.startupOwner == ownership {
		coordinator.startupOwner = nil
		coordinator.ready = err == nil
		coordinator.openCond.Broadcast()
	}
	coordinator.openMu.Unlock()
	return err
}

func (ownership *evaluationStoreOwnership) release() error {
	if ownership == nil || ownership.coordinator == nil {
		return nil
	}
	evaluationRootCoordinators.Lock()
	defer evaluationRootCoordinators.Unlock()
	coordinator := ownership.coordinator
	if evaluationRootCoordinators.byRoot[coordinator.root] != coordinator || coordinator.serviceRefs <= 0 {
		return fmt.Errorf("evaluation store ownership is invalid")
	}
	if coordinator.serviceRefs == 1 {
		if err := coordinator.requireReleaseQuiescence(); err != nil {
			coordinator.releaseBlocked = err
			return fmt.Errorf("%w: evaluation root still has live process state: %w", ErrConflict, err)
		}
		coordinator.releaseBlocked = nil
	}
	coordinator.serviceRefs--
	coordinator.openMu.Lock()
	if coordinator.startupOwner == ownership {
		coordinator.startupOwner = nil
		coordinator.openCond.Broadcast()
	}
	coordinator.openMu.Unlock()
	if coordinator.serviceRefs != 0 {
		ownership.coordinator = nil
		return nil
	}
	closeErr := coordinator.ownershipLock.close()
	if closeErr != nil {
		coordinator.serviceRefs = 1
		coordinator.releaseBlocked = fmt.Errorf("close evaluation root ownership lock: %w", closeErr)
		return coordinator.releaseBlocked
	}
	ownership.coordinator = nil
	coordinator.ownershipLock = nil
	delete(evaluationRootCoordinators.byRoot, coordinator.root)
	return nil
}

func (coordinator *evaluationRootCoordinator) requireReleaseQuiescence() error {
	coordinator.capacityMu.Lock()
	workerSlots := 0
	if coordinator.workerSlots != nil {
		workerSlots = len(coordinator.workerSlots)
	}
	evidenceReads := len(coordinator.evidenceReads)
	coordinator.capacityMu.Unlock()

	coordinator.activityMu.Lock()
	activeRuns := len(coordinator.activeRuns)
	coordinator.activityMu.Unlock()
	coordinator.controlledPairLaunchMu.Lock()
	pairLaunches := len(coordinator.controlledPairLaunches)
	coordinator.controlledPairLaunchMu.Unlock()
	coordinator.eventSubscribers.mu.Lock()
	subscribers := coordinator.eventSubscribers.subscriberCount
	coordinator.eventSubscribers.mu.Unlock()
	if workerSlots != 0 || evidenceReads != 0 || activeRuns != 0 || pairLaunches != 0 || subscribers != 0 {
		return fmt.Errorf(
			"worker_slots=%d evidence_reads=%d active_runs=%d pair_launches=%d subscribers=%d",
			workerSlots, evidenceReads, activeRuns, pairLaunches, subscribers,
		)
	}
	return nil
}
