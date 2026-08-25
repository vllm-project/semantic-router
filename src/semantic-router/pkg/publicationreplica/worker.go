package publicationreplica

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
)

type namespaceWorkerOptions struct {
	store         Store
	snapshots     SnapshotLifecycle
	reference     accesspublisher.NamespacePublication
	replicaID     string
	pollInterval  time.Duration
	renewInterval time.Duration
}

type namespaceWorker struct {
	store         Store
	snapshots     SnapshotLifecycle
	reference     accesspublisher.NamespacePublication
	replicaID     string
	pollInterval  time.Duration
	renewInterval time.Duration

	ctx      context.Context
	cancel   context.CancelFunc
	stopOnce sync.Once

	mu           sync.RWMutex
	processErr   error
	leaseErr     error
	loaded       *accesspublisher.LoadedRoutingPublication
	candidate    *accesspublisher.RuntimePublicationIdentity
	leaseExpiry  time.Time
	registration *accesspublisher.ReplicaRegistration
	prepared     map[string]accesspublisher.LoadedRoutingPublication
	wake         chan struct{}

	registerMu sync.Mutex
}

func newNamespaceWorker(options namespaceWorkerOptions) *namespaceWorker {
	ctx, cancel := context.WithCancel(context.Background())
	return &namespaceWorker{
		store: options.store, snapshots: options.snapshots, reference: options.reference,
		replicaID: options.replicaID, pollInterval: options.pollInterval, renewInterval: options.renewInterval,
		ctx: ctx, cancel: cancel, processErr: ErrNotReady,
		leaseErr: ErrNotReady, prepared: make(map[string]accesspublisher.LoadedRoutingPublication),
		wake: make(chan struct{}, 1),
	}
}

func (w *namespaceWorker) stop() { w.stopOnce.Do(w.cancel) }

func (w *namespaceWorker) wakeNow() {
	select {
	case w.wake <- struct{}{}:
	default:
	}
}

func (w *namespaceWorker) run(parent context.Context) {
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	go func() {
		select {
		case <-w.ctx.Done():
			cancel()
		case <-ctx.Done():
		}
	}()
	var renewWait sync.WaitGroup
	renewWait.Add(1)
	go func() {
		defer renewWait.Done()
		w.renewLoop(ctx)
	}()

	w.process(ctx)
	ticker := time.NewTicker(w.pollInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			renewWait.Wait()
			removeCtx, removeCancel := context.WithTimeout(context.Background(), 5*time.Second)
			_ = w.snapshots.Remove(removeCtx, w.reference)
			removeCancel()
			return
		case <-ticker.C:
			w.process(ctx)
		case <-w.wake:
			w.process(ctx)
		}
	}
}

func (w *namespaceWorker) renewLoop(ctx context.Context) {
	ticker := time.NewTicker(w.renewInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if err := w.renewCurrent(ctx); err != nil {
				w.setLeaseError(err)
			}
		}
	}
}

func (w *namespaceWorker) process(ctx context.Context) {
	heads, err := w.store.ReadPublicationHeads(ctx, w.reference)
	if err != nil {
		w.setProcessError(err)
		return
	}
	w.setCandidate(heads.Candidate)
	if heads.Active == nil {
		w.setProcessError(fmt.Errorf("%w: active publication is absent", ErrNotReady))
		if heads.Candidate == nil || !heads.Candidate.Loadable() {
			return
		}
		registration := accesspublisher.ReplicaRegistration{
			ReplicaID: w.replicaID, RuntimeEpoch: heads.Candidate.RuntimeEpoch,
		}
		if err := w.register(ctx, registration); err != nil {
			w.setLeaseError(err)
			return
		}
		if err := w.prepareAndAcknowledge(ctx, heads, *heads.Candidate); err != nil {
			w.setProcessError(err)
		}
		return
	}
	if err := w.ensureActive(ctx, *heads.Active); err != nil {
		w.setProcessError(err)
		return
	}
	w.setProcessError(nil)
	if w.leaseNeedsRecovery() {
		if err := w.renewCurrent(ctx); err != nil {
			w.setLeaseError(err)
			return
		}
	}
	if heads.Candidate == nil || !heads.Candidate.Loadable() {
		w.prunePrepared(heads.Active.PublicationID, "")
		return
	}
	if err := w.prepareAndAcknowledge(ctx, heads, *heads.Candidate); err != nil {
		w.setProcessError(err)
		return
	}
	w.setProcessError(nil)
	w.prunePrepared(heads.Active.PublicationID, heads.Candidate.PublicationID)
}

func (w *namespaceWorker) ensureActive(
	ctx context.Context,
	identity accesspublisher.RuntimePublicationIdentity,
) error {
	w.mu.RLock()
	loaded := w.loaded
	w.mu.RUnlock()
	if loaded != nil && loaded.Identity.SameGeneration(identity) {
		return nil
	}
	w.setProcessError(fmt.Errorf("%w: local generation differs from active gate", ErrNotReady))
	publication, err := w.preload(ctx, identity)
	if err != nil {
		return err
	}
	current, err := w.store.ReadPublicationHeads(ctx, w.reference)
	if err != nil {
		return err
	}
	if current.Active == nil || !current.Active.SameGeneration(identity) {
		return accesspublisher.ErrPublicationChanged
	}
	if err := w.snapshots.Activate(ctx, publication); err != nil {
		return fmt.Errorf("activate routing publication %s: %w", identity.PublicationID, err)
	}
	registration := accesspublisher.ReplicaRegistration{
		ReplicaID: w.replicaID, RuntimeEpoch: identity.RuntimeEpoch,
		AccessPublication: identity.PublicationID, RoutingPublication: identity.PublicationID,
	}
	if err := w.register(ctx, registration); err != nil {
		return err
	}
	w.mu.Lock()
	copy := publication
	w.loaded = &copy
	w.mu.Unlock()
	return nil
}

func (w *namespaceWorker) prepareAndAcknowledge(
	ctx context.Context,
	heads accesspublisher.PublicationHeads,
	identity accesspublisher.RuntimePublicationIdentity,
) error {
	if _, err := w.preload(ctx, identity); err != nil {
		return err
	}
	current, err := w.store.ReadPublicationHeads(ctx, w.reference)
	if err != nil {
		return err
	}
	if current.Active != nil && current.Active.SameGeneration(identity) {
		return nil
	}
	if current.Candidate == nil || !current.Candidate.SameGeneration(identity) || !current.Candidate.Loadable() {
		return accesspublisher.ErrPublicationChanged
	}
	registration := w.registrationFor(heads, identity)
	if err := w.register(ctx, registration); err != nil {
		return err
	}
	if identity.Restrictive {
		if err := w.store.AcknowledgeBarriers(ctx, identity.NamespaceID, identity.QuotaPartition,
			w.replicaID, identity.PublicationID, identity.PublicationDigest); err != nil {
			return fmt.Errorf("acknowledge publication barriers: %w", err)
		}
	}
	if err := w.store.AcknowledgeRouting(ctx, identity.NamespaceID, identity.QuotaPartition,
		w.replicaID, identity.PublicationID, identity.PublicationDigest); err != nil {
		return fmt.Errorf("acknowledge routing publication: %w", err)
	}
	return nil
}

func (w *namespaceWorker) registrationFor(
	heads accesspublisher.PublicationHeads,
	candidate accesspublisher.RuntimePublicationIdentity,
) accesspublisher.ReplicaRegistration {
	registration := accesspublisher.ReplicaRegistration{ReplicaID: w.replicaID, RuntimeEpoch: candidate.RuntimeEpoch}
	if heads.Active != nil {
		registration.RuntimeEpoch = heads.Active.RuntimeEpoch
		registration.AccessPublication = heads.Active.PublicationID
		registration.RoutingPublication = heads.Active.PublicationID
	}
	return registration
}

func (w *namespaceWorker) preload(
	ctx context.Context,
	identity accesspublisher.RuntimePublicationIdentity,
) (accesspublisher.LoadedRoutingPublication, error) {
	w.mu.RLock()
	prepared, exists := w.prepared[identity.PublicationID]
	w.mu.RUnlock()
	if exists && prepared.Identity.SameGeneration(identity) {
		// The immutable payload is reusable after activation, but lifecycle
		// state comes from the latest observed head. Activate and Current must
		// expose the active identity rather than the earlier validated view.
		prepared.Identity = identity
		return prepared, nil
	}
	publication, err := w.store.LoadRoutingPublication(ctx, identity)
	if err != nil {
		return accesspublisher.LoadedRoutingPublication{}, err
	}
	if err := w.snapshots.Warm(ctx, publication); err != nil {
		return accesspublisher.LoadedRoutingPublication{}, fmt.Errorf("warm routing publication %s: %w", identity.PublicationID, err)
	}
	w.mu.Lock()
	w.prepared[identity.PublicationID] = publication
	w.mu.Unlock()
	return publication, nil
}

func (w *namespaceWorker) register(ctx context.Context, registration accesspublisher.ReplicaRegistration) error {
	w.registerMu.Lock()
	defer w.registerMu.Unlock()
	expiry, err := w.store.RegisterReplica(ctx, w.reference.NamespaceID, w.reference.QuotaPartition, registration)
	if err != nil {
		return err
	}
	w.mu.Lock()
	copy := registration
	w.registration = &copy
	w.leaseExpiry = expiry
	w.leaseErr = nil
	w.mu.Unlock()
	return nil
}

func (w *namespaceWorker) renewCurrent(ctx context.Context) error {
	w.registerMu.Lock()
	defer w.registerMu.Unlock()
	w.mu.RLock()
	registration := w.registration
	w.mu.RUnlock()
	if registration == nil {
		return fmt.Errorf("%w: replica registration is absent", ErrNotReady)
	}
	expiry, err := w.store.RegisterReplica(ctx, w.reference.NamespaceID, w.reference.QuotaPartition, *registration)
	if err != nil {
		return err
	}
	w.mu.Lock()
	w.leaseExpiry = expiry
	w.leaseErr = nil
	w.mu.Unlock()
	return nil
}

func (w *namespaceWorker) leaseNeedsRecovery() bool {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.leaseErr != nil || w.leaseExpiry.IsZero() || !time.Now().Before(w.leaseExpiry)
}

func (w *namespaceWorker) prunePrepared(activeID, candidateID string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	for publicationID := range w.prepared {
		if publicationID != activeID && publicationID != candidateID {
			delete(w.prepared, publicationID)
		}
	}
}

func (w *namespaceWorker) setProcessError(err error) {
	w.mu.Lock()
	w.processErr = err
	w.mu.Unlock()
}

func (w *namespaceWorker) setLeaseError(err error) {
	w.mu.Lock()
	w.leaseErr = err
	w.mu.Unlock()
}

func (w *namespaceWorker) setCandidate(candidate *accesspublisher.RuntimePublicationIdentity) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if candidate == nil {
		w.candidate = nil
		return
	}
	copy := *candidate
	w.candidate = &copy
}

func (w *namespaceWorker) status() NamespaceStatus {
	w.mu.RLock()
	defer w.mu.RUnlock()
	status := NamespaceStatus{
		Namespace: w.reference, LeaseExpiry: w.leaseExpiry,
		Ready: w.processErr == nil && w.leaseErr == nil && w.loaded != nil && time.Now().Before(w.leaseExpiry),
	}
	if w.loaded != nil {
		copy := w.loaded.Identity
		status.Loaded = &copy
	}
	if w.candidate != nil {
		copy := *w.candidate
		status.Candidate = &copy
	}
	switch {
	case w.processErr != nil:
		if errors.Is(w.processErr, accesspublisher.ErrStagedCorrupt) {
			status.Reason = "publication_corrupt"
		} else if errors.Is(w.processErr, accesspublisher.ErrPublicationChanged) {
			status.Reason = "publication_changed"
		} else {
			status.Reason = "publication_unavailable"
		}
	case w.leaseErr != nil || w.leaseExpiry.IsZero() || !time.Now().Before(w.leaseExpiry):
		status.Reason = "lease_unavailable"
	case w.loaded == nil:
		status.Reason = "active_publication_unloaded"
	default:
		status.Reason = "ready"
	}
	return status
}
