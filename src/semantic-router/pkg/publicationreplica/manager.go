package publicationreplica

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
)

// Run continuously reconciles the bounded namespace directory. Each namespace
// owns an independent publication and lease loop, so a slow warm callback does
// not prevent other namespaces from renewing.
func (m *Manager) Run(ctx context.Context) error {
	m.mu.Lock()
	if m.running {
		m.mu.Unlock()
		return fmt.Errorf("publication replica manager is already running")
	}
	m.running = true
	m.mu.Unlock()

	var workerWait sync.WaitGroup
	var fleetWait sync.WaitGroup
	fleetWait.Add(1)
	go func() {
		defer fleetWait.Done()
		m.runFleetLease(ctx)
	}()
	reconcile := func() { m.reconcileNamespaces(ctx, &workerWait) }
	reconcile()
	var notifications <-chan struct{}
	if source, ok := m.store.(NotificationStore); ok {
		notifications = source.PublicationNotifications(ctx)
	}
	ticker := time.NewTicker(m.discoveryInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			m.stopNamespaceWorkers()
			workerWait.Wait()
			fleetWait.Wait()
			return nil
		case <-ticker.C:
			reconcile()
		case _, ok := <-notifications:
			if !ok {
				notifications = nil
				continue
			}
			reconcile()
			m.wakeWorkers()
		}
	}
}

func (m *Manager) reconcileNamespaces(ctx context.Context, workerWait *sync.WaitGroup) {
	references, err := m.store.ListPublicationNamespaces(ctx)
	if err != nil {
		m.recordDirectoryError(err)
		return
	}
	present := make(map[string]accesspublisher.NamespacePublication, len(references))
	for _, reference := range references {
		if err := reference.Validate(); err != nil {
			m.recordDirectoryError(err)
			return
		}
		present[reference.NamespaceID] = reference
	}
	var removed []*namespaceWorker
	m.mu.Lock()
	m.directorySynced = true
	m.directoryErr = nil
	for namespaceID, reference := range present {
		if existing := m.workers[namespaceID]; existing != nil {
			if existing.reference.QuotaPartition != reference.QuotaPartition {
				m.directoryErr = fmt.Errorf("namespace %s changed quota partition", namespaceID)
			}
			continue
		}
		worker := newNamespaceWorker(namespaceWorkerOptions{
			store: m.store, snapshots: m.snapshots, reference: reference, replicaID: m.replicaID,
			pollInterval: m.pollInterval, renewInterval: m.renewInterval,
		})
		m.workers[namespaceID] = worker
		workerWait.Add(1)
		go func() {
			defer workerWait.Done()
			worker.run(ctx)
		}()
	}
	for namespaceID, worker := range m.workers {
		if _, exists := present[namespaceID]; exists {
			continue
		}
		delete(m.workers, namespaceID)
		removed = append(removed, worker)
	}
	m.mu.Unlock()
	for _, worker := range removed {
		worker.stop()
	}
}

func (m *Manager) recordDirectoryError(err error) {
	m.mu.Lock()
	m.directorySynced = true
	m.directoryErr = err
	m.mu.Unlock()
}

func (m *Manager) stopNamespaceWorkers() {
	m.mu.Lock()
	workers := make([]*namespaceWorker, 0, len(m.workers))
	for _, worker := range m.workers {
		workers = append(workers, worker)
	}
	m.workers = make(map[string]*namespaceWorker)
	m.running = false
	m.mu.Unlock()
	for _, worker := range workers {
		worker.stop()
	}
}

func (m *Manager) wakeWorkers() {
	m.mu.RLock()
	workers := make([]*namespaceWorker, 0, len(m.workers))
	for _, worker := range m.workers {
		workers = append(workers, worker)
	}
	m.mu.RUnlock()
	for _, worker := range workers {
		worker.wakeNow()
	}
}

// runFleetLease maintains process liveness independently of namespace
// discovery. Publishers use this bounded membership view to require every live
// Router process to warm and acknowledge a namespace's first publication.
func (m *Manager) runFleetLease(ctx context.Context) {
	_ = m.renewFleetLease(ctx)
	ticker := time.NewTicker(m.renewInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			_ = m.renewFleetLease(ctx)
		}
	}
}

func (m *Manager) renewFleetLease(ctx context.Context) error {
	m.fleetRenewMu.Lock()
	defer m.fleetRenewMu.Unlock()
	expiry, err := m.store.RegisterFleetReplica(ctx, m.replicaID)
	m.mu.Lock()
	if err != nil {
		m.fleetLeaseErr = err
		m.mu.Unlock()
		return err
	}
	m.fleetLeaseExpiry = expiry
	m.fleetLeaseErr = nil
	m.mu.Unlock()
	return nil
}
