package publicationreplica

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
)

var ErrNotReady = errors.New("routing publication replica is not ready")

// Store is the complete Redis-facing contract used by a data-plane replica.
// Implementations keep namespace operations partition-local. The only global
// operations are the bounded namespace locator read and process-liveness lease.
type Store interface {
	RegisterFleetReplica(context.Context, string) (time.Time, error)
	ListPublicationNamespaces(context.Context) ([]accesspublisher.NamespacePublication, error)
	ReadPublicationHeads(context.Context, accesspublisher.NamespacePublication) (accesspublisher.PublicationHeads, error)
	LoadRoutingPublication(context.Context, accesspublisher.RuntimePublicationIdentity) (accesspublisher.LoadedRoutingPublication, error)
	RegisterReplica(context.Context, string, string, accesspublisher.ReplicaRegistration) (time.Time, error)
	AcknowledgeBarriers(context.Context, string, string, string, string, string) error
	AcknowledgeRouting(context.Context, string, string, string, string, string) error
}

// SnapshotLifecycle is the seam to an in-process routing generation registry.
// Warm and Activate must be idempotent for one publication identity. Activate
// is called only after that identity is observed at the active gate. Remove is
// called when a namespace leaves the bounded locator or the manager stops.
type SnapshotLifecycle interface {
	Warm(context.Context, accesspublisher.LoadedRoutingPublication) error
	Activate(context.Context, accesspublisher.LoadedRoutingPublication) error
	Remove(context.Context, accesspublisher.NamespacePublication) error
}

type Options struct {
	Store             Store
	Snapshots         SnapshotLifecycle
	ReplicaID         string
	DiscoveryInterval time.Duration
	PollInterval      time.Duration
	RenewInterval     time.Duration
}

type NamespaceStatus struct {
	Namespace   accesspublisher.NamespacePublication
	Ready       bool
	Reason      string
	Loaded      *accesspublisher.RuntimePublicationIdentity
	Candidate   *accesspublisher.RuntimePublicationIdentity
	LeaseExpiry time.Time
}

type Status struct {
	Ready           bool
	Reason          string
	DirectorySynced bool
	Namespaces      []NamespaceStatus
}

type Manager struct {
	store             Store
	snapshots         SnapshotLifecycle
	replicaID         string
	discoveryInterval time.Duration
	pollInterval      time.Duration
	renewInterval     time.Duration

	mu               sync.RWMutex
	running          bool
	fleetLeaseExpiry time.Time
	fleetLeaseErr    error
	directorySynced  bool
	directoryErr     error
	workers          map[string]*namespaceWorker
	fleetRenewMu     sync.Mutex
}

func New(options Options) (*Manager, error) {
	if options.Store == nil || options.Snapshots == nil {
		return nil, fmt.Errorf("publication replica store and snapshot lifecycle are required")
	}
	if strings.TrimSpace(options.ReplicaID) == "" || len(options.ReplicaID) > 256 || strings.ContainsRune(options.ReplicaID, 0) {
		return nil, fmt.Errorf("publication replica id is required and must not exceed 256 bytes")
	}
	discoveryInterval, err := boundedInterval("discovery", options.DiscoveryInterval, time.Second)
	if err != nil {
		return nil, err
	}
	pollInterval, err := boundedInterval("publication poll", options.PollInterval, 250*time.Millisecond)
	if err != nil {
		return nil, err
	}
	renewInterval, err := boundedInterval("lease renewal", options.RenewInterval, 10*time.Second)
	if err != nil {
		return nil, err
	}
	return &Manager{
		store: options.Store, snapshots: options.Snapshots, replicaID: options.ReplicaID,
		discoveryInterval: discoveryInterval, pollInterval: pollInterval, renewInterval: renewInterval,
		fleetLeaseErr: ErrNotReady, workers: make(map[string]*namespaceWorker),
	}, nil
}

func boundedInterval(name string, value, fallback time.Duration) (time.Duration, error) {
	if value == 0 {
		value = fallback
	}
	if value < 10*time.Millisecond || value > time.Minute {
		return 0, fmt.Errorf("%s interval must be between 10ms and one minute", name)
	}
	return value, nil
}

func (m *Manager) Ready() error {
	status := m.Status()
	if status.Ready {
		return nil
	}
	return fmt.Errorf("%w: %s", ErrNotReady, status.Reason)
}

// EnsureFleetLease synchronously establishes this process's bounded fleet
// membership before a startup reconciliation can publish pending work. Run
// owns subsequent renewal; both paths share one serialized lease operation.
func (m *Manager) EnsureFleetLease(ctx context.Context) error {
	return m.renewFleetLease(ctx)
}

func (m *Manager) Status() Status {
	m.mu.RLock()
	running, fleetExpiry, fleetErr := m.running, m.fleetLeaseExpiry, m.fleetLeaseErr
	synced, directoryErr := m.directorySynced, m.directoryErr
	workers := make([]*namespaceWorker, 0, len(m.workers))
	for _, worker := range m.workers {
		workers = append(workers, worker)
	}
	m.mu.RUnlock()
	fleetReady := fleetErr == nil && !fleetExpiry.IsZero() && time.Now().Before(fleetExpiry)
	status := Status{DirectorySynced: synced, Ready: running && fleetReady && synced && directoryErr == nil}
	switch {
	case !running:
		status.Reason = "not_running"
	case !fleetReady:
		status.Reason = "fleet_lease_unavailable"
	case !synced:
		status.Reason = "namespace_directory_unavailable"
	case directoryErr != nil:
		status.Reason = "namespace_directory_unavailable"
	default:
		status.Reason = "ready"
	}
	for _, worker := range workers {
		namespace := worker.status()
		status.Namespaces = append(status.Namespaces, namespace)
		if !namespace.Ready {
			status.Ready = false
			if status.Reason == "ready" {
				status.Reason = "namespace_" + namespace.Reason
			}
		}
	}
	sort.Slice(status.Namespaces, func(i, j int) bool {
		return status.Namespaces[i].Namespace.NamespaceID < status.Namespaces[j].Namespace.NamespaceID
	})
	return status
}

// Current returns a generation only while its namespace lease and process
// state are healthy. Request admission can compare this identity with its
// Redis-pinned tenant context and fail closed on a rollout boundary.
func (m *Manager) Current(namespaceID string) (accesspublisher.RuntimePublicationIdentity, bool) {
	m.mu.RLock()
	worker := m.workers[namespaceID]
	fleetReady := m.running && m.fleetLeaseErr == nil && !m.fleetLeaseExpiry.IsZero() &&
		time.Now().Before(m.fleetLeaseExpiry)
	m.mu.RUnlock()
	if !fleetReady || worker == nil {
		return accesspublisher.RuntimePublicationIdentity{}, false
	}
	status := worker.status()
	if !status.Ready || status.Loaded == nil {
		return accesspublisher.RuntimePublicationIdentity{}, false
	}
	return *status.Loaded, true
}
