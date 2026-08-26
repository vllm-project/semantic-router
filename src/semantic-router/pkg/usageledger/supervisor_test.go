package usageledger

import (
	"context"
	"database/sql"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"
)

const secondTestNamespaceID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"

type supervisorNamespaceSource struct {
	mu         sync.RWMutex
	namespaces []ActiveNamespace
	err        error
}

func (source *supervisorNamespaceSource) ListActiveNamespaces(context.Context) ([]ActiveNamespace, error) {
	source.mu.RLock()
	defer source.mu.RUnlock()
	return append([]ActiveNamespace(nil), source.namespaces...), source.err
}

func (source *supervisorNamespaceSource) set(namespaces ...ActiveNamespace) {
	source.mu.Lock()
	source.namespaces = append([]ActiveNamespace(nil), namespaces...)
	source.mu.Unlock()
}

type supervisorStreamFactory struct {
	mu        sync.Mutex
	opened    map[string]*supervisorStream
	consumers map[string]string
}

func (factory *supervisorStreamFactory) OpenNamespaceStream(
	namespace ActiveNamespace,
	_, consumer string,
) (Stream, error) {
	factory.mu.Lock()
	defer factory.mu.Unlock()
	if factory.opened == nil {
		factory.opened = make(map[string]*supervisorStream)
		factory.consumers = make(map[string]string)
	}
	stream := &supervisorStream{}
	factory.opened[namespace.ID] = stream
	factory.consumers[namespace.ID] = consumer
	return stream, nil
}

type supervisorStream struct {
	mu       sync.Mutex
	ensured  int
	canceled bool
}

func (stream *supervisorStream) EnsureGroup(context.Context) error {
	stream.mu.Lock()
	stream.ensured++
	stream.mu.Unlock()
	return nil
}

func (*supervisorStream) ClaimStale(context.Context, int64, time.Duration) ([]StreamItem, error) {
	return nil, nil
}

func (stream *supervisorStream) ReadNew(ctx context.Context, _ int64, _ time.Duration) ([]StreamItem, error) {
	select {
	case <-ctx.Done():
		stream.mu.Lock()
		stream.canceled = true
		stream.mu.Unlock()
		return nil, ctx.Err()
	case <-time.After(time.Millisecond):
		return nil, nil
	}
}

func (*supervisorStream) Ack(context.Context, []string) error { return nil }
func (*supervisorStream) Quarantine(context.Context, StreamItem, string) (bool, error) {
	return true, nil
}
func (*supervisorStream) Quarantined(context.Context) (int64, error) { return 0, nil }

type supervisorStore struct{}

func (supervisorStore) PersistBatch(context.Context, []TerminalEvent) (BatchResult, error) {
	return BatchResult{}, nil
}

type supervisorStorage struct {
	mu  sync.RWMutex
	err error
}

func (storage *supervisorStorage) LockWriterTx(context.Context, *sql.Tx) error {
	return storage.currentError()
}

func (storage *supervisorStorage) EnsureTx(context.Context, *sql.Tx, []time.Time) error {
	return storage.currentError()
}

func (storage *supervisorStorage) Reconcile(context.Context) (StorageMaintenance, error) {
	return StorageMaintenance{}, storage.currentError()
}

func (storage *supervisorStorage) currentError() error {
	storage.mu.RLock()
	defer storage.mu.RUnlock()
	return storage.err
}

func (storage *supervisorStorage) setError(err error) {
	storage.mu.Lock()
	storage.err = err
	storage.mu.Unlock()
}

type supervisorRollups struct {
	mu    sync.Mutex
	calls map[string]int
	err   error
}

func (rollups *supervisorRollups) ProcessDirty(_ context.Context, namespaceID string) (RollupResult, error) {
	rollups.mu.Lock()
	defer rollups.mu.Unlock()
	if rollups.calls == nil {
		rollups.calls = make(map[string]int)
	}
	rollups.calls[namespaceID]++
	return RollupResult{}, rollups.err
}

func (rollups *supervisorRollups) AfterCommit(context.Context, []TerminalEvent) error {
	rollups.mu.Lock()
	defer rollups.mu.Unlock()
	return rollups.err
}

func TestSupervisorReconcilesNamespaceWorkersAndLifecycle(t *testing.T) {
	namespaceOne := ActiveNamespace{ID: testNamespaceID, QuotaPartitionID: "partition-one"}
	namespaceTwo := ActiveNamespace{ID: secondTestNamespaceID, QuotaPartitionID: "partition-two"}
	source := &supervisorNamespaceSource{namespaces: []ActiveNamespace{namespaceOne}}
	streams := &supervisorStreamFactory{}
	rollups := &supervisorRollups{}
	supervisor, err := NewSupervisor(SupervisorOptions{
		Namespaces: source, Streams: streams, Store: supervisorStore{}, Rollups: rollups,
		Storage:   &supervisorStorage{},
		ReplicaID: "replica-a", Block: time.Millisecond, ReclaimIdle: time.Millisecond,
		ReconcileInterval: 10 * time.Millisecond, RollupInterval: 10 * time.Millisecond,
		MinBackoff: time.Millisecond, MaxBackoff: 10 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := supervisor.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := supervisor.Ready(context.Background()); err == nil {
		t.Fatal("supervisor became ready before Run")
	}

	runContext, cancel := context.WithCancel(context.Background())
	runResult := make(chan error, 1)
	go func() { runResult <- supervisor.Run(runContext) }()
	select {
	case <-supervisor.Started():
	case <-time.After(time.Second):
		t.Fatal("supervisor did not start")
	}
	if err := supervisor.Ready(context.Background()); err != nil {
		t.Fatalf("Ready() = %v", err)
	}

	source.set(namespaceTwo)
	if err := supervisor.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := supervisor.Ready(context.Background()); err != nil {
		t.Fatalf("Ready() after namespace replacement = %v", err)
	}
	streams.mu.Lock()
	consumerOne := streams.consumers[testNamespaceID]
	consumerTwo := streams.consumers[secondTestNamespaceID]
	firstStream := streams.opened[testNamespaceID]
	streams.mu.Unlock()
	if consumerOne == consumerTwo || len(consumerOne) > 128 || len(consumerTwo) > 128 {
		t.Fatalf("namespace consumers = %q and %q, want distinct bounded names", consumerOne, consumerTwo)
	}
	firstStream.mu.Lock()
	firstCanceled := firstStream.canceled
	firstStream.mu.Unlock()
	if !firstCanceled {
		t.Fatal("removed namespace worker was not synchronously stopped")
	}

	cancel()
	if err := supervisor.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-runResult:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Run() = %v, want context cancellation", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Run did not stop")
	}
	if err := supervisor.Ready(context.Background()); err == nil {
		t.Fatal("closed supervisor remained ready")
	}
}

func TestSupervisorRejectsAmbiguousNamespacePartitions(t *testing.T) {
	source := &supervisorNamespaceSource{namespaces: []ActiveNamespace{
		{ID: testNamespaceID, QuotaPartitionID: "shared"},
		{ID: secondTestNamespaceID, QuotaPartitionID: "shared"},
	}}
	supervisor, err := NewSupervisor(SupervisorOptions{
		Namespaces: source, Streams: &supervisorStreamFactory{}, Store: supervisorStore{},
		Rollups: &supervisorRollups{}, Storage: &supervisorStorage{}, ReplicaID: "replica-a",
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := supervisor.Reconcile(context.Background()); err == nil {
		t.Fatal("shared quota partition unexpectedly reconciled")
	}
}

func TestStorageMaintenanceFailureDegradesReadinessWithoutRemovingWorkers(t *testing.T) {
	source := &supervisorNamespaceSource{namespaces: []ActiveNamespace{{
		ID: testNamespaceID, QuotaPartitionID: "partition-one",
	}}}
	storage := &supervisorStorage{}
	supervisor, err := NewSupervisor(SupervisorOptions{
		Namespaces: source, Streams: &supervisorStreamFactory{}, Store: supervisorStore{},
		Rollups: &supervisorRollups{}, Storage: storage, ReplicaID: "replica-a",
		Block: time.Millisecond, ReclaimIdle: time.Millisecond,
		ReconcileInterval: 10 * time.Millisecond, RollupInterval: 10 * time.Millisecond,
		MinBackoff: time.Millisecond, MaxBackoff: 10 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	runContext, cancel := context.WithCancel(context.Background())
	defer cancel()
	runResult := make(chan error, 1)
	go func() { runResult <- supervisor.Run(runContext) }()
	select {
	case <-supervisor.Started():
	case <-time.After(time.Second):
		t.Fatal("supervisor did not start")
	}
	storage.setError(errors.New("partition maintenance unavailable"))
	if err := supervisor.Reconcile(context.Background()); err == nil {
		t.Fatal("storage maintenance failure unexpectedly reconciled")
	}
	if err := supervisor.Ready(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "partition maintenance unavailable") {
		t.Fatalf("Ready() = %v, want storage maintenance failure", err)
	}
	supervisor.mu.RLock()
	workers := len(supervisor.workers)
	supervisor.mu.RUnlock()
	if workers != 1 {
		t.Fatalf("workers after maintenance failure = %d, want existing worker retained", workers)
	}
	storage.setError(nil)
	if err := supervisor.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := supervisor.Ready(context.Background()); err != nil {
		t.Fatalf("Ready() after storage recovery = %v", err)
	}
	if err := supervisor.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-runResult:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Run() = %v, want context cancellation", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Run did not stop")
	}
}

func TestContiguousIntervalsSplitGapsAndMaximums(t *testing.T) {
	start := time.Date(2026, 8, 22, 0, 0, 0, 0, time.UTC)
	buckets := []time.Time{start, start.Add(time.Minute), start.Add(3 * time.Minute), start.Add(4 * time.Minute)}
	intervals := contiguousIntervals(buckets, time.Minute, 2*time.Minute)
	if len(intervals) != 2 || intervals[0].start != start || intervals[0].end != start.Add(2*time.Minute) ||
		intervals[1].start != start.Add(3*time.Minute) || intervals[1].end != start.Add(5*time.Minute) {
		t.Fatalf("intervals = %+v", intervals)
	}
}
