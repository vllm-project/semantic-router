package usageledger

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"
)

const (
	defaultConsumerGroup     = "usage-writers"
	defaultReconcileInterval = 5 * time.Second
	defaultRollupInterval    = 5 * time.Second
	defaultSupervisorMinWait = 100 * time.Millisecond
	defaultSupervisorMaxWait = 5 * time.Second
	maxReplicaIDLength       = 80
)

// SupervisorOptions defines one Router replica's usage lifecycle.
type SupervisorOptions struct {
	Namespaces NamespaceSource
	Streams    StreamFactory
	Store      Store
	Rollups    RollupProcessor
	Storage    StorageLifecycle

	ReplicaID         string
	ConsumerGroup     string
	BatchSize         int64
	Block             time.Duration
	ReclaimIdle       time.Duration
	ReconcileInterval time.Duration
	RollupInterval    time.Duration
	MinBackoff        time.Duration
	MaxBackoff        time.Duration
}

// Supervisor owns every usage ingestion and rollup worker in one Router
// replica. It continuously reconciles the active PostgreSQL namespace set;
// the Dashboard is not involved in discovery or execution.
type Supervisor struct {
	namespaces  NamespaceSource
	streams     StreamFactory
	store       Store
	rollups     RollupProcessor
	storage     StorageLifecycle
	afterCommit CommittedBatchHook

	replicaID         string
	consumerGroup     string
	workerOptions     WorkerOptions
	reconcileInterval time.Duration
	rollupInterval    time.Duration
	minBackoff        time.Duration
	maxBackoff        time.Duration

	reconcileMu sync.Mutex
	mu          sync.RWMutex
	workers     map[string]*namespaceWorker
	initialized bool
	runInvoked  bool
	running     bool
	closed      bool
	runContext  context.Context
	runCancel   context.CancelFunc
	lastSuccess time.Time
	lastErr     error

	started     chan struct{}
	startedOnce sync.Once
	runDone     chan struct{}
	runDoneOnce sync.Once
	closeOnce   sync.Once
}

func NewSupervisor(options SupervisorOptions) (*Supervisor, error) {
	if options.Namespaces == nil || options.Streams == nil || options.Store == nil ||
		options.Rollups == nil || options.Storage == nil {
		return nil, fmt.Errorf("usage supervisor namespace, stream, store, rollup, and storage dependencies are required")
	}
	afterCommit, ok := options.Rollups.(CommittedBatchHook)
	if !ok {
		return nil, fmt.Errorf("usage rollup processor must project committed stream batches")
	}
	if len(options.ReplicaID) > maxReplicaIDLength || !partitionPattern.MatchString(options.ReplicaID) {
		return nil, fmt.Errorf("usage supervisor replica ID is not canonical")
	}
	if options.ConsumerGroup == "" {
		options.ConsumerGroup = defaultConsumerGroup
	}
	if err := validateConsumerName("group", options.ConsumerGroup); err != nil {
		return nil, err
	}
	if options.ReconcileInterval == 0 {
		options.ReconcileInterval = defaultReconcileInterval
	}
	if options.RollupInterval == 0 {
		options.RollupInterval = defaultRollupInterval
	}
	if options.MinBackoff == 0 {
		options.MinBackoff = defaultSupervisorMinWait
	}
	if options.MaxBackoff == 0 {
		options.MaxBackoff = defaultSupervisorMaxWait
	}
	if options.ReconcileInterval < time.Millisecond || options.ReconcileInterval > time.Minute ||
		options.RollupInterval < time.Millisecond || options.RollupInterval > time.Hour ||
		options.MinBackoff < time.Millisecond || options.MinBackoff > time.Minute ||
		options.MaxBackoff < options.MinBackoff || options.MaxBackoff > time.Minute {
		return nil, fmt.Errorf("usage supervisor intervals are invalid")
	}
	workerOptions := WorkerOptions{
		BatchSize: options.BatchSize, Block: options.Block, ReclaimIdle: options.ReclaimIdle,
	}
	// Validate all worker defaults and bounds once with a harmless placeholder
	// namespace; the real namespace is installed during reconciliation.
	if _, err := normalizedWorkerOptions(workerOptions); err != nil {
		return nil, err
	}
	return &Supervisor{
		namespaces: options.Namespaces, streams: options.Streams, store: options.Store,
		rollups: options.Rollups, storage: options.Storage, afterCommit: afterCommit, replicaID: options.ReplicaID,
		consumerGroup: options.ConsumerGroup, workerOptions: workerOptions,
		reconcileInterval: options.ReconcileInterval, rollupInterval: options.RollupInterval,
		minBackoff: options.MinBackoff, maxBackoff: options.MaxBackoff,
		workers: make(map[string]*namespaceWorker), started: make(chan struct{}), runDone: make(chan struct{}),
	}, nil
}

func normalizedWorkerOptions(options WorkerOptions) (WorkerOptions, error) {
	if options.BatchSize == 0 {
		options.BatchSize = 200
	}
	if options.BatchSize < 1 || options.BatchSize > 1000 {
		return WorkerOptions{}, fmt.Errorf("usage worker batch size must be between 1 and 1000")
	}
	if options.Block == 0 {
		options.Block = time.Second
	}
	if options.Block < 0 || options.Block > time.Minute {
		return WorkerOptions{}, fmt.Errorf("usage worker block duration must be between zero and one minute")
	}
	if options.ReclaimIdle == 0 {
		options.ReclaimIdle = 30 * time.Second
	}
	if options.ReclaimIdle <= 0 {
		return WorkerOptions{}, fmt.Errorf("usage worker reclaim idle time must be positive")
	}
	return options, nil
}

// Reconcile synchronously discovers active namespaces, creates their shared
// consumer groups, and performs one exact rollup pass. Managed startup calls
// this before exposing readiness; Run repeats it and then owns continuous work.
func (supervisor *Supervisor) Reconcile(ctx context.Context) error {
	if supervisor == nil {
		return fmt.Errorf("usage supervisor is unavailable")
	}
	supervisor.reconcileMu.Lock()
	defer supervisor.reconcileMu.Unlock()

	supervisor.mu.RLock()
	closed := supervisor.closed
	supervisor.mu.RUnlock()
	if closed {
		return fmt.Errorf("usage supervisor is closed")
	}
	if _, err := supervisor.storage.Reconcile(ctx); err != nil {
		supervisor.setReconcileError(err)
		return fmt.Errorf("maintain usage storage: %w", err)
	}
	namespaces, err := supervisor.namespaces.ListActiveNamespaces(ctx)
	if err != nil {
		supervisor.setReconcileError(err)
		return err
	}
	desired, err := validateNamespaceSnapshot(namespaces)
	if err != nil {
		supervisor.setReconcileError(err)
		return err
	}

	supervisor.mu.RLock()
	current := make(map[string]*namespaceWorker, len(supervisor.workers))
	for id, worker := range supervisor.workers {
		current[id] = worker
	}
	supervisor.mu.RUnlock()

	prepared := make(map[string]*namespaceWorker)
	for id, namespace := range desired {
		if existing, ok := current[id]; ok && existing.namespace.QuotaPartitionID == namespace.QuotaPartitionID {
			continue
		}
		worker, err := supervisor.prepareNamespace(ctx, namespace)
		if err != nil {
			supervisor.setReconcileError(err)
			return err
		}
		prepared[id] = worker
	}

	supervisor.mu.Lock()
	if supervisor.closed {
		supervisor.mu.Unlock()
		return fmt.Errorf("usage supervisor closed during reconciliation")
	}
	removed := make([]*namespaceWorker, 0)
	for id, worker := range supervisor.workers {
		desiredNamespace, exists := desired[id]
		if !exists || desiredNamespace.QuotaPartitionID != worker.namespace.QuotaPartitionID {
			delete(supervisor.workers, id)
			removed = append(removed, worker)
		}
	}
	for id, worker := range prepared {
		supervisor.workers[id] = worker
	}
	if supervisor.running {
		// Start while holding the lifecycle lock. Close takes the same lock
		// before it snapshots workers, so it can neither miss a just-published
		// worker nor return before that worker has observed cancellation.
		for _, worker := range prepared {
			worker.start(supervisor.runContext)
		}
	}
	supervisor.initialized = true
	supervisor.lastSuccess = time.Now().UTC()
	supervisor.lastErr = nil
	supervisor.mu.Unlock()

	for _, worker := range removed {
		worker.stop()
	}
	return nil
}

func validateNamespaceSnapshot(namespaces []ActiveNamespace) (map[string]ActiveNamespace, error) {
	desired := make(map[string]ActiveNamespace, len(namespaces))
	partitions := make(map[string]string, len(namespaces))
	for _, namespace := range namespaces {
		if err := requireUUID("active namespace ID", namespace.ID, false); err != nil {
			return nil, err
		}
		if !partitionPattern.MatchString(namespace.QuotaPartitionID) {
			return nil, fmt.Errorf("active namespace %q has a non-canonical quota partition", namespace.ID)
		}
		if _, exists := desired[namespace.ID]; exists {
			return nil, fmt.Errorf("active namespace %q is duplicated", namespace.ID)
		}
		if owner, exists := partitions[namespace.QuotaPartitionID]; exists {
			return nil, fmt.Errorf("active namespaces %q and %q share quota partition %q", owner, namespace.ID, namespace.QuotaPartitionID)
		}
		desired[namespace.ID] = namespace
		partitions[namespace.QuotaPartitionID] = namespace.ID
	}
	return desired, nil
}

func (supervisor *Supervisor) prepareNamespace(
	ctx context.Context,
	namespace ActiveNamespace,
) (*namespaceWorker, error) {
	consumer := supervisor.replicaID + "." + namespace.ID
	if err := validateConsumerName("consumer", consumer); err != nil {
		return nil, err
	}
	stream, err := supervisor.streams.OpenNamespaceStream(namespace, supervisor.consumerGroup, consumer)
	if err != nil {
		return nil, fmt.Errorf("open usage stream for namespace %q: %w", namespace.ID, err)
	}
	options, _ := normalizedWorkerOptions(supervisor.workerOptions)
	options.NamespaceID = namespace.ID
	options.AfterCommit = supervisor.afterCommit
	worker, err := NewWorker(stream, supervisor.store, options)
	if err != nil {
		return nil, err
	}
	if err := worker.Ensure(ctx); err != nil {
		return nil, fmt.Errorf("ensure usage stream for namespace %q: %w", namespace.ID, err)
	}
	if _, err := supervisor.rollups.ProcessDirty(ctx, namespace.ID); err != nil {
		return nil, fmt.Errorf("refresh usage rollups for namespace %q: %w", namespace.ID, err)
	}
	return &namespaceWorker{
		namespace: namespace, worker: worker, rollups: supervisor.rollups,
		rollupInterval: supervisor.rollupInterval,
		minBackoff:     supervisor.minBackoff, maxBackoff: supervisor.maxBackoff,
	}, nil
}

func (supervisor *Supervisor) Started() <-chan struct{} {
	if supervisor == nil || supervisor.started == nil {
		closed := make(chan struct{})
		close(closed)
		return closed
	}
	return supervisor.started
}

func (supervisor *Supervisor) Run(ctx context.Context) error {
	if supervisor == nil {
		return fmt.Errorf("usage supervisor is unavailable")
	}
	supervisor.mu.Lock()
	if supervisor.closed {
		supervisor.mu.Unlock()
		return fmt.Errorf("usage supervisor is closed")
	}
	if supervisor.runInvoked {
		supervisor.mu.Unlock()
		return fmt.Errorf("usage supervisor can only run once")
	}
	runContext, cancel := context.WithCancel(ctx)
	supervisor.runInvoked = true
	supervisor.running = true
	supervisor.runContext = runContext
	supervisor.runCancel = cancel
	supervisor.mu.Unlock()
	defer supervisor.finishRun()

	if err := supervisor.Reconcile(runContext); err != nil {
		supervisor.signalStarted()
		return err
	}
	supervisor.mu.Lock()
	for _, worker := range supervisor.workers {
		worker.start(runContext)
	}
	supervisor.mu.Unlock()
	supervisor.signalStarted()

	wait := supervisor.reconcileInterval
	backoff := supervisor.minBackoff
	for {
		if err := waitForUsageWork(runContext, wait); err != nil {
			return err
		}
		if err := supervisor.Reconcile(runContext); err != nil {
			wait = backoff
			backoff = nextUsageBackoff(backoff, supervisor.maxBackoff)
			continue
		}
		wait = supervisor.reconcileInterval
		backoff = supervisor.minBackoff
	}
}

func (supervisor *Supervisor) Ready(context.Context) error {
	if supervisor == nil {
		return fmt.Errorf("usage supervisor is unavailable")
	}
	supervisor.mu.RLock()
	if supervisor.closed {
		supervisor.mu.RUnlock()
		return fmt.Errorf("usage supervisor is closed")
	}
	if !supervisor.initialized {
		supervisor.mu.RUnlock()
		return fmt.Errorf("usage supervisor has not reconciled namespaces")
	}
	if !supervisor.running {
		supervisor.mu.RUnlock()
		return fmt.Errorf("usage supervisor has not started")
	}
	lastErr := supervisor.lastErr
	lastSuccess := supervisor.lastSuccess
	workers := make([]*namespaceWorker, 0, len(supervisor.workers))
	for _, worker := range supervisor.workers {
		workers = append(workers, worker)
	}
	maximumStaleness := 3*supervisor.reconcileInterval + supervisor.maxBackoff
	supervisor.mu.RUnlock()
	if lastErr != nil {
		return fmt.Errorf("usage namespace reconciliation is unhealthy: %w", lastErr)
	}
	if lastSuccess.IsZero() || time.Since(lastSuccess) > maximumStaleness {
		return fmt.Errorf("usage namespace reconciliation is stale")
	}
	sort.Slice(workers, func(i, j int) bool { return workers[i].namespace.ID < workers[j].namespace.ID })
	for _, worker := range workers {
		if err := worker.ready(); err != nil {
			return fmt.Errorf("usage namespace %q is unhealthy: %w", worker.namespace.ID, err)
		}
	}
	return nil
}

func (supervisor *Supervisor) Close() error {
	if supervisor == nil {
		return nil
	}
	supervisor.closeOnce.Do(func() {
		supervisor.mu.Lock()
		supervisor.closed = true
		cancel := supervisor.runCancel
		runInvoked := supervisor.runInvoked
		workers := make([]*namespaceWorker, 0, len(supervisor.workers))
		for _, worker := range supervisor.workers {
			workers = append(workers, worker)
		}
		supervisor.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		for _, worker := range workers {
			worker.stop()
		}
		if runInvoked {
			<-supervisor.runDone
		} else {
			supervisor.signalStarted()
			supervisor.runDoneOnce.Do(func() { close(supervisor.runDone) })
		}
	})
	return nil
}

func (supervisor *Supervisor) setReconcileError(err error) {
	supervisor.mu.Lock()
	supervisor.lastErr = err
	supervisor.mu.Unlock()
}

func (supervisor *Supervisor) signalStarted() {
	supervisor.startedOnce.Do(func() { close(supervisor.started) })
}

func (supervisor *Supervisor) finishRun() {
	supervisor.mu.Lock()
	cancel := supervisor.runCancel
	workers := make([]*namespaceWorker, 0, len(supervisor.workers))
	for _, worker := range supervisor.workers {
		workers = append(workers, worker)
	}
	supervisor.running = false
	supervisor.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	for _, worker := range workers {
		worker.stop()
	}
	supervisor.signalStarted()
	supervisor.runDoneOnce.Do(func() { close(supervisor.runDone) })
}

type namespaceWorker struct {
	namespace      ActiveNamespace
	worker         *Worker
	rollups        RollupProcessor
	rollupInterval time.Duration
	minBackoff     time.Duration
	maxBackoff     time.Duration

	mu        sync.RWMutex
	started   bool
	cancel    context.CancelFunc
	ingestErr error
	rollupErr error
	wait      sync.WaitGroup
}

func (worker *namespaceWorker) start(parent context.Context) {
	worker.mu.Lock()
	if worker.started {
		worker.mu.Unlock()
		return
	}
	ctx, cancel := context.WithCancel(parent)
	worker.started = true
	worker.cancel = cancel
	worker.wait.Add(2)
	worker.mu.Unlock()
	go worker.runIngestion(ctx)
	go worker.runRollups(ctx)
}

func (worker *namespaceWorker) stop() {
	worker.mu.RLock()
	cancel := worker.cancel
	worker.mu.RUnlock()
	if cancel != nil {
		cancel()
	}
	worker.wait.Wait()
}

func (worker *namespaceWorker) ready() error {
	worker.mu.RLock()
	defer worker.mu.RUnlock()
	if !worker.started {
		return fmt.Errorf("worker has not started")
	}
	return errors.Join(worker.ingestErr, worker.rollupErr)
}

func (worker *namespaceWorker) runIngestion(ctx context.Context) {
	defer worker.wait.Done()
	backoff := worker.minBackoff
	for {
		_, err := worker.worker.ProcessOnce(ctx)
		if err == nil {
			worker.setIngestError(nil)
			backoff = worker.minBackoff
			continue
		}
		if ctx.Err() != nil {
			return
		}
		worker.setIngestError(err)
		if isTerminalUsageWorkerError(err) {
			return
		}
		if waitForUsageWork(ctx, backoff) != nil {
			return
		}
		backoff = nextUsageBackoff(backoff, worker.maxBackoff)
	}
}

func (worker *namespaceWorker) runRollups(ctx context.Context) {
	defer worker.wait.Done()
	wait := time.Duration(0)
	backoff := worker.minBackoff
	for {
		if wait > 0 && waitForUsageWork(ctx, wait) != nil {
			return
		}
		result, err := worker.rollups.ProcessDirty(ctx, worker.namespace.ID)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			worker.setRollupError(err)
			wait = backoff
			backoff = nextUsageBackoff(backoff, worker.maxBackoff)
			continue
		}
		worker.setRollupError(nil)
		backoff = worker.minBackoff
		if result.More {
			wait = 0
		} else {
			wait = worker.rollupInterval
		}
	}
}

func (worker *namespaceWorker) setIngestError(err error) {
	worker.mu.Lock()
	worker.ingestErr = err
	worker.mu.Unlock()
}

func (worker *namespaceWorker) setRollupError(err error) {
	worker.mu.Lock()
	worker.rollupErr = err
	worker.mu.Unlock()
}

func isTerminalUsageWorkerError(err error) bool {
	return errors.Is(err, ErrPoisonedStreamItem) || errors.Is(err, ErrConflict) || errors.Is(err, ErrLedgerCorrupt)
}

func waitForUsageWork(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		return nil
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func nextUsageBackoff(current, maximum time.Duration) time.Duration {
	if current >= maximum {
		return maximum
	}
	next := current * 2
	if next > maximum {
		return maximum
	}
	return next
}
