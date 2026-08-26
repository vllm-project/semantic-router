// Package quotarecovery owns background settlement of admissions whose Router
// replica disappeared before terminal accounting. It is data-plane lifecycle:
// namespace discovery comes from durable state and does not depend on the
// Dashboard or Management API being present.
package quotarecovery

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

const (
	defaultPollInterval = 250 * time.Millisecond
	defaultBatchSize    = 64
	maximumBatchSize    = 1024
	minimumReadyWindow  = 10 * time.Second
)

type Runtime interface {
	RecoverOldestExpiredAdmission(context.Context, string) (quotaruntime.ExpiredRecoveryResult, error)
}

type SupervisorOptions struct {
	Namespaces   usageledger.NamespaceSource
	Runtime      Runtime
	PollInterval time.Duration
	BatchSize    int
}

type Diagnostics struct {
	Initialized       bool
	Running           bool
	LastSuccess       time.Time
	LastError         string
	RecoveredTotal    uint64
	IdempotentTotal   uint64
	UnknownTotal      uint64
	LastAdmissionID   string
	LastPartition     string
	LastEvidenceState string
}

// Supervisor performs bounded passes. Multiple replicas intentionally run the
// same pass: terminal CAS and evidence-revision CAS make recovery exactly once
// without a second lock/claim protocol that itself could be abandoned.
type Supervisor struct {
	namespaces   usageledger.NamespaceSource
	runtime      Runtime
	pollInterval time.Duration
	batchSize    int

	reconcileMu     sync.Mutex
	partitionOffset int
	mu              sync.RWMutex
	diagnostics     Diagnostics
	runInvoked      bool
	closed          bool
	cancel          context.CancelFunc
	started         chan struct{}
	startedOnce     sync.Once
	done            chan struct{}
	doneOnce        sync.Once
	closeOnce       sync.Once
}

func NewSupervisor(options SupervisorOptions) (*Supervisor, error) {
	if options.Namespaces == nil || options.Runtime == nil {
		return nil, fmt.Errorf("quota recovery namespaces and runtime are required")
	}
	if options.PollInterval == 0 {
		options.PollInterval = defaultPollInterval
	}
	if options.BatchSize == 0 {
		options.BatchSize = defaultBatchSize
	}
	if options.PollInterval < 10*time.Millisecond || options.PollInterval > time.Minute ||
		options.BatchSize < 1 || options.BatchSize > maximumBatchSize {
		return nil, fmt.Errorf("quota recovery interval or batch size is outside its bound")
	}
	return &Supervisor{
		namespaces: options.Namespaces, runtime: options.Runtime,
		pollInterval: options.PollInterval, batchSize: options.BatchSize,
		started: make(chan struct{}), done: make(chan struct{}),
	}, nil
}

// Reconcile performs one globally bounded pass. A busy partition cannot make
// one pass unbounded, and the next sorted pass revisits all active partitions.
func (supervisor *Supervisor) Reconcile(ctx context.Context) error {
	if supervisor == nil {
		return fmt.Errorf("quota recovery supervisor is unavailable")
	}
	supervisor.reconcileMu.Lock()
	defer supervisor.reconcileMu.Unlock()
	supervisor.mu.RLock()
	closed := supervisor.closed
	supervisor.mu.RUnlock()
	if closed {
		return fmt.Errorf("quota recovery supervisor is closed")
	}
	namespaces, err := supervisor.namespaces.ListActiveNamespaces(ctx)
	if err != nil {
		supervisor.recordError(err)
		return fmt.Errorf("discover quota recovery namespaces: %w", err)
	}
	partitions, err := activePartitions(namespaces)
	if err != nil {
		supervisor.recordError(err)
		return err
	}
	partitions = supervisor.nextPartitionBatch(partitions)
	active := append([]string(nil), partitions...)
	remaining := supervisor.batchSize
	for remaining > 0 && len(active) > 0 {
		next := make([]string, 0, len(active))
		for index, partition := range active {
			if remaining == 0 {
				next = append(next, active[index:]...)
				break
			}
			remaining--
			result, recoverErr := supervisor.runtime.RecoverOldestExpiredAdmission(ctx, partition)
			if recoverErr != nil {
				supervisor.recordError(recoverErr)
				logging.ComponentWarnEvent("quota_recovery", "expired_admission_recovery_failed", map[string]interface{}{
					"partition": partition, "error_class": fmt.Sprintf("%T", recoverErr),
				})
				return fmt.Errorf("recover expired admission in partition %q: %w", partition, recoverErr)
			}
			if result.Recovered {
				supervisor.recordResult(partition, result)
			}
			if result.Retry {
				next = append(next, partition)
			}
		}
		active = next
	}
	supervisor.mu.Lock()
	supervisor.diagnostics.Initialized = true
	supervisor.diagnostics.LastSuccess = time.Now().UTC()
	supervisor.diagnostics.LastError = ""
	supervisor.mu.Unlock()
	return nil
}

func (supervisor *Supervisor) nextPartitionBatch(partitions []string) []string {
	if len(partitions) == 0 {
		supervisor.partitionOffset = 0
		return nil
	}
	limit := supervisor.batchSize
	if limit > len(partitions) {
		limit = len(partitions)
	}
	start := supervisor.partitionOffset % len(partitions)
	result := make([]string, 0, limit)
	for index := 0; index < limit; index++ {
		result = append(result, partitions[(start+index)%len(partitions)])
	}
	supervisor.partitionOffset = (start + limit) % len(partitions)
	return result
}

func activePartitions(namespaces []usageledger.ActiveNamespace) ([]string, error) {
	seenNamespaces := make(map[string]struct{}, len(namespaces))
	seenPartitions := make(map[string]struct{}, len(namespaces))
	partitions := make([]string, 0, len(namespaces))
	for _, namespace := range namespaces {
		if namespace.ID == "" || namespace.QuotaPartitionID == "" {
			return nil, fmt.Errorf("quota recovery namespace identity is incomplete")
		}
		if _, duplicate := seenNamespaces[namespace.ID]; duplicate {
			return nil, fmt.Errorf("quota recovery namespace %q is duplicated", namespace.ID)
		}
		seenNamespaces[namespace.ID] = struct{}{}
		if _, duplicate := seenPartitions[namespace.QuotaPartitionID]; duplicate {
			continue
		}
		seenPartitions[namespace.QuotaPartitionID] = struct{}{}
		partitions = append(partitions, namespace.QuotaPartitionID)
	}
	sort.Strings(partitions)
	return partitions, nil
}

func (supervisor *Supervisor) Run(ctx context.Context) error {
	if supervisor == nil {
		return fmt.Errorf("quota recovery supervisor is unavailable")
	}
	supervisor.mu.Lock()
	if supervisor.closed {
		supervisor.mu.Unlock()
		return fmt.Errorf("quota recovery supervisor is closed")
	}
	if supervisor.runInvoked {
		supervisor.mu.Unlock()
		return fmt.Errorf("quota recovery supervisor can only run once")
	}
	runContext, cancel := context.WithCancel(ctx)
	supervisor.runInvoked = true
	supervisor.cancel = cancel
	supervisor.diagnostics.Running = true
	supervisor.mu.Unlock()
	supervisor.startedOnce.Do(func() { close(supervisor.started) })
	defer func() {
		supervisor.mu.Lock()
		supervisor.diagnostics.Running = false
		supervisor.mu.Unlock()
		supervisor.doneOnce.Do(func() { close(supervisor.done) })
	}()

	timer := time.NewTimer(0)
	defer timer.Stop()
	for {
		select {
		case <-runContext.Done():
			return runContext.Err()
		case <-timer.C:
			_ = supervisor.Reconcile(runContext)
			timer.Reset(supervisor.pollInterval)
		}
	}
}

func (supervisor *Supervisor) Started() <-chan struct{} {
	if supervisor == nil {
		closed := make(chan struct{})
		close(closed)
		return closed
	}
	return supervisor.started
}

func (supervisor *Supervisor) Ready(context.Context) error {
	if supervisor == nil {
		return fmt.Errorf("quota recovery supervisor is unavailable")
	}
	supervisor.mu.RLock()
	diagnostics, closed := supervisor.diagnostics, supervisor.closed
	supervisor.mu.RUnlock()
	if closed {
		return fmt.Errorf("quota recovery supervisor is closed")
	}
	if !diagnostics.Initialized || !diagnostics.Running {
		return fmt.Errorf("quota recovery supervisor has not started and reconciled")
	}
	if diagnostics.LastError != "" {
		return fmt.Errorf("quota recovery is unhealthy: %s", diagnostics.LastError)
	}
	readyWindow := 4 * supervisor.pollInterval
	if readyWindow < minimumReadyWindow {
		readyWindow = minimumReadyWindow
	}
	if diagnostics.LastSuccess.IsZero() || time.Since(diagnostics.LastSuccess) > readyWindow {
		return fmt.Errorf("quota recovery is stale")
	}
	return nil
}

func (supervisor *Supervisor) Diagnostics() Diagnostics {
	if supervisor == nil {
		return Diagnostics{}
	}
	supervisor.mu.RLock()
	defer supervisor.mu.RUnlock()
	return supervisor.diagnostics
}

func (supervisor *Supervisor) Close() error {
	if supervisor == nil {
		return nil
	}
	supervisor.closeOnce.Do(func() {
		supervisor.mu.Lock()
		supervisor.closed = true
		cancel, runInvoked := supervisor.cancel, supervisor.runInvoked
		supervisor.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		if runInvoked {
			<-supervisor.done
		} else {
			supervisor.startedOnce.Do(func() { close(supervisor.started) })
			supervisor.doneOnce.Do(func() { close(supervisor.done) })
		}
	})
	return nil
}

func (supervisor *Supervisor) recordError(err error) {
	if err == nil || errors.Is(err, context.Canceled) {
		return
	}
	supervisor.mu.Lock()
	supervisor.diagnostics.LastError = err.Error()
	supervisor.mu.Unlock()
}

func (supervisor *Supervisor) recordResult(partition string, result quotaruntime.ExpiredRecoveryResult) {
	supervisor.mu.Lock()
	supervisor.diagnostics.LastPartition = partition
	supervisor.diagnostics.LastAdmissionID = result.AdmissionID
	supervisor.diagnostics.LastEvidenceState = result.EvidenceState
	if result.Recovered {
		supervisor.diagnostics.RecoveredTotal++
	}
	if result.Idempotent {
		supervisor.diagnostics.IdempotentTotal++
	}
	if result.EvidenceState == "unknown" || result.EvidenceState == "mixed" {
		supervisor.diagnostics.UnknownTotal++
	}
	supervisor.mu.Unlock()
	logging.ComponentEvent("quota_recovery", "expired_admission_recovered", map[string]interface{}{
		"partition": partition, "admission_id": result.AdmissionID,
		"evidence_state": result.EvidenceState, "idempotent": result.Idempotent,
	})
}
