package agentruntime

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type WorkerOptions struct {
	Store         agentmanagement.Store
	Authority     agentmanagement.SessionAuthority
	Registries    agentmanagement.RegistrySource
	Inference     PublicInferenceClient
	Notifier      agentmanagement.TurnNotifier
	LiveEvents    agentmanagement.LiveEventPublisher
	WorkerID      string
	Concurrency   int
	PollInterval  time.Duration
	LeaseDuration time.Duration
	RenewInterval time.Duration
	DelegationTTL time.Duration
	Now           func() time.Time
}

// Worker is a PostgreSQL-fenced Agent turn consumer. PostgreSQL is the sole
// durable queue and lease authority; notifications only reduce pickup latency.
type Worker struct {
	store         agentmanagement.Store
	authority     agentmanagement.SessionAuthority
	registries    agentmanagement.RegistrySource
	inference     PublicInferenceClient
	notifier      agentmanagement.TurnNotifier
	liveEvents    agentmanagement.LiveEventPublisher
	workerID      string
	concurrency   int
	pollInterval  time.Duration
	leaseDuration time.Duration
	renewInterval time.Duration
	delegationTTL time.Duration
	now           func() time.Time
}

func NewWorker(options WorkerOptions) (*Worker, error) {
	if options.Store == nil || options.Authority == nil || options.Registries == nil ||
		options.Inference == nil || !validWorkerID(options.WorkerID) {
		return nil, errors.New("agent worker dependencies are incomplete")
	}
	if options.Concurrency == 0 {
		options.Concurrency = 2
	}
	if options.PollInterval == 0 {
		options.PollInterval = 250 * time.Millisecond
	}
	if options.LeaseDuration == 0 {
		options.LeaseDuration = 30 * time.Second
	}
	if options.RenewInterval == 0 {
		options.RenewInterval = 10 * time.Second
	}
	if options.DelegationTTL == 0 {
		options.DelegationTTL = 5 * time.Minute
	}
	if options.Concurrency < 1 || options.Concurrency > 128 ||
		options.PollInterval < 25*time.Millisecond || options.PollInterval > time.Minute ||
		options.LeaseDuration < 5*time.Second || options.LeaseDuration > 5*time.Minute ||
		options.RenewInterval < time.Second || options.RenewInterval*2 >= options.LeaseDuration ||
		options.DelegationTTL < time.Minute || options.DelegationTTL > time.Hour {
		return nil, errors.New("agent worker timing or concurrency is invalid")
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	return &Worker{
		store: options.Store, authority: options.Authority, registries: options.Registries,
		inference: options.Inference, notifier: options.Notifier, liveEvents: options.LiveEvents,
		workerID:    options.WorkerID,
		concurrency: options.Concurrency, pollInterval: options.PollInterval,
		leaseDuration: options.LeaseDuration, renewInterval: options.RenewInterval,
		delegationTTL: options.DelegationTTL, now: options.Now,
	}, nil
}

func (worker *Worker) Ready(ctx context.Context) error {
	if worker == nil || worker.store == nil || worker.registries == nil || worker.inference == nil {
		return errors.New("agent worker is unavailable")
	}
	return worker.store.Ready(ctx)
}

func (worker *Worker) Run(ctx context.Context) error {
	if err := worker.Ready(ctx); err != nil {
		return err
	}
	workerContext, cancel := context.WithCancel(ctx)
	defer cancel()
	errorsByWorker := make(chan error, worker.concurrency)
	var active sync.WaitGroup
	for index := 0; index < worker.concurrency; index++ {
		active.Add(1)
		go func(ordinal int) {
			defer active.Done()
			errorsByWorker <- worker.runClaimLoop(workerContext, fmt.Sprintf("%s/%d", worker.workerID, ordinal))
		}(index)
	}
	go func() {
		active.Wait()
		close(errorsByWorker)
	}()
	var failures []error
	for err := range errorsByWorker {
		if err != nil && !errors.Is(err, context.Canceled) {
			failures = append(failures, err)
			cancel()
		}
	}
	if len(failures) > 0 {
		return errors.Join(failures...)
	}
	return ctx.Err()
}

func (worker *Worker) runClaimLoop(ctx context.Context, workerID string) error {
	ticker := time.NewTicker(worker.pollInterval)
	defer ticker.Stop()
	for {
		lease, err := worker.store.ClaimNextTurn(ctx, workerID, worker.now().UTC().Add(worker.leaseDuration))
		switch {
		case err == nil:
			worker.processLease(ctx, lease)
		case errors.Is(err, agentmanagement.ErrNotFound), errors.Is(err, agentmanagement.ErrConflict):
			// A conflict is expected queue contention between workers, not a
			// process-level failure. Back off exactly like an empty queue so one
			// claimant cannot cancel an unrelated turn owned by another loop.
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-ticker.C:
			}
		default:
			return fmt.Errorf("claim Agent turn: %w", err)
		}
	}
}

func (worker *Worker) processLease(parent context.Context, lease agentmanagement.TurnLease) {
	leaseContext, cancel := context.WithCancel(parent)
	defer cancel()
	leaseFailures := make(chan error, 1)
	go worker.maintainLease(leaseContext, cancel, lease, leaseFailures)

	transition, err := worker.executeTurn(leaseContext, lease)
	cancel()
	select {
	case leaseErr := <-leaseFailures:
		if leaseErr != nil && !errors.Is(leaseErr, context.Canceled) {
			return
		}
	default:
	}
	if err != nil {
		cancelled, cancelErr := worker.store.CancellationRequested(context.WithoutCancel(parent), lease)
		if cancelErr == nil && cancelled {
			err = agentmanagement.ErrCancelled
		}
	}
	if err != nil {
		status := agentmanagement.TurnFailed
		failure := safeWorkerFailure(err)
		if errors.Is(err, agentmanagement.ErrCancelled) {
			status, failure = agentmanagement.TurnCancelled, nil
		} else {
			diagnostic := safeWorkerFailureDiagnostic(err)
			fields := map[string]interface{}{
				"session_id": lease.SessionID, "turn_id": lease.TurnID,
				"failure_code": failure.Code, "failure_class": diagnostic.class,
			}
			if diagnostic.upstreamStatus != 0 {
				fields["upstream_status"] = diagnostic.upstreamStatus
			}
			if diagnostic.modelStepStage != "" {
				fields["model_step_stage"] = diagnostic.modelStepStage
			}
			if diagnostic.protocolCategory != "" {
				fields["protocol_category"] = diagnostic.protocolCategory
			}
			if diagnostic.protocolCode != "" {
				fields["protocol_code"] = diagnostic.protocolCode
			}
			logging.ComponentErrorEvent("agent-runtime", "agent_turn_failed", fields)
		}
		transition = agentmanagement.TurnTransition{
			Lease: lease, Status: status, Failure: failure, CompletedAt: worker.now().UTC(),
		}
	}
	event, transitionErr := worker.store.TransitionTurn(context.WithoutCancel(parent), transition)
	if transitionErr == nil {
		worker.notifyEvent(context.WithoutCancel(parent), lease, event)
	}
}

func (worker *Worker) maintainLease(
	ctx context.Context, cancel context.CancelFunc, lease agentmanagement.TurnLease, result chan<- error,
) {
	ticker := time.NewTicker(worker.renewInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			result <- ctx.Err()
			return
		case <-ticker.C:
			cancelled, err := worker.store.CancellationRequested(ctx, lease)
			if err != nil {
				result <- err
				cancel()
				return
			}
			if cancelled {
				result <- agentmanagement.ErrCancelled
				cancel()
				return
			}
			lease, err = worker.store.RenewTurn(ctx, lease, worker.now().UTC().Add(worker.leaseDuration))
			if err != nil {
				result <- err
				cancel()
				return
			}
		}
	}
}

func (worker *Worker) notifyEvent(ctx context.Context, lease agentmanagement.TurnLease, event agentmanagement.Event) {
	if worker.notifier != nil && event.Sequence > 0 {
		_ = worker.notifier.NotifyEvents(ctx, lease.NamespaceID, lease.SessionID, event.Sequence)
	}
}

func validWorkerID(value string) bool {
	if value == "" || len(value) > 96 || value != strings.TrimSpace(value) {
		return false
	}
	for _, character := range value {
		if character < 0x21 || character == 0x7f {
			return false
		}
	}
	return true
}

func safeWorkerFailure(err error) *agentmanagement.Failure {
	switch {
	case errors.Is(err, agentmanagement.ErrDenied):
		return &agentmanagement.Failure{Code: "authorization_changed", Message: "Access changed before the Agent could continue.", Retryable: false}
	case errors.Is(err, agentmanagement.ErrToolUnavailable):
		return &agentmanagement.Failure{Code: "tool_unavailable", Message: "A required tool is no longer available.", Retryable: true}
	case errors.Is(err, agentmanagement.ErrConflict):
		return &agentmanagement.Failure{Code: "revision_conflict", Message: "A required resource changed. Refresh and try again.", Retryable: true}
	case errors.Is(err, context.DeadlineExceeded):
		return &agentmanagement.Failure{Code: "turn_timeout", Message: "The Agent turn reached its time limit.", Retryable: true}
	default:
		return &agentmanagement.Failure{Code: "agent_failed", Message: "The Agent could not complete this turn.", Retryable: true}
	}
}
