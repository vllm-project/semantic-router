package accesspublisher

import (
	"context"
	"errors"
	"fmt"
	"time"
)

type EngineOptions struct {
	Outbox          OutboxStore
	Desired         DesiredStateReader
	Runtime         RuntimeStore
	WorkerID        string
	ClaimLease      time.Duration
	RetryDelay      time.Duration
	CompactionBatch int
}

// Engine is a deliberately small orchestration layer. Correct transition
// ordering lives in State, PostgreSQL owns desired/applied durability, and
// the selected runtime store owns publication CAS. ProcessOnce is safe for repeated workers;
// namespace ownership is acquired from the transactional outbox.
type Engine struct {
	outbox          OutboxStore
	desired         DesiredStateReader
	runtime         RuntimeStore
	workerID        string
	claimLease      time.Duration
	retryDelay      time.Duration
	compactionBatch int
}

type ProcessDisposition string

const (
	ProcessApplied        ProcessDisposition = "applied"
	ProcessWaitingForAcks ProcessDisposition = "waiting_for_acknowledgements"
	ProcessSuperseded     ProcessDisposition = "superseded"
	ProcessNoWork         ProcessDisposition = "no_work"
)

type ProcessResult struct {
	Disposition            ProcessDisposition
	NamespaceID            string
	Revision               uint64
	PublicationID          string
	MissingBarrierReplicas []string
	MissingRoutingReplicas []string
}

func NewEngine(options EngineOptions) (*Engine, error) {
	if options.Outbox == nil || options.Desired == nil || options.Runtime == nil {
		return nil, fmt.Errorf("outbox, desired-state reader, and runtime store are required")
	}
	if err := validateWorker(options.WorkerID); err != nil {
		return nil, err
	}
	if options.ClaimLease == 0 {
		options.ClaimLease = 30 * time.Second
	}
	if options.ClaimLease < time.Second || options.ClaimLease > time.Hour {
		return nil, fmt.Errorf("claim lease must be between one second and one hour")
	}
	if options.RetryDelay == 0 {
		options.RetryDelay = time.Second
	}
	if options.RetryDelay < 0 || options.RetryDelay > time.Hour {
		return nil, fmt.Errorf("retry delay is invalid")
	}
	if options.CompactionBatch == 0 {
		options.CompactionBatch = 500
	}
	if options.CompactionBatch < 1 || options.CompactionBatch > 1000 {
		return nil, fmt.Errorf("compaction batch must be between 1 and 1000")
	}
	return &Engine{
		outbox: options.Outbox, desired: options.Desired, runtime: options.Runtime,
		workerID: options.WorkerID, claimLease: options.ClaimLease,
		retryDelay: options.RetryDelay, compactionBatch: options.CompactionBatch,
	}, nil
}

func (e *Engine) ProcessOnce(ctx context.Context) (ProcessResult, error) {
	batch, processOnceErr := e.outbox.ClaimLatest(ctx, e.workerID, e.claimLease)
	if errors.Is(processOnceErr, ErrNoWork) {
		return ProcessResult{Disposition: ProcessNoWork}, nil
	}
	if processOnceErr != nil {
		return ProcessResult{}, processOnceErr
	}
	result := ProcessResult{NamespaceID: batch.NamespaceID, Revision: batch.DesiredRevision}

	state, processOnceErr := e.desired.LoadDesiredState(ctx, batch.NamespaceID, batch.DesiredRevision)
	if processOnceErr != nil {
		return result, e.handleBuildFailure(ctx, batch, processOnceErr)
	}
	publication, processOnceErr := Compile(state)
	if processOnceErr != nil {
		return result, e.handleBuildFailure(ctx, batch, processOnceErr)
	}
	result.PublicationID = publication.ID
	if err := validateBatchPublication(batch, publication); err != nil {
		return result, e.handleBuildFailure(ctx, batch, err)
	}
	if err := e.outbox.RecordStaged(ctx, batch, publication); err != nil {
		return result, e.handleBuildFailure(ctx, batch, err)
	}
	plan, processOnceErr := e.runtime.Prepare(ctx, publication)
	if processOnceErr != nil {
		return result, e.handleRuntimeFailure(ctx, batch, processOnceErr)
	}
	if plan.Restrictive() {
		if err := e.runtime.InstallBarriers(ctx, plan); err != nil {
			return result, e.handleRuntimeFailure(ctx, batch, err)
		}
	}
	if err := e.runtime.Stage(ctx, plan); err != nil {
		return result, e.handleRuntimeFailure(ctx, batch, err)
	}
	if err := e.runtime.ValidateStaged(ctx, plan); err != nil {
		// Validation failures are terminal and deliberately leave already
		// installed barriers in place. A later repaired desired revision can
		// supersede and clear them only after it is fully applied.
		return result, e.fail(ctx, batch, err)
	}
	barrierAck := AckStatus{}
	if plan.Restrictive() {
		barrierAck, processOnceErr = e.runtime.BarrierAcknowledgements(ctx, plan)
		if processOnceErr != nil {
			return result, e.handleRuntimeFailure(ctx, batch, processOnceErr)
		}
	}
	routingAck, processOnceErr := e.runtime.RoutingAcknowledgements(ctx, plan)
	if processOnceErr != nil {
		return result, e.handleRuntimeFailure(ctx, batch, processOnceErr)
	}
	if !barrierAck.Complete() || !routingAck.Complete() {
		result.Disposition = ProcessWaitingForAcks
		result.MissingBarrierReplicas = append([]string(nil), barrierAck.Missing...)
		result.MissingRoutingReplicas = append([]string(nil), routingAck.Missing...)
		if err := e.outbox.Release(ctx, batch, ErrAcknowledgements, e.retryDelay); err != nil {
			return result, err
		}
		return result, nil
	}

	processOnceErr = e.outbox.WithRevisionFence(ctx, batch, func(fenced context.Context) error {
		return e.activateFenced(fenced, plan)
	})
	if processOnceErr != nil {
		if errors.Is(processOnceErr, ErrSuperseded) {
			result.Disposition = ProcessSuperseded
			if releaseErr := e.outbox.Release(ctx, batch, processOnceErr, 0); releaseErr != nil {
				return result, releaseErr
			}
			return result, nil
		}
		return result, e.handleRuntimeFailure(ctx, batch, processOnceErr)
	}
	if err := e.runtime.MarkApplied(ctx, plan); err != nil {
		// PostgreSQL is already applied. Do not fail or release the applied outbox;
		// a retry/reconciler must finish this idempotent Redis watermark step.
		return result, fmt.Errorf("mark applied runtime publication: %w", err)
	}
	if err := e.runtime.ClearAppliedBarriers(ctx, plan); err != nil {
		return result, fmt.Errorf("clear applied publication barriers: %w", err)
	}
	result.Disposition = ProcessApplied
	return result, nil
}

func (e *Engine) activateFenced(ctx context.Context, plan PublicationPlan) error {
	// Recheck acknowledgements inside the PostgreSQL namespace fence. Redis
	// activation also performs the same check atomically against live leases,
	// closing the replica-join race.
	if plan.Restrictive() {
		status, err := e.runtime.BarrierAcknowledgements(ctx, plan)
		if err != nil {
			return err
		}
		if !status.Complete() {
			return ErrAcknowledgements
		}
	}
	status, err := e.runtime.RoutingAcknowledgements(ctx, plan)
	if err != nil {
		return err
	}
	if !status.Complete() {
		return ErrAcknowledgements
	}
	if err := e.runtime.Activate(ctx, plan); err != nil {
		return err
	}
	for {
		complete, err := e.runtime.Compact(ctx, plan, e.compactionBatch)
		if err != nil {
			return err
		}
		if complete {
			return nil
		}
	}
}

// ReconcileApplied completes idempotent runtime-store finalization after a process
// loss between the PostgreSQL applied commit and MarkApplied/ClearBarriers.
// It never advances PostgreSQL and therefore cannot publish a desired revision.
func (e *Engine) ReconcileApplied(ctx context.Context, namespaceID string) (ProcessResult, error) {
	applied, err := e.outbox.Applied(ctx, namespaceID)
	if err != nil {
		return ProcessResult{}, err
	}
	result := ProcessResult{
		NamespaceID: namespaceID, Revision: applied.DesiredRevision,
		Disposition: ProcessApplied,
	}
	if applied.DesiredRevision == 0 {
		return ProcessResult{}, ErrNoWork
	}
	if err := e.runtime.ReconcileApplied(ctx, applied); err != nil {
		return result, err
	}
	return result, nil
}

func (e *Engine) handleBuildFailure(ctx context.Context, batch OutboxBatch, cause error) error {
	if errors.Is(cause, ErrSuperseded) {
		if err := e.outbox.Release(ctx, batch, cause, 0); err != nil {
			return err
		}
		return cause
	}
	return e.fail(ctx, batch, cause)
}

func (e *Engine) handleRuntimeFailure(ctx context.Context, batch OutboxBatch, cause error) error {
	if errors.Is(cause, ErrStagedCorrupt) {
		return e.fail(ctx, batch, cause)
	}
	if err := e.outbox.Release(ctx, batch, cause, e.retryDelay); err != nil {
		return fmt.Errorf("publication failed (%w) and outbox release failed: %w", cause, err)
	}
	return cause
}

func (e *Engine) fail(ctx context.Context, batch OutboxBatch, cause error) error {
	if err := e.outbox.Fail(ctx, batch, cause); err != nil {
		return fmt.Errorf("publication failed (%w) and outbox failure persistence failed: %w", cause, err)
	}
	return cause
}
