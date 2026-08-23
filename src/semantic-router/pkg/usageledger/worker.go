package usageledger

import (
	"context"
	"errors"
	"fmt"
	"time"
)

var ErrPoisonedStreamItem = errors.New("poisoned usage stream item")

// WorkerOptions configures one namespace-local stream consumer.
type WorkerOptions struct {
	NamespaceID string
	BatchSize   int64
	Block       time.Duration
	ReclaimIdle time.Duration
	AfterCommit CommittedBatchHook
}

// CommittedBatchHook completes durable projections derived from a committed
// ledger batch before its stream items are acknowledged. A hook failure leaves
// the items pending, so another replica can replay the idempotent transaction
// and finish the projection.
type CommittedBatchHook interface {
	AfterCommit(context.Context, []TerminalEvent) error
}

// Worker projects terminal stream items into the immutable ledger and any
// required committed-batch projections before acknowledgement.
type Worker struct {
	stream      Stream
	store       Store
	batchSize   int64
	block       time.Duration
	reclaimIdle time.Duration
	namespaceID string
	afterCommit CommittedBatchHook
}

func NewWorker(stream Stream, store Store, options WorkerOptions) (*Worker, error) {
	if stream == nil || store == nil {
		return nil, fmt.Errorf("usage worker stream and store are required")
	}
	if err := requireUUID("usage worker namespace ID", options.NamespaceID, false); err != nil {
		return nil, err
	}
	if options.BatchSize == 0 {
		options.BatchSize = 200
	}
	if options.BatchSize < 1 || options.BatchSize > 1000 {
		return nil, fmt.Errorf("usage worker batch size must be between 1 and 1000")
	}
	if options.Block == 0 {
		options.Block = time.Second
	}
	if options.Block < 0 || options.Block > time.Minute {
		return nil, fmt.Errorf("usage worker block duration must be between zero and one minute")
	}
	if options.ReclaimIdle == 0 {
		options.ReclaimIdle = 30 * time.Second
	}
	if options.ReclaimIdle <= 0 {
		return nil, fmt.Errorf("usage worker reclaim idle time must be positive")
	}
	return &Worker{
		stream: stream, store: store, namespaceID: options.NamespaceID,
		batchSize: options.BatchSize, block: options.Block, reclaimIdle: options.ReclaimIdle,
		afterCommit: options.AfterCommit,
	}, nil
}

func (w *Worker) Ensure(ctx context.Context) error {
	return w.stream.EnsureGroup(ctx)
}

// ProcessOnce reclaims a stale pending batch before reading new work. It
// acknowledges nothing unless the complete PostgreSQL batch committed.
func (w *Worker) ProcessOnce(ctx context.Context) (BatchResult, error) {
	items, processOnceErr := w.stream.ClaimStale(ctx, w.batchSize, w.reclaimIdle)
	if processOnceErr != nil {
		return BatchResult{}, processOnceErr
	}
	if len(items) == 0 {
		items, processOnceErr = w.stream.ReadNew(ctx, w.batchSize, w.block)
		if processOnceErr != nil {
			return BatchResult{}, processOnceErr
		}
	}
	if len(items) == 0 {
		return BatchResult{}, nil
	}
	events := make([]TerminalEvent, 0, len(items))
	ids := make([]string, 0, len(items))
	for _, item := range items {
		event, err := decodeStreamItem(item)
		if err != nil {
			return BatchResult{}, err
		}
		if event.NamespaceID != w.namespaceID {
			return BatchResult{}, fmt.Errorf("%w: item %q belongs to another namespace", ErrPoisonedStreamItem, item.ID)
		}
		events = append(events, event)
		ids = append(ids, item.ID)
	}
	result, processOnceErr := w.store.PersistBatch(ctx, events)
	if processOnceErr != nil {
		return BatchResult{}, processOnceErr
	}
	if w.afterCommit != nil {
		if err := w.afterCommit.AfterCommit(ctx, result.projectionEvents); err != nil {
			// The ledger commit remains authoritative and the stream items stay
			// pending. Reclaim replays the digest-idempotent store path before
			// retrying this projection.
			return result, fmt.Errorf("project committed usage batch: %w", err)
		}
	}
	if err := w.stream.Ack(ctx, ids); err != nil {
		// The database commit remains authoritative. Redelivery will take the
		// idempotent settlement path and retry only the acknowledgement.
		return result, err
	}
	return result, nil
}

func (w *Worker) Run(ctx context.Context) error {
	if err := w.Ensure(ctx); err != nil {
		return err
	}
	backoff := 100 * time.Millisecond
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		if _, err := w.ProcessOnce(ctx); err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return err
			}
			if errors.Is(err, ErrPoisonedStreamItem) || errors.Is(err, ErrConflict) || errors.Is(err, ErrLedgerCorrupt) {
				return err
			}
			timer := time.NewTimer(backoff)
			select {
			case <-ctx.Done():
				timer.Stop()
				return ctx.Err()
			case <-timer.C:
			}
			if backoff < 5*time.Second {
				backoff *= 2
				if backoff > 5*time.Second {
					backoff = 5 * time.Second
				}
			}
			continue
		}
		backoff = 100 * time.Millisecond
	}
}

func decodeStreamItem(item StreamItem) (TerminalEvent, error) {
	required := []string{"admission_id", "admission_digest", "finalization_digest", "evidence_state", "event"}
	if len(item.Values) != len(required) {
		return TerminalEvent{}, fmt.Errorf("%w: item %q has an unexpected field set", ErrPoisonedStreamItem, item.ID)
	}
	for _, field := range required {
		if item.Values[field] == "" {
			return TerminalEvent{}, fmt.Errorf("%w: item %q is missing %q", ErrPoisonedStreamItem, item.ID, field)
		}
	}
	event, err := DecodeTerminalEvent(item.Values["event"])
	if err != nil {
		return TerminalEvent{}, fmt.Errorf("%w: item %q: %w", ErrPoisonedStreamItem, item.ID, err)
	}
	if event.AdmissionID != item.Values["admission_id"] ||
		event.FinalizationDigest != item.Values["finalization_digest"] ||
		string(event.EvidenceState) != item.Values["evidence_state"] {
		return TerminalEvent{}, fmt.Errorf("%w: item %q envelope does not match its terminal event", ErrPoisonedStreamItem, item.ID)
	}
	if !isHexDigest(item.Values["admission_digest"]) {
		return TerminalEvent{}, fmt.Errorf("%w: item %q has an invalid admission digest", ErrPoisonedStreamItem, item.ID)
	}
	return event, nil
}
