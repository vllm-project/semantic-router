package usageledger

import (
	"context"
	"errors"
	"fmt"
	"time"
)

var ErrPoisonedStreamItem = errors.New("poisoned usage stream item")

type poisonedStreamItemError struct {
	itemID string
	reason string
	cause  error
}

func (err *poisonedStreamItemError) Error() string {
	if err.cause == nil {
		return fmt.Sprintf("%v: item %q (%s)", ErrPoisonedStreamItem, err.itemID, err.reason)
	}
	return fmt.Sprintf("%v: item %q (%s): %v", ErrPoisonedStreamItem, err.itemID, err.reason, err.cause)
}

func (*poisonedStreamItemError) Unwrap() error { return ErrPoisonedStreamItem }

func poisonedStreamItem(itemID, reason string, cause error) error {
	return &poisonedStreamItemError{itemID: itemID, reason: reason, cause: cause}
}

func streamItemPoisonReason(err error) (string, bool) {
	var poisoned *poisonedStreamItemError
	if !errors.As(err, &poisoned) {
		return "", false
	}
	return poisoned.reason, true
}

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

func (w *Worker) Quarantined(ctx context.Context) (int64, error) {
	return w.stream.Quarantined(ctx)
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
	result := BatchResult{}
	events := make([]TerminalEvent, 0, len(items))
	ids := make([]string, 0, len(items))
	for _, item := range items {
		event, err := decodeStreamItem(item)
		if err != nil {
			reason, poisoned := streamItemPoisonReason(err)
			if !poisoned {
				return result, err
			}
			moved, quarantineErr := w.stream.Quarantine(ctx, item, reason)
			if quarantineErr != nil {
				return result, fmt.Errorf("retain poisoned usage stream item: %w", quarantineErr)
			}
			if moved {
				result.Quarantined++
			}
			continue
		}
		if event.NamespaceID != w.namespaceID {
			err := poisonedStreamItem(item.ID, "namespace_mismatch", nil)
			moved, quarantineErr := w.stream.Quarantine(ctx, item, "namespace_mismatch")
			if quarantineErr != nil {
				return result, fmt.Errorf("retain %v: %w", err, quarantineErr)
			}
			if moved {
				result.Quarantined++
			}
			continue
		}
		events = append(events, event)
		ids = append(ids, item.ID)
	}
	if len(events) == 0 {
		return result, nil
	}
	persisted, processOnceErr := w.store.PersistBatch(ctx, events)
	if processOnceErr != nil {
		return result, processOnceErr
	}
	persisted.Quarantined += result.Quarantined
	result = persisted
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
		return TerminalEvent{}, poisonedStreamItem(item.ID, "field_set_invalid", nil)
	}
	for _, field := range required {
		if item.Values[field] == "" {
			return TerminalEvent{}, poisonedStreamItem(item.ID, "required_field_missing", nil)
		}
	}
	event, err := DecodeTerminalEvent(item.Values["event"])
	if err != nil {
		return TerminalEvent{}, poisonedStreamItem(item.ID, "terminal_event_invalid", err)
	}
	if event.AdmissionID != item.Values["admission_id"] ||
		event.FinalizationDigest != item.Values["finalization_digest"] ||
		string(event.EvidenceState) != item.Values["evidence_state"] {
		return TerminalEvent{}, poisonedStreamItem(item.ID, "event_envelope_mismatch", nil)
	}
	if !isHexDigest(item.Values["admission_digest"]) {
		return TerminalEvent{}, poisonedStreamItem(item.ID, "admission_digest_invalid", nil)
	}
	return event, nil
}
