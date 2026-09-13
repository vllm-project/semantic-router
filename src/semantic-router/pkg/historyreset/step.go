package historyreset

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// ReasonNotEvaluated marks a configured action whose callback never ran, for
// example because an earlier terminal condition stopped the plan. It is
// reported explicitly rather than being mistaken for an evaluated skip.
const ReasonNotEvaluated = "not_evaluated"

// Action adapts the pure policy to one request's transformation plan. It owns
// the request-local diagnostic and is not safe for concurrent use across
// requests, matching the plan it registers into.
type Action struct {
	inherited bool
	policy    Policy
	trigger   TriggerResult
	blocked   string
	recovery  RecoveryWriter
	detached  map[int]llmprotocol.Message
	key       string
	result    Diagnostics
	evaluated bool
}

// NewInheritedAction builds the adapter for an internal follow-up of a public
// turn that was already evaluated. It never removes anything: appended tool
// results and model hops continue an exchange rather than starting a new
// topic. It stays enabled so the shared live-history compression protection an
// enabled history policy implies remains active on the follow-up's own IR.
func NewInheritedAction(policy Policy) *Action {
	return &Action{policy: policy, inherited: true}
}

// NewAction builds the per-request adapter. blocked carries a terminal reason
// established before planning, such as an unsupported request representation
// or unavailable required recovery; an empty value means the action may run.
func NewAction(policy Policy, trigger TriggerResult, blocked string) *Action {
	return &Action{policy: policy, trigger: trigger, blocked: blocked}
}

// WithRecovery makes removal recoverable. detached holds the complete neutral
// messages captured before any transformation, keyed by their stable request
// ID; the callback's text-only view cannot serialize tool payloads or media,
// so the trusted caller supplies them. Removal only commits once the payload
// has been stored.
func (a *Action) WithRecovery(
	writer RecoveryWriter,
	detached map[int]llmprotocol.Message,
) *Action {
	a.recovery, a.detached = writer, detached
	return a
}

// RecoveryKey returns the key issued for this request, or an empty string when
// nothing was stored. The caller adds it to the request-level key set; it is
// never written to receipts or metrics labels.
func (a *Action) RecoveryKey() string {
	return a.key
}

// Step returns the shared transformation step. The proposal only removes whole
// eligible turns; text replacement belongs to compression.
func (a *Action) Step() contextcompression.TransformationStep {
	mode := contextcompression.FailureOpen
	if a.policy.FailClosed {
		mode = contextcompression.FailureClosed
	}
	return contextcompression.TransformationStep{
		Kind:        contextcompression.TransformReset,
		Enabled:     true,
		FailureMode: mode,
		Propose:     a.propose,
	}
}

func (a *Action) propose(
	ctx context.Context,
	view contextcompression.TransformationView,
) (contextcompression.TransformationEdits, error) {
	// A blocked action is a step failure in both modes. The step's declared
	// failure mode then decides whether the remaining plan continues, which is
	// the shared executor's existing contract.
	// An inherited follow-up evaluates nothing: the public turn it continues
	// was already decided, and appended tool results are not a new topic.
	if a.inherited {
		a.record(skipped(a.trigger, ReasonInheritedCompleted))
		return contextcompression.TransformationEdits{}, nil
	}
	if a.blocked != "" {
		a.record(failed(a.trigger, a.blocked))
		return contextcompression.TransformationEdits{}, errBlocked(a.blocked)
	}
	edits, diagnostics := Plan(ctx, a.policy, a.trigger, view)
	if len(edits.RemoveMessages) == 0 || a.recovery == nil {
		if len(edits.RemoveMessages) > 0 {
			diagnostics.RecoveryStatus = RecoveryNotRequired
		}
		a.record(diagnostics)
		return edits, nil
	}
	if err := a.persist(ctx, edits.RemoveMessages, view, &diagnostics); err != nil {
		a.record(diagnostics)
		return contextcompression.TransformationEdits{}, err
	}
	a.record(diagnostics)
	return edits, nil
}

// persist stores the removed turns before the executor commits the removal.
// Once the shared executor deletes messages there is no rollback, so a failed
// or oversized write must stop the action here rather than afterwards.
func (a *Action) persist(
	ctx context.Context,
	ids []int,
	view contextcompression.TransformationView,
	diagnostics *Diagnostics,
) error {
	turns := make(map[int]int, len(view.Messages))
	for _, message := range view.Messages {
		turns[message.ID] = message.TurnID
	}
	payload, err := a.buildEnvelope(ids, turns)
	if err != nil {
		diagnostics.Outcome, diagnostics.Reason = OutcomeFailed, ReasonRecoveryWriteFailed
		diagnostics.RecoveryStatus = RecoveryFailed
		return err
	}
	if a.policy.MaxRecoveryBytes > 0 && len(payload) > a.policy.MaxRecoveryBytes {
		diagnostics.Outcome, diagnostics.Reason = OutcomeFailed, ReasonRecoveryLimitExceeded
		diagnostics.RecoveryStatus = RecoveryFailed
		return errBlocked(ReasonRecoveryLimitExceeded)
	}
	key, err := a.recovery.Store(ctx, payload)
	if err != nil {
		diagnostics.Outcome, diagnostics.Reason = OutcomeFailed, ReasonRecoveryWriteFailed
		diagnostics.RecoveryStatus = RecoveryFailed
		return err
	}
	a.key = key
	diagnostics.RecoveryStatus, diagnostics.RecoveryEntries = RecoveryStored, 1
	return nil
}

// record keeps the first finalized result. Re-entering a completed kind within
// one request must not emit a second applied event.
func (a *Action) record(diagnostics Diagnostics) {
	if a.evaluated {
		return
	}
	a.result, a.evaluated = diagnostics, true
}

// Reconcile returns the final diagnostic for the request. The executor's
// receipt is authoritative for committed status and removal counts; the action
// only explains outcomes the generic receipt reports as policy_failed or
// no_changes.
func (a *Action) Reconcile(
	receipts []contextcompression.TransformationReceipt,
) Diagnostics {
	result := a.result
	if !a.evaluated {
		result = Diagnostics{
			Signal:       a.trigger.Signal,
			TriggerClass: a.trigger.Class,
			Version:      a.trigger.Version,
			Outcome:      OutcomeSkipped,
			Reason:       ReasonNotEvaluated,
		}
	}
	receipt, found := resetReceipt(receipts)
	if !found {
		result.RemovedMessages, result.RemovedTurns = 0, 0
		if result.Outcome == OutcomeApplied {
			result.Outcome, result.Reason = OutcomeSkipped, ReasonNotEvaluated
		}
		return result
	}
	return applyReceipt(result, receipt)
}

// applyReceipt lets the committed receipt override a proposed outcome.
func applyReceipt(
	result Diagnostics,
	receipt contextcompression.TransformationReceipt,
) Diagnostics {
	switch receipt.Status {
	case contextcompression.TransformationApplied:
		result.Outcome, result.Reason = OutcomeApplied, ReasonApplied
		result.RemovedMessages = receipt.MessagesRemoved
		result.RetainedMessages = result.ExaminedMessages - receipt.MessagesRemoved
	case contextcompression.TransformationFailed:
		result.Outcome = OutcomeFailed
		// A bounded action reason explains the executor's generic
		// policy_failed; an invariant violation it detected itself does not.
		if result.Reason == "" || receipt.Reason != "policy_failed" {
			result.Reason = receipt.Reason
		}
		result.RemovedMessages, result.RemovedTurns = 0, 0
	default:
		// Never upgrade an evaluated failure into a skip.
		if result.Outcome != OutcomeFailed {
			result.Outcome = OutcomeSkipped
		}
		if result.Reason == "" {
			result.Reason = receipt.Reason
		}
		result.RemovedMessages, result.RemovedTurns = 0, 0
	}
	return result
}

func resetReceipt(
	receipts []contextcompression.TransformationReceipt,
) (contextcompression.TransformationReceipt, bool) {
	for _, receipt := range receipts {
		if receipt.Kind == contextcompression.TransformReset {
			return receipt, true
		}
	}
	return contextcompression.TransformationReceipt{}, false
}

func failed(trigger TriggerResult, reason string) Diagnostics {
	result := skipped(trigger, reason)
	result.Outcome = OutcomeFailed
	return result
}

type blockedError string

func (e blockedError) Error() string { return string(e) }

func errBlocked(reason string) error { return blockedError(reason) }
