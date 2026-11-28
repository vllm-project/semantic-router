package turnselection

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
)

// Action adapts turn selection to one request's shared TransformationPlan.
// It is request-local and preserves the first evaluation only, so replaying a
// completed TransformSelectTurns step keeps its original Diagnostics.
type Action struct {
	policy    Policy
	input     RequestInput
	result    Diagnostics
	evaluated bool
}

// NewAction builds a request-local turn-selection action for one request.
func NewAction(policy Policy, input RequestInput) *Action {
	return &Action{
		policy: policy,
		input:  input,
	}
}

// Step returns the contextcompression.TransformationStep registered under
// TransformSelectTurns.
func (a *Action) Step() contextcompression.TransformationStep {
	return contextcompression.TransformationStep{
		Kind:        contextcompression.TransformSelectTurns,
		Enabled:     true,
		FailureMode: a.policy.EffectiveFailureMode(),
		Propose:     a.propose,
	}
}

func (a *Action) propose(
	ctx context.Context,
	view contextcompression.TransformationView,
) (contextcompression.TransformationEdits, error) {
	norm := a.policy.normalized()
	callCtx := ctx
	var cancel context.CancelFunc
	if norm.Scoring.Timeout > 0 {
		callCtx, cancel = context.WithTimeout(ctx, norm.Scoring.Timeout)
		defer cancel()
	}

	edits, diagnostics, err := Plan(callCtx, norm, a.input, view)
	// Enforce derived timeout before returning to the shared executor, which
	// checks only the parent context.
	if callCtx.Err() != nil && err == nil {
		diagnostics = withoutRemoval(diagnostics)
		diagnostics.Outcome = OutcomeFailed
		diagnostics.Reason = ReasonCancelled
		edits = contextcompression.TransformationEdits{}
		err = callCtx.Err()
	}
	a.record(diagnostics)
	return edits, err
}

func (a *Action) record(diagnostics Diagnostics) {
	if a.evaluated {
		return
	}
	a.result, a.evaluated = diagnostics, true
}

// Reconcile merges the action's diagnostics with the shared executor's
// authoritative TransformationReceipt. A rejected or skipped step never reports
// removed turns or messages, whatever the action proposed.
func (a *Action) Reconcile(receipts []contextcompression.TransformationReceipt) Diagnostics {
	result := a.result
	if !a.evaluated {
		norm := a.policy.normalized()
		result = Diagnostics{
			Strategy:           norm.Strategy,
			Outcome:            OutcomeSkipped,
			Reason:             ReasonNotEvaluated,
			BudgetStatus:       BudgetUnbounded,
			ScoringSource:      ScoringNone,
			TokenCounterSource: "heuristic",
			RecoveryStatus:     RecoveryNotRequired,
		}
	}
	receipt, found := selectTurnsReceipt(receipts)
	if !found {
		result = withoutRemoval(result)
		if result.Outcome == OutcomeApplied {
			result.Outcome, result.Reason = OutcomeSkipped, ReasonNotEvaluated
		}
		return result
	}
	return applyReceipt(result, a.evaluated, receipt)
}

func applyReceipt(
	result Diagnostics,
	evaluated bool,
	receipt contextcompression.TransformationReceipt,
) Diagnostics {
	switch receipt.Status {
	case contextcompression.TransformationApplied:
		result.Outcome = OutcomeApplied
		if result.Reason == "" {
			result.Reason = ReasonApplied
		}
		result.RemovedMessages = receipt.MessagesRemoved
		result.RetainedMessages = result.ExaminedMessages - receipt.MessagesRemoved
	case contextcompression.TransformationFailed:
		result.Outcome = OutcomeFailed
		if !evaluated || result.Reason == "" || receipt.Reason != "policy_failed" {
			result.Reason = receipt.Reason
		}
		result = withoutRemoval(result)
	default:
		if result.Outcome != OutcomeFailed {
			result.Outcome = OutcomeSkipped
		}
		if (!evaluated && receipt.Reason != "") || result.Reason == "" {
			result.Reason = receipt.Reason
		}
		result = withoutRemoval(result)
	}
	return result
}

func withoutRemoval(result Diagnostics) Diagnostics {
	if result.RemovedHistoryTokens > 0 {
		result.RetainedHistoryTokens += result.RemovedHistoryTokens
		result.RemovedHistoryTokens = 0
	}
	result.RemovedMessages = 0
	result.RemovedTurns = 0
	result.RetainedMessages = result.ExaminedMessages
	result.RetainedTurns = result.ExaminedTurns
	result.RecoveryStatus = RecoveryNotRequired
	return result
}

func selectTurnsReceipt(
	receipts []contextcompression.TransformationReceipt,
) (contextcompression.TransformationReceipt, bool) {
	for _, receipt := range receipts {
		if receipt.Kind == contextcompression.TransformSelectTurns {
			return receipt, true
		}
	}
	return contextcompression.TransformationReceipt{}, false
}
