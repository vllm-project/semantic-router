package contextdedup

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const (
	llmRoleUser      = llmprotocol.RoleUser
	llmRoleAssistant = llmprotocol.RoleAssistant
)

// Action adapts the policy to one request. It is request-local and records
// the first evaluation only, so a repeated plan reports the original result.
type Action struct {
	policy    Policy
	blocked   string
	resolver  Resolver
	result    Diagnostics
	evaluated bool
}

// NewAction builds the action for one request. A non-empty blocked reason
// makes the step fail with that reason instead of evaluating.
func NewAction(policy Policy, blocked string) *Action {
	return &Action{policy: policy, blocked: blocked}
}

// WithResolver supplies the neutral request messages for stage-two proof.
func (a *Action) WithResolver(resolver Resolver) *Action {
	a.resolver = resolver
	return a
}

// Step returns the shared transformation step.
func (a *Action) Step() contextcompression.TransformationStep {
	mode := contextcompression.FailureOpen
	if a.policy.FailClosed {
		mode = contextcompression.FailureClosed
	}
	return contextcompression.TransformationStep{
		Kind:        contextcompression.TransformDeduplicate,
		Enabled:     true,
		FailureMode: mode,
		Propose:     a.propose,
	}
}

func (a *Action) propose(
	ctx context.Context,
	view contextcompression.TransformationView,
) (contextcompression.TransformationEdits, error) {
	if a.blocked != "" {
		a.record(failed(a.policy, view, a.blocked))
		return contextcompression.TransformationEdits{}, errBlocked(a.blocked)
	}
	if a.policy.Timeout > 0 {
		bounded, cancel := context.WithTimeout(ctx, a.policy.Timeout)
		defer cancel()
		ctx = bounded
	}
	edits, diagnostics := plan(ctx, a.policy, a.resolver, view)
	if evaluationFailure(diagnostics.Reason) {
		diagnostics.Outcome = OutcomeFailed
		a.record(diagnostics)
		return contextcompression.TransformationEdits{}, errBlocked(diagnostics.Reason)
	}
	a.record(diagnostics)
	return edits, nil
}

func (a *Action) record(diagnostics Diagnostics) {
	if a.evaluated {
		return
	}
	a.result, a.evaluated = diagnostics, true
}

// Reconcile lets the shared executor's receipt be authoritative for the
// committed status. A proposal the executor rejected or skipped removed
// nothing, whatever the action computed.
func (a *Action) Reconcile(receipts []contextcompression.TransformationReceipt) Diagnostics {
	result := a.result
	if !a.evaluated {
		result = Diagnostics{
			Scope: ScopeEligibleHistory, Normalization: a.policy.Normalization,
			Outcome: OutcomeSkipped, Reason: ReasonNotEvaluated,
			RecoveryStatus: RecoveryNotRequired, Recovery: RecoveryRetainedTwin,
		}
		if result.Normalization == "" {
			result.Normalization = NormalizationExact
		}
	}
	receipt, found := dedupReceipt(receipts)
	if !found {
		result = withoutRemoval(result)
		if result.Outcome == OutcomeApplied {
			result.Outcome, result.Reason = OutcomeSkipped, ReasonNotEvaluated
		}
		return result
	}
	return applyReceipt(result, receipt)
}

func applyReceipt(result Diagnostics, receipt contextcompression.TransformationReceipt) Diagnostics {
	switch receipt.Status {
	case contextcompression.TransformationApplied:
		result.Outcome, result.Reason = OutcomeApplied, ReasonApplied
		result.RemovedMessages = receipt.MessagesRemoved
		result.RetainedMessages = result.ExaminedMessages - receipt.MessagesRemoved
	case contextcompression.TransformationFailed:
		result = withoutRemoval(result)
		result.Outcome = OutcomeFailed
		switch {
		case result.Reason == "" || receipt.Reason != "policy_failed":
			result.Reason = receipt.Reason
		case !evaluationFailure(result.Reason):
			// The executor refuses a proposal it received after the context
			// expired, so a policy that reported success was overtaken by
			// cancellation, not by its own evaluation.
			result.Reason = ReasonCancelled
		}
	default:
		result = withoutRemoval(result)
		if result.Outcome != OutcomeFailed {
			result.Outcome = OutcomeSkipped
		}
		if result.Reason == "" {
			result.Reason = receipt.Reason
		}
	}
	return result
}

func withoutRemoval(result Diagnostics) Diagnostics {
	result.RemovedMessages, result.RemovedTurns, result.RemovedTextBytes = 0, 0, 0
	result.DuplicateSegments, result.Segments, result.SegmentsTruncated = 0, nil, false
	result.RetainedMessages = result.ExaminedMessages
	return result
}

func dedupReceipt(receipts []contextcompression.TransformationReceipt) (contextcompression.TransformationReceipt, bool) {
	for _, receipt := range receipts {
		if receipt.Kind == contextcompression.TransformDeduplicate {
			return receipt, true
		}
	}
	return contextcompression.TransformationReceipt{}, false
}

func failed(policy Policy, view contextcompression.TransformationView, reason string) Diagnostics {
	result := skipped(policy, view, reason)
	result.Outcome = OutcomeFailed
	return result
}

type blockedError string

func (e blockedError) Error() string { return string(e) }

func errBlocked(reason string) error { return blockedError(reason) }
