package historyreset

import (
	"context"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
)

// plan evaluates one request and returns the messages the action proposes to
// remove. An empty proposal is a normal outcome, not a failure. The shared
// executor independently revalidates every proposal before committing it, so
// this function is the first of two guards rather than the only one.
//
// The order is deliberate. A request with nothing removable is an ordinary
// no-op, so evidence is judged only once the authoritative removable set is
// known: otherwise every opening turn, whose topic result is always unknown,
// would fail under fail_closed. A validated continuation is the one evidence
// outcome decided early, because it can never authorize a removal and must
// stay a no-op however long the history is. The history limits are enforced
// before the tool-exchange closure, which is the expensive part of planning.
func plan(
	ctx context.Context,
	policy Policy,
	trigger TriggerResult,
	view contextcompression.TransformationView,
) (contextcompression.TransformationEdits, Diagnostics) {
	// The caller owns the budget: the action applies the policy timeout before
	// calling in, because it must also cover envelope construction and the
	// recovery write that follow selection.
	if ctx.Err() != nil {
		return contextcompression.TransformationEdits{}, skipped(trigger, ReasonCancelled)
	}
	history, ok := surveyHistory(ctx, policy, view)
	if !ok {
		return contextcompression.TransformationEdits{}, skipped(trigger, ReasonCancelled)
	}
	unchanged := func(reason string) (contextcompression.TransformationEdits, Diagnostics) {
		diagnostics := inputCounts(view, trigger)
		diagnostics.Outcome, diagnostics.Reason = OutcomeSkipped, reason
		return contextcompression.TransformationEdits{}, diagnostics
	}
	if !history.candidate {
		return unchanged(ReasonNoEligibleHistory)
	}
	authorization := policy.authorize(trigger)
	if authorization == ReasonEvidenceContinuation {
		return unchanged(authorization)
	}
	if !history.withinLimits {
		return unchanged(ReasonHistoryLimitExceeded)
	}
	if !closeOverExchanges(ctx, view, history.removable) {
		return contextcompression.TransformationEdits{}, skipped(trigger, ReasonCancelled)
	}
	ids, turns := collectMessages(view, history.removable)
	if len(ids) == 0 {
		return unchanged(ReasonNoEligibleHistory)
	}
	if authorization != "" {
		return unchanged(authorization)
	}
	diagnostics := inputCounts(view, trigger)
	diagnostics.Outcome, diagnostics.Reason = OutcomeApplied, ReasonApplied
	diagnostics.RemovedMessages, diagnostics.RemovedTurns = len(ids), turns
	diagnostics.RetainedMessages -= len(ids)
	return contextcompression.TransformationEdits{RemoveMessages: ids}, diagnostics
}

type historySurvey struct {
	removable    map[int]bool
	candidate    bool
	withinLimits bool
}

// surveyHistory makes the single preflight pass. A turn is removable only
// when every one of its messages is eligible, because the shared layer
// rejects a partial turn; closure over tool exchanges happens later.
func surveyHistory(
	ctx context.Context,
	policy Policy,
	view contextcompression.TransformationView,
) (historySurvey, bool) {
	survey := historySurvey{removable: make(map[int]bool)}
	historyTurns := make(map[int]struct{})
	historyBytes := 0
	for index, message := range view.Messages {
		if index%cancellationCheckInterval == 0 && ctx.Err() != nil {
			return historySurvey{}, false
		}
		if _, seen := survey.removable[message.TurnID]; !seen {
			survey.removable[message.TurnID] = true
		}
		if eligible(message) {
			survey.candidate = true
		} else {
			survey.removable[message.TurnID] = false
		}
		if message.Source == contextcompression.SourceHistory {
			historyTurns[message.TurnID] = struct{}{}
			for _, block := range message.Blocks {
				historyBytes += len(block.Text)
			}
		}
	}
	survey.withinLimits = (policy.MaxHistoryTurns <= 0 || len(historyTurns) <= policy.MaxHistoryTurns) &&
		(policy.MaxHistoryBytes <= 0 || historyBytes <= policy.MaxHistoryBytes)
	return survey, true
}

// cancellationCheckInterval keeps the deadline responsive without paying for a
// context read on every message of a short history.
const cancellationCheckInterval = 64

// eligible reports whether one message may be removed on its own merits. Both
// conditions are required: the shared layer marks eligibility, and any
// protection at all keeps the message.
func eligible(message contextcompression.MessageView) bool {
	return message.Eligibility&contextcompression.EligibleHistoryRemoval != 0 &&
		message.Protection == 0
}

// closeOverExchanges retains any turn that shares a tool exchange with a
// retained message, repeating until the set stops shrinking. Retention
// propagates, so one protected result can pin several turns; the set never
// grows, which bounds the iteration by the number of turns.
func closeOverExchanges(
	ctx context.Context,
	view contextcompression.TransformationView,
	removable map[int]bool,
) bool {
	for {
		if ctx.Err() != nil {
			return false
		}
		retained := make(map[string]bool)
		for _, message := range view.Messages {
			if removable[message.TurnID] {
				continue
			}
			for _, exchange := range message.ExchangeIDs {
				retained[exchange] = true
			}
		}
		changed := false
		for _, message := range view.Messages {
			if !removable[message.TurnID] {
				continue
			}
			for _, exchange := range message.ExchangeIDs {
				if retained[exchange] {
					removable[message.TurnID] = false
					changed = true
					break
				}
			}
		}
		if !changed {
			return true
		}
	}
}

// collectMessages returns the stable IDs of every message in a removable turn,
// in ascending order, together with the number of turns they cover.
func collectMessages(
	view contextcompression.TransformationView,
	removable map[int]bool,
) ([]int, int) {
	var ids []int
	turns := make(map[int]struct{})
	for _, message := range view.Messages {
		if !removable[message.TurnID] {
			continue
		}
		ids = append(ids, message.ID)
		turns[message.TurnID] = struct{}{}
	}
	sort.Ints(ids)
	return ids, len(turns)
}

// inputCounts describes the reset-stage input with nothing removed: every
// examined message is retained. An accepted proposal subtracts its removals;
// the executor's receipt remains authoritative once a removal is committed.
func inputCounts(view contextcompression.TransformationView, trigger TriggerResult) Diagnostics {
	diagnostics := Diagnostics{
		Signal:           trigger.Signal,
		Scope:            ScopeEligibleHistory,
		TriggerClass:     trigger.Class,
		Version:          trigger.Version,
		ExaminedMessages: len(view.Messages),
		RetainedMessages: len(view.Messages),
	}
	for _, message := range view.Messages {
		if message.Protection != 0 {
			diagnostics.ProtectedMessages++
		}
	}
	return diagnostics
}
