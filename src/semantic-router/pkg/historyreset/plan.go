package historyreset

import (
	"context"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
)

// Plan evaluates one request and returns the messages the action proposes to
// remove. An empty proposal is a normal outcome, not a failure. The shared
// executor independently revalidates every proposal before committing it, so
// this function is the first of two guards rather than the only one.
func Plan(
	ctx context.Context,
	policy Policy,
	trigger TriggerResult,
	view contextcompression.TransformationView,
) (contextcompression.TransformationEdits, Diagnostics) {
	if policy.Timeout > 0 {
		bounded, cancel := context.WithTimeout(ctx, policy.Timeout)
		defer cancel()
		ctx = bounded
	}
	if ctx.Err() != nil {
		return contextcompression.TransformationEdits{}, skipped(trigger, ReasonCancelled)
	}
	if reason := policy.authorize(trigger); reason != "" {
		return contextcompression.TransformationEdits{}, skipped(trigger, reason)
	}
	if !policy.withinLimits(view) {
		return contextcompression.TransformationEdits{}, skipped(trigger, ReasonHistoryLimitExceeded)
	}

	removable, ok := selectRemovableTurns(ctx, view)
	if !ok {
		return contextcompression.TransformationEdits{}, skipped(trigger, ReasonCancelled)
	}
	ids, turns := collectMessages(view, removable)
	diagnostics := countsFor(view, trigger, ids)
	if len(ids) == 0 {
		diagnostics.Outcome, diagnostics.Reason = OutcomeSkipped, ReasonNoEligibleHistory
		return contextcompression.TransformationEdits{}, diagnostics
	}
	diagnostics.Outcome, diagnostics.Reason = OutcomeApplied, ReasonApplied
	diagnostics.RemovedTurns = turns
	return contextcompression.TransformationEdits{RemoveMessages: ids}, diagnostics
}

// selectRemovableTurns returns the turns whose every message may be removed.
// Removal is decided per turn because the shared layer rejects a partial turn,
// and the result is then closed over tool exchanges so a retained message can
// never lose the call or result it depends on.
func selectRemovableTurns(
	ctx context.Context,
	view contextcompression.TransformationView,
) (map[int]bool, bool) {
	removable := make(map[int]bool)
	for index, message := range view.Messages {
		if index%cancellationCheckInterval == 0 && ctx.Err() != nil {
			return nil, false
		}
		if _, seen := removable[message.TurnID]; !seen {
			removable[message.TurnID] = true
		}
		if !eligible(message) {
			removable[message.TurnID] = false
		}
	}
	return removable, closeOverExchanges(ctx, view, removable)
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

// countsFor describes the reset-stage input and the proposal against it.
// Retained counts are computed against this stage's input, not against a later
// compressed request, and removed counts are still only proposed here.
func countsFor(
	view contextcompression.TransformationView,
	trigger TriggerResult,
	proposed []int,
) Diagnostics {
	diagnostics := Diagnostics{
		Signal:           trigger.Signal,
		Scope:            ScopeEligibleHistory,
		TriggerClass:     trigger.Class,
		Version:          trigger.Version,
		ExaminedMessages: len(view.Messages),
		RemovedMessages:  len(proposed),
		RetainedMessages: len(view.Messages) - len(proposed),
	}
	for _, message := range view.Messages {
		if message.Protection != 0 {
			diagnostics.ProtectedMessages++
		}
	}
	return diagnostics
}
