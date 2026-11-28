package turnselection

import (
	"context"
	"errors"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type resolvedBudget struct {
	effectiveInputBudget   int
	reservedOutputTokens   int
	fixedOverheadTokens    int
	availableHistoryBudget int
	hasTokenCap            bool
	hasTurnCap             bool
	protectedOverflow      bool
}

// Plan computes a pure, deterministic complete-turn removal proposal over a
// detached TransformationView without mutating any protocol request.
//
// An empty TransformationEdits with a nil error is a normal no-op outcome
// (for example, when history already fits within budget or no eligible prior
// turns exist). Explicit budget conflicts, invalid policies, scan-limit
// breaches, cancellation, and fail-closed scorer failures return a non-nil
// sentinel error alongside bounded content-minimized Diagnostics.
func Plan(
	ctx context.Context,
	policy Policy,
	input RequestInput,
	view contextcompression.TransformationView,
) (contextcompression.TransformationEdits, Diagnostics, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	norm := policy.normalized()
	diag := Diagnostics{
		Strategy:           norm.Strategy,
		Outcome:            OutcomeSkipped,
		BudgetStatus:       BudgetUnbounded,
		ScoringSource:      ScoringNone,
		TokenCounterSource: "heuristic",
		RecoveryStatus:     RecoveryNotRequired,
		MinRecentRequested: norm.Budget.MinRecentTurns,
		ExaminedMessages:   len(view.Messages),
		RetainedMessages:   len(view.Messages),
	}

	if err := norm.Validate(); err != nil {
		diag.Outcome, diag.Reason = OutcomeFailed, ReasonInvalidPolicy
		return contextcompression.TransformationEdits{}, diag, err
	}
	if ctx.Err() != nil {
		diag.Outcome, diag.Reason = OutcomeFailed, ReasonCancelled
		return contextcompression.TransformationEdits{}, diag, ctx.Err()
	}

	turns := GroupTurns(view, norm.Scoring.MaxTurnTextBytes)
	diag.ExaminedTurns = len(turns)
	diag.RetainedTurns = len(turns)

	if norm.Scoring.MaxHistoryTurns > 0 && len(turns) > norm.Scoring.MaxHistoryTurns {
		diag.Outcome, diag.Reason = OutcomeFailed, ReasonScanLimitExceeded
		return contextcompression.TransformationEdits{}, diag, ErrScanLimitExceeded
	}

	estimator := costEstimatorForInput(input)
	fixedNonMessage, counterSource := estimator.FixedOverheadTokens(input.ModelBudget.Model)
	diag.TokenCounterSource = sanitizeCounterSource(counterSource)

	eligibleIndices := make([]int, 0, len(turns))
	pinnedTokens := 0
	totalEligibleTokens := 0
	for i := range turns {
		turnTokens := 0
		for _, msg := range turns[i].Messages {
			msgTokens, msgSource := estimator.MessageTokens(input.ModelBudget.Model, msg)
			turnTokens = llmprotocol.SaturatingTokenSum(turnTokens, max(0, msgTokens))
			if msgSource != "" {
				diag.TokenCounterSource = sanitizeCounterSource(msgSource)
			}
		}
		turns[i].Tokens = turnTokens
		if turns[i].Eligible {
			eligibleIndices = append(eligibleIndices, i)
			totalEligibleTokens = llmprotocol.SaturatingTokenSum(totalEligibleTokens, turnTokens)
		} else {
			diag.ProtectedTurns++
			diag.ProtectedMessages += len(turns[i].MessageIDs)
			pinnedTokens = llmprotocol.SaturatingTokenSum(pinnedTokens, turnTokens)
		}
	}
	diag.EligibleTurns = len(eligibleIndices)
	diag.MinRecentRequested = min(norm.Budget.MinRecentTurns, len(eligibleIndices))

	fixedTotal := llmprotocol.SaturatingTokenSum(max(0, fixedNonMessage), pinnedTokens)
	budget := resolveSelectionBudget(norm.Budget, input.ModelBudget, fixedTotal)
	diag.EffectiveInputBudget = budget.effectiveInputBudget
	diag.ReservedOutputTokens = budget.reservedOutputTokens
	diag.FixedOverheadTokens = budget.fixedOverheadTokens
	diag.AvailableHistoryBudget = budget.availableHistoryBudget

	if len(eligibleIndices) == 0 {
		if budget.protectedOverflow {
			return handleProtectedOverflowNoEligible(norm, turns, diag)
		}
		if budget.hasTokenCap || budget.hasTurnCap {
			diag.BudgetStatus = BudgetSatisfied
		}
		diag.Outcome, diag.Reason = OutcomeSkipped, ReasonNoEligibleHistory
		recordAllPinnedDecisions(&diag, turns)
		return contextcompression.TransformationEdits{}, diag, nil
	}

	if norm.Strategy == StrategyFull {
		return planFullBaseline(turns, eligibleIndices, totalEligibleTokens, budget, norm.Budget.MaxPriorTurns, diag)
	}

	if !budget.hasTokenCap && !budget.hasTurnCap {
		diag.Outcome, diag.Reason = OutcomeSkipped, ReasonUnboundedPolicy
		diag.BudgetStatus = BudgetUnbounded
		diag.RetainedHistoryTokens = totalEligibleTokens
		recordDecisions(&diag, turns, allRetainedMap(eligibleIndices, TurnOutcomeRetainedFull, TurnReasonFullBaseline), nil)
		return contextcompression.TransformationEdits{}, diag, nil
	}

	fitsTokens := !budget.hasTokenCap || (!budget.protectedOverflow && totalEligibleTokens <= budget.availableHistoryBudget)
	fitsTurns := !budget.hasTurnCap || len(eligibleIndices) <= norm.Budget.MaxPriorTurns
	if fitsTokens && fitsTurns {
		diag.Outcome, diag.Reason = OutcomeSkipped, ReasonWithinBudget
		diag.BudgetStatus = BudgetSatisfied
		diag.MinRecentRetained = diag.MinRecentRequested
		diag.RetainedHistoryTokens = totalEligibleTokens
		retained := make(map[int]decisionTag, len(eligibleIndices))
		recentCutoff := len(eligibleIndices) - diag.MinRecentRequested
		for pos, turnIdx := range eligibleIndices {
			if pos >= recentCutoff {
				retained[turnIdx] = decisionTag{outcome: TurnOutcomeRetainedRecent, reason: TurnReasonMinRecency}
			} else {
				retained[turnIdx] = decisionTag{outcome: TurnOutcomeRetainedRanked, reason: TurnReasonRecency}
			}
		}
		recordDecisions(&diag, turns, retained, nil)
		return contextcompression.TransformationEdits{}, diag, nil
	}

	if budget.protectedOverflow {
		return handleProtectedOverflow(norm, turns, eligibleIndices, totalEligibleTokens, diag)
	}

	return selectEligibleTurns(ctx, norm, input, turns, eligibleIndices, totalEligibleTokens, budget, diag)
}

type decisionTag struct {
	outcome string
	reason  string
}

func costEstimatorForInput(input RequestInput) TurnCostEstimator {
	if input.CostEstimator != nil {
		return input.CostEstimator
	}
	return ViewCostEstimator{Counter: input.TokenCounter}
}

func resolveSelectionBudget(
	policy BudgetPolicy,
	model ModelBudget,
	fixedTotal int,
) resolvedBudget {
	reserve := model.ReserveOutputTokens
	if reserve <= 0 {
		reserve = max(model.Capabilities.RequestedOutput, policy.OutputReserveTokens)
	}
	if reserve <= 0 {
		reserve = model.Capabilities.MaxOutputTokens
	}
	reserve = max(0, reserve)

	hasInputCap := false
	rawInputBudget := 0
	if model.Capabilities.ContextWindow > 0 {
		hasInputCap = true
		rawInputBudget = model.Capabilities.ContextWindow - reserve
	}
	if policy.MaxInputTokens > 0 {
		if !hasInputCap || policy.MaxInputTokens < rawInputBudget {
			rawInputBudget = policy.MaxInputTokens
			hasInputCap = true
		}
	}

	result := resolvedBudget{
		reservedOutputTokens: reserve,
		fixedOverheadTokens:  fixedTotal,
		hasTurnCap:           policy.MaxPriorTurns > 0,
	}

	if hasInputCap {
		result.hasTokenCap = true
		result.effectiveInputBudget = max(0, rawInputBudget)
		available := rawInputBudget - fixedTotal
		if available < 0 {
			result.protectedOverflow = true
			result.availableHistoryBudget = 0
		} else {
			result.availableHistoryBudget = available
		}
	}

	if policy.MaxHistoryTokens > 0 {
		if !result.hasTokenCap {
			result.hasTokenCap = true
			result.availableHistoryBudget = policy.MaxHistoryTokens
		} else if !result.protectedOverflow {
			result.availableHistoryBudget = min(result.availableHistoryBudget, policy.MaxHistoryTokens)
		}
	}
	return result
}

func handleProtectedOverflowNoEligible(
	policy Policy,
	turns []Turn,
	diag Diagnostics,
) (contextcompression.TransformationEdits, Diagnostics, error) {
	recordAllPinnedDecisions(&diag, turns)
	if policy.Budget.ConflictPolicy == ConflictDropEligible {
		diag.Outcome, diag.Reason = OutcomeSkipped, ReasonProtectedOverflow
		diag.BudgetStatus = BudgetProtectedOverflow
		return contextcompression.TransformationEdits{}, diag, nil
	}
	diag.Outcome, diag.Reason = OutcomeFailed, ReasonBudgetUnsatisfiable
	diag.BudgetStatus = BudgetUnsatisfiable
	return contextcompression.TransformationEdits{}, diag, ErrBudgetUnsatisfiable
}

func handleProtectedOverflow(
	policy Policy,
	turns []Turn,
	eligibleIndices []int,
	totalEligibleTokens int,
	diag Diagnostics,
) (contextcompression.TransformationEdits, Diagnostics, error) {
	if policy.Budget.ConflictPolicy != ConflictDropEligible {
		diag.Outcome, diag.Reason = OutcomeFailed, ReasonBudgetUnsatisfiable
		diag.BudgetStatus = BudgetUnsatisfiable
		diag.RetainedHistoryTokens = totalEligibleTokens
		return contextcompression.TransformationEdits{}, diag, ErrBudgetUnsatisfiable
	}
	removedIDs := make([]int, 0)
	removed := make(map[int]string, len(eligibleIndices))
	for _, turnIdx := range eligibleIndices {
		removedIDs = append(removedIDs, turns[turnIdx].MessageIDs...)
		removed[turnIdx] = TurnReasonProtectedOverflow
	}
	sort.Ints(removedIDs)
	diag.Outcome, diag.Reason = OutcomeApplied, ReasonProtectedOverflow
	diag.BudgetStatus = BudgetProtectedOverflow
	diag.RecoveryStatus = RecoveryNotConfigured
	diag.RemovedMessages = len(removedIDs)
	diag.RemovedTurns = len(eligibleIndices)
	diag.RetainedMessages = diag.ExaminedMessages - diag.RemovedMessages
	diag.RetainedTurns = diag.ExaminedTurns - diag.RemovedTurns
	diag.RemovedHistoryTokens = totalEligibleTokens
	diag.RetainedHistoryTokens = 0
	recordDecisions(&diag, turns, nil, removed)
	return contextcompression.TransformationEdits{RemoveMessages: removedIDs}, diag, nil
}

func planFullBaseline(
	turns []Turn,
	eligibleIndices []int,
	totalEligibleTokens int,
	budget resolvedBudget,
	maxPriorTurns int,
	diag Diagnostics,
) (contextcompression.TransformationEdits, Diagnostics, error) {
	diag.Outcome, diag.Reason = OutcomeSkipped, ReasonFullHistoryBaseline
	diag.MinRecentRetained = diag.MinRecentRequested
	diag.RetainedHistoryTokens = totalEligibleTokens
	switch {
	case !budget.hasTokenCap && !budget.hasTurnCap:
		diag.BudgetStatus = BudgetUnbounded
	case budget.protectedOverflow:
		diag.BudgetStatus = BudgetProtectedOverflow
	case (budget.hasTokenCap && totalEligibleTokens > budget.availableHistoryBudget) ||
		(budget.hasTurnCap && len(eligibleIndices) > maxPriorTurns):
		diag.BudgetStatus = BudgetUnsatisfiable
	default:
		diag.BudgetStatus = BudgetSatisfied
	}
	recordDecisions(&diag, turns, allRetainedMap(eligibleIndices, TurnOutcomeRetainedFull, TurnReasonFullBaseline), nil)
	return contextcompression.TransformationEdits{}, diag, nil
}

func selectEligibleTurns(
	ctx context.Context,
	policy Policy,
	input RequestInput,
	turns []Turn,
	eligibleIndices []int,
	totalEligibleTokens int,
	budget resolvedBudget,
	diag Diagnostics,
) (contextcompression.TransformationEdits, Diagnostics, error) {
	remainingTokens := budget.availableHistoryBudget
	remainingTurns := policy.Budget.MaxPriorTurns
	retained := make(map[int]decisionTag, len(eligibleIndices))
	retainedTokens := 0
	retainedEligible := 0

	canFit := func(turn Turn) bool {
		if budget.hasTurnCap && retainedEligible >= remainingTurns {
			return false
		}
		if budget.hasTokenCap && turn.Tokens > remainingTokens {
			return false
		}
		return true
	}

	retain := func(turnIdx int, tag decisionTag) {
		retained[turnIdx] = tag
		retainedEligible++
		retainedTokens = llmprotocol.SaturatingTokenSum(retainedTokens, turns[turnIdx].Tokens)
		if budget.hasTokenCap {
			remainingTokens -= turns[turnIdx].Tokens
		}
	}

	// Step 1: guarantee the contiguous newest MinRecentTurns window.
	minRecentTarget := diag.MinRecentRequested
	minRecentRelaxed := false
	for offset := 0; offset < minRecentTarget; offset++ {
		turnIdx := eligibleIndices[len(eligibleIndices)-1-offset]
		if !canFit(turns[turnIdx]) {
			if policy.Budget.ConflictPolicy == ConflictStrict {
				diag.Outcome, diag.Reason = OutcomeFailed, ReasonMinRecencyConflict
				diag.BudgetStatus = BudgetUnsatisfiable
				diag.RetainedHistoryTokens = totalEligibleTokens
				return contextcompression.TransformationEdits{}, diag, ErrMinRecencyConflict
			}
			minRecentRelaxed = true
			break
		}
		retain(turnIdx, decisionTag{outcome: TurnOutcomeRetainedRecent, reason: TurnReasonMinRecency})
		diag.MinRecentRetained++
	}

	// Step 2: rank and fill remaining budget from non-retained eligible turns.
	remainingCandidates := make([]int, 0, len(eligibleIndices)-retainedEligible)
	for _, turnIdx := range eligibleIndices {
		if _, kept := retained[turnIdx]; !kept {
			remainingCandidates = append(remainingCandidates, turnIdx)
		}
	}

	anyCanFit := false
	for _, turnIdx := range remainingCandidates {
		if canFit(turns[turnIdx]) {
			anyCanFit = true
			break
		}
	}

	var scores map[int]turnScore
	if policy.Strategy == StrategyRelevance && anyCanFit {
		scored, err := scoreEligibleTurns(ctx, policy, input.Scorer, turns, eligibleIndices)
		diag.ScoringSource = scored.source
		diag.FallbackReason = scored.fallbackReason
		diag.ScoredTurns = scored.scoredTurns
		diag.CandidatesTruncated = scored.candidatesTruncated
		diag.ContinuityBoostedTurns = scored.continuityBoosted
		if err != nil {
			diag.Outcome = OutcomeFailed
			diag.RetainedHistoryTokens = totalEligibleTokens
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				diag.Reason = ReasonCancelled
			} else {
				diag.Reason = ReasonScoringFailed
			}
			return contextcompression.TransformationEdits{}, diag, err
		}
		scores = scored.scores
		sort.SliceStable(remainingCandidates, func(left, right int) bool {
			lt, rt := turns[remainingCandidates[left]], turns[remainingCandidates[right]]
			ls, rs := scores[lt.Ordinal], scores[rt.Ordinal]
			if ls.total != rs.total {
				return ls.total > rs.total
			}
			if ls.relevance != rs.relevance {
				return ls.relevance > rs.relevance
			}
			if ls.continuity != rs.continuity {
				return ls.continuity > rs.continuity
			}
			if lt.Ordinal != rt.Ordinal {
				return lt.Ordinal > rt.Ordinal
			}
			if lt.TurnID != rt.TurnID {
				return lt.TurnID > rt.TurnID
			}
			return lt.FirstMessageID > rt.FirstMessageID
		})
	} else {
		// StrategyRecency: walk remaining turns newest -> oldest.
		for left, right := 0, len(remainingCandidates)-1; left < right; left, right = left+1, right-1 {
			remainingCandidates[left], remainingCandidates[right] = remainingCandidates[right], remainingCandidates[left]
		}
	}

	for _, turnIdx := range remainingCandidates {
		if !canFit(turns[turnIdx]) {
			continue
		}
		reason := TurnReasonRecency
		if policy.Strategy == StrategyRelevance {
			sc := scores[turns[turnIdx].Ordinal]
			switch {
			case sc.relevance == 0 && sc.continuity > 0:
				reason = TurnReasonContinuity
			case sc.relevance > 0:
				reason = TurnReasonRelevance
			default:
				reason = TurnReasonRecency
			}
		}
		retain(turnIdx, decisionTag{outcome: TurnOutcomeRetainedRanked, reason: reason})
	}

	// Step 3: emit sorted stable original message IDs for unretained eligible turns.
	removedIDs := make([]int, 0)
	removedReasons := make(map[int]string)
	removedTurns := 0
	removedTokens := 0
	for _, turnIdx := range eligibleIndices {
		if _, kept := retained[turnIdx]; kept {
			continue
		}
		turn := turns[turnIdx]
		removedTurns++
		removedTokens = llmprotocol.SaturatingTokenSum(removedTokens, turn.Tokens)
		removedIDs = append(removedIDs, turn.MessageIDs...)
		if budget.hasTurnCap && retainedEligible >= remainingTurns && (!budget.hasTokenCap || turn.Tokens <= remainingTokens) {
			removedReasons[turnIdx] = TurnReasonOverTurnLimit
		} else {
			removedReasons[turnIdx] = TurnReasonOverTokenBudget
		}
	}
	sort.Ints(removedIDs)

	diag.RetainedHistoryTokens = retainedTokens
	diag.RemovedHistoryTokens = removedTokens
	diag.RemovedMessages = len(removedIDs)
	diag.RemovedTurns = removedTurns
	diag.RetainedMessages = diag.ExaminedMessages - diag.RemovedMessages
	diag.RetainedTurns = diag.ExaminedTurns - diag.RemovedTurns
	if minRecentRelaxed {
		diag.BudgetStatus = BudgetMinRecencyRelaxed
	} else {
		diag.BudgetStatus = BudgetSatisfied
	}
	if len(removedIDs) == 0 {
		diag.Outcome, diag.Reason = OutcomeSkipped, ReasonWithinBudget
		diag.RecoveryStatus = RecoveryNotRequired
	} else {
		diag.Outcome = OutcomeApplied
		diag.RecoveryStatus = RecoveryNotConfigured
		if minRecentRelaxed {
			diag.Reason = ReasonMinRecencyRelaxed
		} else {
			diag.Reason = ReasonApplied
		}
	}
	recordDecisions(&diag, turns, retained, removedReasons)
	return contextcompression.TransformationEdits{RemoveMessages: removedIDs}, diag, nil
}

func allRetainedMap(indices []int, outcome, reason string) map[int]decisionTag {
	result := make(map[int]decisionTag, len(indices))
	for _, idx := range indices {
		result[idx] = decisionTag{outcome: outcome, reason: reason}
	}
	return result
}

func recordAllPinnedDecisions(diag *Diagnostics, turns []Turn) {
	recordDecisions(diag, turns, nil, nil)
}

func recordDecisions(
	diag *Diagnostics,
	turns []Turn,
	retained map[int]decisionTag,
	removed map[int]string,
) {
	limit := min(len(turns), MaxReceiptTurns)
	if len(turns) > MaxReceiptTurns {
		diag.DecisionsTruncated = true
	}
	decisions := make([]TurnDecision, 0, limit)
	for i := 0; i < limit; i++ {
		turn := turns[i]
		entry := TurnDecision{
			TurnID:         turn.TurnID,
			FirstMessageID: turn.FirstMessageID,
			MessageCount:   len(turn.MessageIDs),
			Tokens:         turn.Tokens,
		}
		switch {
		case !turn.Eligible:
			entry.Outcome = TurnOutcomePinned
			entry.Reason = turn.PinReason
			if entry.Reason == "" {
				entry.Reason = TurnReasonProtected
			}
		case removed != nil && removed[i] != "":
			entry.Outcome = TurnOutcomeRemoved
			entry.Reason = removed[i]
		case retained != nil && retained[i].outcome != "":
			entry.Outcome = retained[i].outcome
			entry.Reason = retained[i].reason
		default:
			entry.Outcome = TurnOutcomeRetainedFull
			entry.Reason = TurnReasonFullBaseline
		}
		decisions = append(decisions, entry)
	}
	diag.Decisions = decisions
}
