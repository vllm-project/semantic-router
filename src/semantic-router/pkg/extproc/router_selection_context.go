package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func (r *OpenAIRouter) buildSelectionContext(
	ctx *RequestContext,
	modelRefs []config.ModelRef,
	decisionName string,
	query string,
	originalModel string,
	algorithm *config.AlgorithmConfig,
	categoryName string,
) *selection.SelectionContext {
	costWeight, qualityWeight := r.getSelectionWeights(algorithm)
	latencyAwareTPOTPercentile, latencyAwareTTFTPercentile := r.getLatencyAwarePercentiles(algorithm)
	taskFamily := resolveSelectionTaskFamily(categoryName, decisionName)
	currentModel := strings.TrimSpace(ctx.PreviousModel)
	qualityGapByCandidate, handoffPenaltyByCandidate := r.lookupCandidateSessionFacts(taskFamily, currentModel, modelRefs)

	return &selection.SelectionContext{
		Query:                      query,
		DecisionName:               decisionName,
		CategoryName:               categoryName,
		CandidateModels:            modelRefs,
		CostWeight:                 costWeight,
		QualityWeight:              qualityWeight,
		UserID:                     extractUserID(ctx),
		SessionID:                  ctx.SessionID,
		TurnIndex:                  ctx.TurnIndex,
		PreviousModel:              ctx.PreviousModel,
		CurrentModel:               currentModel,
		OriginalModel:              originalModel,
		CacheWarmthEstimate:        ctx.CacheWarmthEstimate,
		RemainingTurnsEstimate:     r.lookupRemainingTurns(taskFamily, ctx.TurnIndex),
		QualityGapByCandidate:      qualityGapByCandidate,
		HandoffPenaltyByCandidate:  handoffPenaltyByCandidate,
		LatencyAwareTPOTPercentile: latencyAwareTPOTPercentile,
		LatencyAwareTTFTPercentile: latencyAwareTTFTPercentile,
	}
}

func resolveSelectionTaskFamily(categoryName string, decisionName string) string {
	if trimmed := strings.TrimSpace(categoryName); trimmed != "" {
		return trimmed
	}
	return strings.TrimSpace(decisionName)
}

func (r *OpenAIRouter) lookupCandidateSessionFacts(
	taskFamily string,
	currentModel string,
	modelRefs []config.ModelRef,
) (map[string]float64, map[string]float64) {
	qualityGapByCandidate := make(map[string]float64)
	handoffPenaltyByCandidate := make(map[string]float64)
	if r == nil || r.LookupTable == nil || strings.TrimSpace(currentModel) == "" {
		return qualityGapByCandidate, handoffPenaltyByCandidate
	}

	for _, ref := range modelRefs {
		candidateModel := strings.TrimSpace(ref.Model)
		if candidateModel == "" || candidateModel == strings.TrimSpace(currentModel) {
			continue
		}
		qualityGapByCandidate[candidateModel] = r.lookupQualityGap(taskFamily, currentModel, candidateModel)
		handoffPenaltyByCandidate[candidateModel] = r.lookupHandoffPenalty(currentModel, candidateModel)
	}
	return qualityGapByCandidate, handoffPenaltyByCandidate
}

func (r *OpenAIRouter) lookupRemainingTurns(taskFamily string, turnIndex int) float64 {
	if r == nil || r.LookupTable == nil || strings.TrimSpace(taskFamily) == "" {
		return 0
	}
	prior, ok := r.LookupTable.RemainingTurnPrior(strings.TrimSpace(taskFamily))
	if !ok {
		return 0
	}
	remaining := prior - float64(turnIndex)
	if remaining < 0 {
		return 0
	}
	return remaining
}

func (r *OpenAIRouter) lookupQualityGap(taskFamily string, currentModel string, candidateModel string) float64 {
	if r == nil || r.LookupTable == nil || strings.TrimSpace(taskFamily) == "" || strings.TrimSpace(currentModel) == "" || strings.TrimSpace(candidateModel) == "" {
		return 0
	}
	value, ok := r.LookupTable.QualityGap(strings.TrimSpace(taskFamily), strings.TrimSpace(currentModel), strings.TrimSpace(candidateModel))
	if !ok {
		return 0
	}
	return value
}

func (r *OpenAIRouter) lookupHandoffPenalty(currentModel string, candidateModel string) float64 {
	if r == nil || r.LookupTable == nil || strings.TrimSpace(currentModel) == "" || strings.TrimSpace(candidateModel) == "" {
		return 0
	}
	value, ok := r.LookupTable.HandoffPenalty(strings.TrimSpace(currentModel), strings.TrimSpace(candidateModel))
	if !ok {
		return 0
	}
	return value
}
