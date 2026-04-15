package selection

import (
	"context"
	"fmt"
	"math"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/lookuptable"
)

// SessionAwareConfig configures stay-versus-switch routing for multi-turn sessions.
type SessionAwareConfig struct {
	FallbackMethod       string  `yaml:"fallback_method"`
	MinTurnsBeforeSwitch int     `yaml:"min_turns_before_switch"`
	StayBias             float64 `yaml:"stay_bias"`
	QualityGapMultiplier float64 `yaml:"quality_gap_multiplier"`
	HandoffPenaltyWeight float64 `yaml:"handoff_penalty_weight"`
	RemainingTurnWeight  float64 `yaml:"remaining_turn_weight"`
}

// DefaultSessionAwareConfig returns conservative defaults that prefer staying on
// the current session model unless a candidate has a materially better replay-backed score.
func DefaultSessionAwareConfig() *SessionAwareConfig {
	return &SessionAwareConfig{
		FallbackMethod:       string(MethodStatic),
		MinTurnsBeforeSwitch: 1,
		StayBias:             0.25,
		QualityGapMultiplier: 1.0,
		HandoffPenaltyWeight: 1.0,
		RemainingTurnWeight:  0.15,
	}
}

// SessionAwareSelector selects models using runtime session facts and lookup-table priors.
type SessionAwareSelector struct {
	config      *SessionAwareConfig
	lookupTable lookuptable.LookupTable
}

// NewSessionAwareSelector creates a new session-aware selector.
func NewSessionAwareSelector(cfg *SessionAwareConfig) *SessionAwareSelector {
	if cfg == nil {
		cfg = DefaultSessionAwareConfig()
	}
	return &SessionAwareSelector{config: cfg}
}

// Method returns the selection method type.
func (s *SessionAwareSelector) Method() SelectionMethod {
	return MethodSessionAware
}

// SetLookupTable attaches replay-derived lookup-table priors.
func (s *SessionAwareSelector) SetLookupTable(lt lookuptable.LookupTable) {
	s.lookupTable = lt
}

// UpdateFeedback is currently a no-op because the selector consumes replay-derived priors.
func (s *SessionAwareSelector) UpdateFeedback(ctx context.Context, feedback *Feedback) error {
	_ = ctx
	_ = feedback
	return nil
}

// Select chooses the best candidate for a continuing conversation.
func (s *SessionAwareSelector) Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
	_ = ctx
	if selCtx == nil {
		return nil, fmt.Errorf("selection context is required")
	}
	if len(selCtx.CandidateModels) == 0 {
		return nil, fmt.Errorf("no candidate models provided")
	}

	if len(selCtx.CandidateModels) == 1 {
		candidate := selCtx.CandidateModels[0]
		return &SelectionResult{
			SelectedModel: candidate.Model,
			LoRAName:      candidate.LoRAName,
			Score:         1.0,
			Confidence:    1.0,
			Method:        MethodSessionAware,
			Reasoning:     "Single candidate available",
			AllScores: map[string]float64{
				candidate.Model: 1.0,
			},
		}, nil
	}

	currentModel := sessionCurrentModel(selCtx)
	if selCtx.SessionID == "" || currentModel == "" {
		return s.fallbackSelect(ctx, selCtx, "Session context unavailable")
	}

	if selCtx.TurnIndex < s.config.MinTurnsBeforeSwitch {
		if candidate := findCandidateByCurrentModel(selCtx.CandidateModels, currentModel); candidate != nil {
			return &SelectionResult{
				SelectedModel: candidate.Model,
				LoRAName:      candidate.LoRAName,
				Score:         1.0,
				Confidence:    1.0,
				Method:        MethodSessionAware,
				Reasoning: fmt.Sprintf(
					"Staying on %s because turn_index=%d is below min_turns_before_switch=%d",
					currentModel,
					selCtx.TurnIndex,
					s.config.MinTurnsBeforeSwitch,
				),
				AllScores: map[string]float64{
					candidate.Model: 1.0,
				},
			}, nil
		}
	}

	allScores := make(map[string]float64, len(selCtx.CandidateModels))
	bestIdx := 0
	bestScore := math.Inf(-1)
	secondBest := math.Inf(-1)
	bestReason := ""

	for i := range selCtx.CandidateModels {
		candidate := &selCtx.CandidateModels[i]
		score, reason := s.scoreCandidate(selCtx, currentModel, candidate)
		allScores[candidate.Model] = score
		if score > bestScore {
			secondBest = bestScore
			bestScore = score
			bestIdx = i
			bestReason = reason
		} else if score > secondBest {
			secondBest = score
		}
	}

	bestCandidate := selCtx.CandidateModels[bestIdx]
	confidence := 1.0
	if secondBest > math.Inf(-1) {
		denominator := math.Max(math.Abs(bestScore), math.Abs(secondBest))
		if denominator < 1.0 {
			denominator = 1.0
		}
		confidence = clampScore((bestScore-secondBest)/denominator, 0, 1)
	}

	logging.Infof("[SessionAwareSelector] Selected %s for session=%s turn=%d current=%s (score=%.4f, confidence=%.2f)",
		bestCandidate.Model, selCtx.SessionID, selCtx.TurnIndex, currentModel, bestScore, confidence)

	return &SelectionResult{
		SelectedModel: bestCandidate.Model,
		LoRAName:      bestCandidate.LoRAName,
		Score:         bestScore,
		Confidence:    confidence,
		Method:        MethodSessionAware,
		Reasoning:     bestReason,
		AllScores:     allScores,
	}, nil
}

func (s *SessionAwareSelector) fallbackSelect(ctx context.Context, selCtx *SelectionContext, reason string) (*SelectionResult, error) {
	method := strings.TrimSpace(s.config.FallbackMethod)
	if method == "" {
		method = string(MethodStatic)
	}
	if SelectionMethod(method) == MethodSessionAware {
		method = string(MethodStatic)
	}
	result, err := Select(ctx, SelectionMethod(method), selCtx)
	if err != nil {
		return nil, err
	}
	if result == nil {
		return nil, fmt.Errorf("fallback selector %q returned no result", method)
	}
	result.Reasoning = fmt.Sprintf("%s; fallback=%s: %s", reason, method, result.Reasoning)
	return result, nil
}

func (s *SessionAwareSelector) scoreCandidate(selCtx *SelectionContext, currentModel string, candidate *config.ModelRef) (float64, string) {
	qualityGap := s.qualityGap(selCtx, currentModel, candidate.Model)
	handoffPenalty := s.handoffPenalty(selCtx, currentModel, candidate.Model)
	remainingTurns := s.remainingTurns(selCtx)

	score := s.config.QualityGapMultiplier * qualityGap
	reasons := []string{fmt.Sprintf("quality_gap=%.4f", qualityGap)}

	if matchesCurrentModel(*candidate, currentModel) {
		stayScore := s.config.StayBias + s.config.RemainingTurnWeight*remainingTurns + clampScore(selCtx.CacheWarmthEstimate, 0, 1)
		score += stayScore
		reasons = append(reasons,
			fmt.Sprintf("stay_bias=%.4f", s.config.StayBias),
			fmt.Sprintf("remaining_turns=%.4f", remainingTurns),
			fmt.Sprintf("cache_warmth=%.4f", clampScore(selCtx.CacheWarmthEstimate, 0, 1)),
		)
	} else {
		penalty := s.config.HandoffPenaltyWeight * handoffPenalty
		score -= penalty
		reasons = append(reasons, fmt.Sprintf("handoff_penalty=-%.4f", penalty))
	}

	return score, strings.Join(reasons, ", ")
}

func (s *SessionAwareSelector) qualityGap(selCtx *SelectionContext, currentModel, candidateModel string) float64 {
	if candidateModel == "" || currentModel == "" || currentModel == candidateModel {
		return 0
	}
	if selCtx.QualityGapByCandidate != nil {
		if value, ok := selCtx.QualityGapByCandidate[candidateModel]; ok {
			return value
		}
	}
	if s.lookupTable == nil {
		return 0
	}
	taskFamily := sessionTaskFamily(selCtx)
	if taskFamily == "" {
		return 0
	}
	value, ok := s.lookupTable.QualityGap(taskFamily, currentModel, candidateModel)
	if !ok {
		return 0
	}
	return value
}

func (s *SessionAwareSelector) handoffPenalty(selCtx *SelectionContext, currentModel, candidateModel string) float64 {
	if candidateModel == "" || currentModel == "" || currentModel == candidateModel {
		return 0
	}
	if selCtx.HandoffPenaltyByCandidate != nil {
		if value, ok := selCtx.HandoffPenaltyByCandidate[candidateModel]; ok {
			return value
		}
	}
	if s.lookupTable == nil {
		return 0
	}
	value, ok := s.lookupTable.HandoffPenalty(currentModel, candidateModel)
	if !ok {
		return 0
	}
	return value
}

func (s *SessionAwareSelector) remainingTurns(selCtx *SelectionContext) float64 {
	if selCtx.RemainingTurnsEstimate > 0 {
		return selCtx.RemainingTurnsEstimate
	}
	if s.lookupTable == nil {
		return 0
	}
	taskFamily := sessionTaskFamily(selCtx)
	if taskFamily == "" {
		return 0
	}
	prior, ok := s.lookupTable.RemainingTurnPrior(taskFamily)
	if !ok {
		return 0
	}
	remaining := prior - float64(selCtx.TurnIndex)
	if remaining < 0 {
		return 0
	}
	return remaining
}

func sessionTaskFamily(selCtx *SelectionContext) string {
	if selCtx == nil {
		return ""
	}
	if strings.TrimSpace(selCtx.CategoryName) != "" {
		return strings.TrimSpace(selCtx.CategoryName)
	}
	return strings.TrimSpace(selCtx.DecisionName)
}

func sessionCurrentModel(selCtx *SelectionContext) string {
	if selCtx == nil {
		return ""
	}
	for _, candidate := range []string{selCtx.CurrentModel, selCtx.PreviousModel, selCtx.OriginalModel} {
		if trimmed := strings.TrimSpace(candidate); trimmed != "" {
			return trimmed
		}
	}
	return ""
}

func findCandidateByCurrentModel(candidates []config.ModelRef, currentModel string) *config.ModelRef {
	for i := range candidates {
		if matchesCurrentModel(candidates[i], currentModel) {
			return &candidates[i]
		}
	}
	return nil
}

func matchesCurrentModel(candidate config.ModelRef, currentModel string) bool {
	return strings.TrimSpace(candidate.Model) == strings.TrimSpace(currentModel) ||
		(strings.TrimSpace(candidate.LoRAName) != "" && strings.TrimSpace(candidate.LoRAName) == strings.TrimSpace(currentModel))
}

func clampScore(value, minValue, maxValue float64) float64 {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}
