package selection

import (
	"math"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/lookuptable"
)

const (
	ModelSwitchGateModeShadow  = "shadow"
	ModelSwitchGateModeEnforce = "enforce"
)

// ModelSwitchGate evaluates whether a session should stay on its current model
// or accept the selector's candidate model.
type ModelSwitchGate struct {
	config      config.ModelSwitchGateConfig
	lookupTable lookuptable.LookupTable
}

// ModelSwitchGateInput carries the selector decision plus session-aware evidence.
type ModelSwitchGateInput struct {
	SelectionContext *SelectionContext
	SelectionResult  *SelectionResult
	CurrentModel     string
	CandidateModel   string

	// CacheWarmth is the current-model cache warmth prior (0=cold, 1=warm).
	// Producers should compute this from latency history at request time, not
	// from a response-side TTFT estimate.
	CacheWarmth float64
	// CacheWarmthOK indicates whether CacheWarmth is backed by reliable evidence.
	// When false the gate records cache_warmth as a missing signal but still
	// evaluates evidence (cache warmth contributes zero penalty in that case).
	CacheWarmthOK bool
}

// ModelSwitchGateDecision is the auditable stay-vs-switch decision.
type ModelSwitchGateDecision struct {
	Enabled bool
	Mode    string

	WouldSwitch  bool
	EnforcedStay bool

	CurrentModel   string
	CandidateModel string
	FinalModel     string

	CurrentScore       float64
	CandidateScore     float64
	SelectorScoreDelta float64
	QualityGap         float64
	HandoffPenalty     float64
	CacheWarmth        float64
	CacheWarmthPenalty float64
	NetSwitchAdvantage float64
	// NetSwitchAdvantageOK is true when evidence collection ran and
	// NetSwitchAdvantage reflects a real comparison (not a fallback default).
	NetSwitchAdvantageOK bool

	StayCostEstimate   float64
	SwitchCostEstimate float64

	MissingSignals []string
	Reason         string
}

// NewModelSwitchGate creates a gate with optional lookup-backed evidence.
func NewModelSwitchGate(cfg config.ModelSwitchGateConfig, lookupTable lookuptable.LookupTable) *ModelSwitchGate {
	return &ModelSwitchGate{config: cfg, lookupTable: lookupTable}
}

// Evaluate returns a decision without mutating selector state.
func (g *ModelSwitchGate) Evaluate(input ModelSwitchGateInput) ModelSwitchGateDecision {
	candidateModel := candidateModelFromInput(input)
	decision := newModelSwitchGateDecision(g, input, candidateModel)
	if !decision.Enabled {
		decision.Reason = "disabled"
		return decision
	}

	blocking, informational := classifyModelSwitchSignals(input, candidateModel)
	if len(blocking) > 0 {
		return decision.withMissingFallback(append(blocking, informational...))
	}
	if input.CurrentModel == candidateModel {
		decision.Reason = "candidate_is_current_model"
		decision.FinalModel = input.CurrentModel
		decision.MissingSignals = append(decision.MissingSignals, informational...)
		return decision
	}
	if !candidateSetContains(input.SelectionContext.CandidateModels, input.CurrentModel) {
		return decision.withMissingFallback(append([]string{"previous_model_not_in_candidates"}, informational...))
	}

	evidence, evidenceMissing, ok := g.collectSwitchEvidence(input, candidateModel)
	if !ok {
		return decision.withMissingFallback(append([]string{"selector_score_or_quality_gap"}, informational...))
	}
	missing := append(informational, evidenceMissing...)
	decision.applyEvidence(evidence, missing, gateConfig(g))

	if decision.WouldSwitch {
		decision.Reason = "switch_advantage_exceeds_threshold"
		return decision
	}

	decision.Reason = "stay_cost_not_exceeded"
	if decision.Mode == ModelSwitchGateModeEnforce {
		decision.FinalModel = input.CurrentModel
		decision.EnforcedStay = true
	}
	return decision
}

type modelSwitchGateEvidence struct {
	CurrentScore       float64
	CandidateScore     float64
	SelectorScoreDelta float64
	QualityGap         float64
	HandoffPenalty     float64
	ScoreDeltaOK       bool
	QualityGapOK       bool
}

func candidateModelFromInput(input ModelSwitchGateInput) string {
	if input.CandidateModel != "" {
		return input.CandidateModel
	}
	if input.SelectionResult != nil {
		return input.SelectionResult.SelectedModel
	}
	return ""
}

func newModelSwitchGateDecision(g *ModelSwitchGate, input ModelSwitchGateInput, candidateModel string) ModelSwitchGateDecision {
	return ModelSwitchGateDecision{
		Enabled:        g != nil && g.config.Enabled,
		Mode:           normalizeModelSwitchGateMode(gateConfig(g).Mode),
		CurrentModel:   input.CurrentModel,
		CandidateModel: candidateModel,
		FinalModel:     candidateModel,
		CacheWarmth:    clamp01(input.CacheWarmth),
	}
}

// classifyModelSwitchSignals splits gate input signals into blocking (gate falls
// back without evaluating evidence) and informational (recorded but evidence
// still runs). previous_model is blocking because there is no current-model
// reference to compare against; session_id is informational so Chat Completions
// traffic without per-turn history can still be observed in shadow mode.
func classifyModelSwitchSignals(input ModelSwitchGateInput, candidateModel string) (blocking, informational []string) {
	if input.SelectionContext == nil {
		blocking = append(blocking, "selection_context")
	} else if input.SelectionContext.SessionID == "" {
		informational = append(informational, "session_id")
	}
	if input.CurrentModel == "" {
		blocking = append(blocking, "previous_model")
	}
	if candidateModel == "" {
		blocking = append(blocking, "candidate_model")
	}
	return blocking, informational
}

func (g *ModelSwitchGate) collectSwitchEvidence(input ModelSwitchGateInput, candidateModel string) (modelSwitchGateEvidence, []string, bool) {
	evidence := modelSwitchGateEvidence{}
	currentScore, currentScoreOK := scoreForModel(input.SelectionResult, input.CurrentModel)
	candidateScore, candidateScoreOK := scoreForModel(input.SelectionResult, candidateModel)
	if currentScoreOK && candidateScoreOK {
		evidence.CurrentScore = currentScore
		evidence.CandidateScore = candidateScore
		evidence.SelectorScoreDelta = candidateScore - currentScore
		evidence.ScoreDeltaOK = true
	}

	qualityGap, qualityGapOK := g.lookupQualityGap(input.SelectionContext, input.CurrentModel, candidateModel)
	if qualityGapOK {
		evidence.QualityGap = qualityGap
		evidence.QualityGapOK = true
	}
	if !evidence.ScoreDeltaOK && !evidence.QualityGapOK {
		return evidence, nil, false
	}

	missing := make([]string, 0, 2)
	handoffPenalty, handoffPenaltyOK := g.lookupHandoffPenalty(input.CurrentModel, candidateModel)
	if !handoffPenaltyOK {
		missing = append(missing, "handoff_penalty")
		handoffPenalty = gateConfig(g).DefaultHandoffPenalty
	}
	evidence.HandoffPenalty = handoffPenalty
	if !input.CacheWarmthOK {
		missing = append(missing, "cache_warmth")
	}
	return evidence, missing, true
}

func (d *ModelSwitchGateDecision) applyEvidence(
	evidence modelSwitchGateEvidence,
	missing []string,
	cfg config.ModelSwitchGateConfig,
) {
	if evidence.ScoreDeltaOK {
		d.CurrentScore = evidence.CurrentScore
		d.CandidateScore = evidence.CandidateScore
		d.SelectorScoreDelta = evidence.SelectorScoreDelta
		d.StayCostEstimate = boundedCostEstimate(evidence.CurrentScore)
		d.SwitchCostEstimate = boundedCostEstimate(evidence.CandidateScore)
	}
	if evidence.QualityGapOK {
		d.QualityGap = evidence.QualityGap
	}
	d.HandoffPenalty = evidence.HandoffPenalty
	d.CacheWarmthPenalty = d.CacheWarmth * cfg.CacheWarmthWeight
	d.NetSwitchAdvantage = evidence.switchBenefit() - d.HandoffPenalty - d.CacheWarmthPenalty
	d.NetSwitchAdvantageOK = true
	d.SwitchCostEstimate = clamp01(d.SwitchCostEstimate + d.HandoffPenalty + d.CacheWarmthPenalty)
	d.WouldSwitch = d.NetSwitchAdvantage > cfg.MinSwitchAdvantage
	d.MissingSignals = missing
}

// switchBenefit returns the quality benefit of switching to the candidate.
// QualityGap (lookup-table evidence keyed on task family) is preferred when
// available; SelectorScoreDelta is the fallback. The two are NOT summed —
// many selectors already incorporate quality into their score, so adding
// QualityGap on top would double-count.
func (e modelSwitchGateEvidence) switchBenefit() float64 {
	if e.QualityGapOK {
		return e.QualityGap
	}
	if e.ScoreDeltaOK {
		return e.SelectorScoreDelta
	}
	return 0
}

// LogFields returns stable structured fields for audit logs.
func (d ModelSwitchGateDecision) LogFields() map[string]interface{} {
	return map[string]interface{}{
		"enabled":                 d.Enabled,
		"mode":                    d.Mode,
		"would_switch":            d.WouldSwitch,
		"enforced_stay":           d.EnforcedStay,
		"current_model":           d.CurrentModel,
		"candidate_model":         d.CandidateModel,
		"final_model":             d.FinalModel,
		"current_score":           d.CurrentScore,
		"candidate_score":         d.CandidateScore,
		"selector_score_delta":    d.SelectorScoreDelta,
		"quality_gap":             d.QualityGap,
		"handoff_penalty":         d.HandoffPenalty,
		"cache_warmth":            d.CacheWarmth,
		"cache_warmth_penalty":    d.CacheWarmthPenalty,
		"net_switch_advantage":    d.NetSwitchAdvantage,
		"net_switch_advantage_ok": d.NetSwitchAdvantageOK,
		"stay_cost_estimate":      d.StayCostEstimate,
		"switch_cost_estimate":    d.SwitchCostEstimate,
		"missing_signals":         d.MissingSignals,
		"reason":                  d.Reason,
	}
}

func (d ModelSwitchGateDecision) withMissingFallback(missing []string) ModelSwitchGateDecision {
	d.MissingSignals = append(d.MissingSignals, missing...)
	d.Reason = "missing_signal_fallback"
	return d
}

func (g *ModelSwitchGate) lookupQualityGap(selCtx *SelectionContext, currentModel, candidateModel string) (float64, bool) {
	if g == nil || g.lookupTable == nil || selCtx == nil {
		return 0, false
	}
	if selCtx.CategoryName != "" {
		if gap, ok := g.lookupTable.QualityGap(selCtx.CategoryName, currentModel, candidateModel); ok {
			return gap, true
		}
	}
	if selCtx.DecisionName != "" && selCtx.DecisionName != selCtx.CategoryName {
		return g.lookupTable.QualityGap(selCtx.DecisionName, currentModel, candidateModel)
	}
	return 0, false
}

func (g *ModelSwitchGate) lookupHandoffPenalty(currentModel, candidateModel string) (float64, bool) {
	if g == nil || g.lookupTable == nil {
		return 0, false
	}
	return g.lookupTable.HandoffPenalty(currentModel, candidateModel)
}

func gateConfig(g *ModelSwitchGate) config.ModelSwitchGateConfig {
	if g == nil {
		return config.ModelSwitchGateConfig{}
	}
	return g.config
}

func normalizeModelSwitchGateMode(mode string) string {
	if mode == ModelSwitchGateModeEnforce {
		return ModelSwitchGateModeEnforce
	}
	return ModelSwitchGateModeShadow
}

func scoreForModel(result *SelectionResult, model string) (float64, bool) {
	if result == nil || model == "" {
		return 0, false
	}
	if result.AllScores != nil {
		if score, ok := result.AllScores[model]; ok && isFinite(score) {
			return score, true
		}
	}
	if result.SelectedModel == model && isFinite(result.Score) {
		return result.Score, true
	}
	return 0, false
}

func candidateSetContains(candidates []config.ModelRef, model string) bool {
	for _, candidate := range candidates {
		if candidate.Model == model || candidate.LoRAName == model {
			return true
		}
	}
	return false
}

func boundedCostEstimate(score float64) float64 {
	if !isFinite(score) {
		return 0
	}
	return 1 - clamp01(score)
}

func clamp01(v float64) float64 {
	if !isFinite(v) {
		return 0
	}
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func isFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}
