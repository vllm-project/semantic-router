/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package selection

import (
	"context"
	"fmt"
	"math"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/latency"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// MultiFactorConfig configures the multi_factor selector that compares raw
// quality / latency / cost / load signals with either a weighted or ordered
// objective. Optional hard ceilings prune candidates before optimization.
type MultiFactorConfig struct {
	Objective          MultiFactorObjective
	Weights            MultiFactorWeights
	SLO                MultiFactorSLO
	QualityIndex       string
	QualityOnMissing   string
	QualityMinCoverage float64
	QualityMinScore    *float64
	LatencyPercentile  int
	OnNoCandidates     string
}

// MultiFactorObjective selects weighted or ordered lexicographic comparison.
type MultiFactorObjective struct {
	Strategy   string
	Priorities []MultiFactorPriority
}

// MultiFactorPriority retains candidates within a relative Tolerance of the
// best raw value for Factor before the next priority is applied.
type MultiFactorPriority struct {
	Factor    string
	Tolerance float64
}

// MultiFactorWeights are the per-signal weights in the scoring formula
// score = wQ*quality + wL*latency + wC*cost + wLoad*load.
// All values default to 0.25 (equal weighting). Negative values are clamped
// to zero. Weights are normalized to sum to 1 at selector construction.
type MultiFactorWeights struct {
	Quality float64
	Latency float64
	Cost    float64
	Load    float64
}

// MultiFactorSLO sets hard ceilings: any candidate exceeding a non-zero
// ceiling is removed before scoring. A zero value means "no ceiling".
type MultiFactorSLO struct {
	MaxTPOTMs    float64
	MaxTTFTMs    float64
	MaxCostPer1M float64
	MaxInflight  int
}

// Defaults captured for documentation + tests.
const (
	defaultMFLatencyPercentile = 95
	defaultMFOnNoCandidates    = "cheapest"
)

// DefaultMultiFactorConfig returns the balanced default (equal weights, no
// SLOs, p95 latency, "cheapest" fallback).
func DefaultMultiFactorConfig() *MultiFactorConfig {
	return &MultiFactorConfig{
		Objective: MultiFactorObjective{Strategy: config.MultiFactorObjectiveWeighted},
		Weights: MultiFactorWeights{
			Quality: 0.25,
			Latency: 0.25,
			Cost:    0.25,
			Load:    0.25,
		},
		LatencyPercentile: defaultMFLatencyPercentile,
		OnNoCandidates:    defaultMFOnNoCandidates,
		QualityOnMissing:  config.QualityEvidenceOnMissingDisable,
	}
}

// MultiFactorSelector implements MethodMultiFactor.
type MultiFactorSelector struct {
	config *MultiFactorConfig

	mu          sync.RWMutex
	modelParams map[string]config.ModelParams

	// Injectable for tests; production points at the real package globals.
	getInflight func(model string) int
	getTPOT     func(model string, percentile int) (float64, bool)
	getTTFT     func(model string, percentile int) (float64, bool)
}

// NewMultiFactorSelector builds a selector with the given config and the
// production signal sources. Weights are clamped non-negative and normalized
// to sum to 1.
func NewMultiFactorSelector(cfg *MultiFactorConfig) *MultiFactorSelector {
	if cfg == nil {
		cfg = DefaultMultiFactorConfig()
	}
	normalizeWeights(&cfg.Weights)
	normalizeMultiFactorObjective(&cfg.Objective)
	if cfg.LatencyPercentile <= 0 || cfg.LatencyPercentile > 100 {
		cfg.LatencyPercentile = defaultMFLatencyPercentile
	}
	if cfg.OnNoCandidates == "" {
		cfg.OnNoCandidates = defaultMFOnNoCandidates
	}
	if cfg.QualityOnMissing == "" {
		cfg.QualityOnMissing = config.QualityEvidenceOnMissingDisable
	}
	return &MultiFactorSelector{
		config:      cfg,
		modelParams: make(map[string]config.ModelParams),
		getInflight: inflight.Get,
		getTPOT:     latency.GetTPOTPercentile,
		getTTFT:     latency.GetTTFTPercentile,
	}
}

// Method returns MethodMultiFactor.
func (s *MultiFactorSelector) Method() SelectionMethod {
	return MethodMultiFactor
}

// InitializeFromConfig captures per-model quality + pricing for use by Select.
func (s *MultiFactorSelector) InitializeFromConfig(modelConfig map[string]config.ModelParams) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.modelParams = make(map[string]config.ModelParams, len(modelConfig))
	for k, v := range modelConfig {
		s.modelParams[k] = v
	}
}

// UpdateFeedback is a no-op: multi_factor uses live signals only.
func (s *MultiFactorSelector) UpdateFeedback(_ context.Context, _ *Feedback) error {
	return nil
}

// Select applies hard eligibility filters, then the configured objective.
func (s *MultiFactorSelector) Select(_ context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
	if len(selCtx.CandidateModels) == 0 {
		return nil, fmt.Errorf("no candidate models provided")
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	kept, dropped := s.applySLOFilter(selCtx.CandidateModels, selCtx)
	if len(kept) == 0 {
		return s.applyNoCandidatePolicy(selCtx, "slo", len(dropped))
	}

	signals := s.gatherSignals(kept, selCtx)
	kept, signals, qualityFloorExcluded := s.applyQualityFloor(kept, signals)
	if len(kept) == 0 {
		return s.applyNoCandidatePolicy(selCtx, "quality_floor", qualityFloorExcluded)
	}
	kept, signals, qualityExcluded, qualityDisabled := s.applyQualityEvidencePolicy(kept, signals)
	if len(kept) == 0 {
		return s.applyNoCandidatePolicy(selCtx, "quality_evidence", qualityExcluded)
	}
	mins, maxs := signalExtrema(signals)
	bestIdx, allScores, bestScore, secondBest := s.chooseCandidate(signals, mins, maxs)

	chosen := kept[bestIdx]
	confidence := 0.5
	if !math.IsInf(secondBest, -1) && bestScore > 0 {
		gap := bestScore - secondBest
		if gap < 0 {
			gap = 0
		}
		confidence = math.Min(1.0, gap/bestScore+0.5)
	} else if len(kept) == 1 {
		confidence = 1.0
	}

	reasoning := fmt.Sprintf(
		"multi_factor: objective=%s weights{q=%.2f l=%.2f c=%.2f L=%.2f} quality_index=%q quality_missing=%s quality_min_coverage=%.2f quality_disabled=%t latency_p%d, kept=%d, dropped=%d, quality_floor_excluded=%d, quality_excluded=%d",
		multiFactorObjectiveDescription(s.config.Objective),
		s.config.Weights.Quality, s.config.Weights.Latency,
		s.config.Weights.Cost, s.config.Weights.Load,
		s.config.QualityIndex, s.config.QualityOnMissing, s.config.QualityMinCoverage, qualityDisabled,
		s.config.LatencyPercentile, len(kept), len(dropped), qualityFloorExcluded, qualityExcluded,
	)

	logging.Infof("[MultiFactor] candidates=%d -> %s (score=%.4f confidence=%.2f, dropped_by_slo=%d)",
		len(selCtx.CandidateModels), chosen.Model, bestScore, confidence, len(dropped))

	return &SelectionResult{
		SelectedModel: chosen.Model,
		LoRAName:      chosen.LoRAName,
		Score:         bestScore,
		Confidence:    confidence,
		Method:        MethodMultiFactor,
		Tier:          TierSupported,
		Reasoning:     reasoning,
		AllScores:     allScores,
	}, nil
}

type signalSet struct {
	model   string
	quality float64
	hasQ    bool
	latency float64
	hasLat  bool
	cost    float64
	hasCost bool
	load    float64
}

func (s *MultiFactorSelector) gatherSignals(candidates []config.ModelRef, selCtx *SelectionContext) []signalSet {
	out := make([]signalSet, 0, len(candidates))
	for _, c := range candidates {
		sig := signalSet{model: c.Model}
		if params, ok := s.modelParams[c.Model]; ok {
			if result, available := params.EvidenceResultAt(s.config.QualityIndex, c.ReasoningEffort); available &&
				result.Coverage >= s.config.QualityMinCoverage {
				sig.quality = *result.Score
				sig.hasQ = true
			}
			if cost, available := estimatedRequestCost(params.Pricing, selCtx); available {
				sig.cost = cost
				sig.hasCost = true
			}
		}
		if v, ok := s.latencySignal(c.Model); ok {
			sig.latency = v
			sig.hasLat = true
		}
		sig.load = float64(s.getInflight(c.Model))
		out = append(out, sig)
	}
	return out
}

func (s *MultiFactorSelector) applyQualityFloor(
	candidates []config.ModelRef,
	signals []signalSet,
) ([]config.ModelRef, []signalSet, int) {
	if s.config.QualityMinScore == nil {
		return candidates, signals, 0
	}
	keptCandidates := make([]config.ModelRef, 0, len(candidates))
	keptSignals := make([]signalSet, 0, len(signals))
	excluded := 0
	for index, signal := range signals {
		if !signal.hasQ || signal.quality < *s.config.QualityMinScore {
			excluded++
			continue
		}
		keptCandidates = append(keptCandidates, candidates[index])
		keptSignals = append(keptSignals, signal)
	}
	return keptCandidates, keptSignals, excluded
}

func (s *MultiFactorSelector) applyQualityEvidencePolicy(
	candidates []config.ModelRef,
	signals []signalSet,
) ([]config.ModelRef, []signalSet, int, bool) {
	if !s.qualityRelevant() {
		return candidates, signals, 0, false
	}
	missing := 0
	for _, signal := range signals {
		if !signal.hasQ {
			missing++
		}
	}
	if missing == 0 {
		return candidates, signals, 0, false
	}
	if s.config.QualityOnMissing == config.QualityEvidenceOnMissingExclude {
		keptCandidates := make([]config.ModelRef, 0, len(candidates)-missing)
		keptSignals := make([]signalSet, 0, len(signals)-missing)
		for index, signal := range signals {
			if !signal.hasQ {
				continue
			}
			keptCandidates = append(keptCandidates, candidates[index])
			keptSignals = append(keptSignals, signal)
		}
		return keptCandidates, keptSignals, missing, false
	}
	for index := range signals {
		signals[index].hasQ = false
	}
	return candidates, signals, 0, true
}

func (s *MultiFactorSelector) qualityRelevant() bool {
	if s.config.Objective.Strategy != config.MultiFactorObjectiveLexicographic {
		return s.config.Weights.Quality > 0
	}
	for _, priority := range s.config.Objective.Priorities {
		if priority.Factor == config.MultiFactorFactorQuality {
			return true
		}
	}
	return false
}

// latencySignal returns a single representative latency (TPOT-prioritized) at
// the configured percentile. Returns ok=false when no observations exist.
func (s *MultiFactorSelector) latencySignal(model string) (float64, bool) {
	if v, ok := s.getTPOT(model, s.config.LatencyPercentile); ok {
		return v, true
	}
	if v, ok := s.getTTFT(model, s.config.LatencyPercentile); ok {
		return v, true
	}
	return 0, false
}

func (s *MultiFactorSelector) applySLOFilter(candidates []config.ModelRef, selCtx *SelectionContext) (kept, dropped []config.ModelRef) {
	for _, c := range candidates {
		if reason, drop := s.exceedsSLO(c.Model, selCtx); drop {
			logging.Debugf("[MultiFactor] dropping %s by SLO: %s", c.Model, reason)
			dropped = append(dropped, c)
			continue
		}
		kept = append(kept, c)
	}
	return kept, dropped
}

func (s *MultiFactorSelector) exceedsSLO(model string, selCtx *SelectionContext) (string, bool) {
	slo := s.config.SLO

	if slo.MaxTPOTMs > 0 {
		if v, ok := s.getTPOT(model, s.config.LatencyPercentile); ok && v*1000 > slo.MaxTPOTMs {
			return fmt.Sprintf("tpot_p%d=%.1fms>%.1fms", s.config.LatencyPercentile, v*1000, slo.MaxTPOTMs), true
		}
	}
	if slo.MaxTTFTMs > 0 {
		if v, ok := s.getTTFT(model, s.config.LatencyPercentile); ok && v*1000 > slo.MaxTTFTMs {
			return fmt.Sprintf("ttft_p%d=%.1fms>%.1fms", s.config.LatencyPercentile, v*1000, slo.MaxTTFTMs), true
		}
	}
	if slo.MaxCostPer1M > 0 {
		if params, ok := s.modelParams[model]; ok {
			if rate, available := effectiveCostPer1M(params.Pricing, selCtx); available && rate > slo.MaxCostPer1M {
				return fmt.Sprintf("cost=$%.2f>$%.2f per 1M", rate, slo.MaxCostPer1M), true
			}
		}
	}
	if slo.MaxInflight > 0 {
		if got := s.getInflight(model); got > slo.MaxInflight {
			return fmt.Sprintf("inflight=%d>%d", got, slo.MaxInflight), true
		}
	}
	return "", false
}

func (s *MultiFactorSelector) applyNoCandidatePolicy(selCtx *SelectionContext, cause string, excluded int) (*SelectionResult, error) {
	switch strings.ToLower(s.config.OnNoCandidates) {
	case "fail":
		return nil, fmt.Errorf(
			"%w: multi_factor excluded all %d candidates by %s",
			ErrNoEligibleCandidates,
			excluded,
			cause,
		)
	case "first":
		c := selCtx.CandidateModels[0]
		return s.noCandidateResult(c, "all_candidates_excluded_by_"+cause+":first"), nil
	default:
		c := s.cheapestCandidate(selCtx.CandidateModels, selCtx)
		return s.noCandidateResult(c, "all_candidates_excluded_by_"+cause+":cheapest"), nil
	}
}

func (s *MultiFactorSelector) cheapestCandidate(candidates []config.ModelRef, selCtx *SelectionContext) config.ModelRef {
	best := candidates[0]
	bestCost := math.Inf(1)
	for _, c := range candidates {
		cost := math.Inf(1)
		if params, ok := s.modelParams[c.Model]; ok {
			if estimate, available := estimatedRequestCost(params.Pricing, selCtx); available {
				cost = estimate
			}
		}
		if cost < bestCost {
			bestCost = cost
			best = c
		}
	}
	return best
}

func estimatedRequestCost(pricing config.ModelPricing, selCtx *SelectionContext) (float64, bool) {
	if !modelPricingConfigured(pricing) {
		return 0, false
	}
	inputTokens, outputTokens := selectionTokenCounts(selCtx)
	if inputTokens+outputTokens > 0 {
		cost := (float64(inputTokens)*pricing.PromptPer1M +
			float64(outputTokens)*pricing.CompletionPer1M) / 1_000_000
		return cost, true
	}
	if pricing.PromptPer1M > 0 {
		return pricing.PromptPer1M, true
	}
	if pricing.CompletionPer1M > 0 {
		return pricing.CompletionPer1M, true
	}
	return 0, true
}

func effectiveCostPer1M(pricing config.ModelPricing, selCtx *SelectionContext) (float64, bool) {
	if !modelPricingConfigured(pricing) {
		return 0, false
	}
	inputTokens, outputTokens := selectionTokenCounts(selCtx)
	if total := inputTokens + outputTokens; total > 0 {
		rate := (float64(inputTokens)*pricing.PromptPer1M +
			float64(outputTokens)*pricing.CompletionPer1M) / float64(total)
		return rate, true
	}
	return estimatedRequestCost(pricing, nil)
}

func modelPricingConfigured(pricing config.ModelPricing) bool {
	return strings.TrimSpace(pricing.Currency) != "" || pricing.PromptPer1M != 0 ||
		pricing.CompletionPer1M != 0 || pricing.CachedInputPer1M != 0 || pricing.CacheWritePer1M != nil
}

func selectionTokenCounts(selCtx *SelectionContext) (int, int) {
	if selCtx == nil {
		return 0, 0
	}
	return max(selCtx.InputTokens, 0), max(selCtx.ExpectedOutputTokens, 0)
}

func (s *MultiFactorSelector) noCandidateResult(c config.ModelRef, reason string) *SelectionResult {
	return &SelectionResult{
		SelectedModel: c.Model,
		LoRAName:      c.LoRAName,
		Score:         0,
		Confidence:    0.0,
		Method:        MethodMultiFactor,
		Tier:          TierSupported,
		Reasoning:     "multi_factor no-candidate policy: " + reason,
	}
}

type extrema struct {
	quality, latency, cost, load float64
	hasQ, hasLat, hasCost        bool
}

func signalExtrema(signals []signalSet) (mins, maxs extrema) {
	mins = extrema{quality: math.Inf(1), latency: math.Inf(1), cost: math.Inf(1), load: math.Inf(1)}
	maxs = extrema{quality: math.Inf(-1), latency: math.Inf(-1), cost: math.Inf(-1), load: math.Inf(-1)}
	for _, s := range signals {
		updateOptionalExtrema(&mins.quality, &maxs.quality, &mins.hasQ, &maxs.hasQ, s.quality, s.hasQ)
		updateOptionalExtrema(&mins.latency, &maxs.latency, &mins.hasLat, &maxs.hasLat, s.latency, s.hasLat)
		updateOptionalExtrema(&mins.cost, &maxs.cost, &mins.hasCost, &maxs.hasCost, s.cost, s.hasCost)
		updateExtrema(&mins.load, &maxs.load, s.load)
	}
	return mins, maxs
}

// updateExtrema folds value v into mandatory (min, max) accumulators.
func updateExtrema(min, max *float64, v float64) {
	if v < *min {
		*min = v
	}
	if v > *max {
		*max = v
	}
}

// updateOptionalExtrema folds value v into (min, max) accumulators only when
// the signal is present (ok == true), and marks the accumulators as having
// observed at least one value via the corresponding hasMin / hasMax flags.
func updateOptionalExtrema(min, max *float64, hasMin, hasMax *bool, v float64, ok bool) {
	if !ok {
		return
	}
	*hasMin, *hasMax = true, true
	updateExtrema(min, max, v)
}

// scoreCandidate computes the weighted score. Each component is normalized to
// [0, 1] across the surviving candidate set. Latency / cost / load are
// inverted because lower-is-better; quality is direct.
func (s *MultiFactorSelector) scoreCandidate(sig signalSet, mins, maxs extrema) float64 {
	w := s.config.Weights
	score := 0.0
	activeWeight := 0.0

	if w.Quality > 0 && maxs.hasQ && sig.hasQ {
		score += w.Quality * normalizeDirect(sig.quality, mins.quality, maxs.quality, sig.hasQ)
		activeWeight += w.Quality
	}
	if w.Latency > 0 && maxs.hasLat && sig.hasLat {
		score += w.Latency * normalizeInverted(sig.latency, mins.latency, maxs.latency, sig.hasLat)
		activeWeight += w.Latency
	}
	if w.Cost > 0 && maxs.hasCost && sig.hasCost {
		score += w.Cost * normalizeInverted(sig.cost, mins.cost, maxs.cost, sig.hasCost)
		activeWeight += w.Cost
	}
	if w.Load > 0 {
		score += w.Load * normalizeInverted(sig.load, mins.load, maxs.load, true)
		activeWeight += w.Load
	}
	if activeWeight == 0 {
		return 0
	}
	return score / activeWeight
}

// normalizeDirect maps [min, max] -> [0, 1] linearly. Missing observation
// yields 0 so signals without data don't dominate.
func normalizeDirect(v, min, max float64, ok bool) float64 {
	if !ok {
		return 0
	}
	if max-min <= 0 {
		return 0.5
	}
	return (v - min) / (max - min)
}

// normalizeInverted maps [min, max] -> [1, 0] linearly. Used for
// lower-is-better signals (latency, cost, load).
func normalizeInverted(v, min, max float64, ok bool) float64 {
	if !ok {
		return 0
	}
	if max-min <= 0 {
		return 0.5
	}
	return 1.0 - (v-min)/(max-min)
}

// normalizeWeights clamps negative weights to zero and rescales so the sum is
// 1. If all weights are zero, sets equal weights as a recoverable default.
func normalizeWeights(w *MultiFactorWeights) {
	if w.Quality < 0 {
		w.Quality = 0
	}
	if w.Latency < 0 {
		w.Latency = 0
	}
	if w.Cost < 0 {
		w.Cost = 0
	}
	if w.Load < 0 {
		w.Load = 0
	}
	sum := w.Quality + w.Latency + w.Cost + w.Load
	if sum <= 0 {
		w.Quality, w.Latency, w.Cost, w.Load = 0.25, 0.25, 0.25, 0.25
		return
	}
	w.Quality /= sum
	w.Latency /= sum
	w.Cost /= sum
	w.Load /= sum
}
