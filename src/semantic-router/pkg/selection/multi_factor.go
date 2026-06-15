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

// MultiFactorConfig configures the multi_factor selector that composes raw
// quality / latency / cost / load signals into a single weighted score per
// candidate, with optional SLO hard ceilings that prune candidates before
// scoring. See issue #37.
type MultiFactorConfig struct {
	Weights           MultiFactorWeights
	SLO               MultiFactorSLO
	LatencyPercentile int
	OnNoCandidates    string
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
		Weights: MultiFactorWeights{
			Quality: 0.25,
			Latency: 0.25,
			Cost:    0.25,
			Load:    0.25,
		},
		LatencyPercentile: defaultMFLatencyPercentile,
		OnNoCandidates:    defaultMFOnNoCandidates,
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
	if cfg.LatencyPercentile <= 0 || cfg.LatencyPercentile > 100 {
		cfg.LatencyPercentile = defaultMFLatencyPercentile
	}
	if cfg.OnNoCandidates == "" {
		cfg.OnNoCandidates = defaultMFOnNoCandidates
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

// Select runs the SLO filter, then the weighted multi-signal scoring.
func (s *MultiFactorSelector) Select(_ context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
	if len(selCtx.CandidateModels) == 0 {
		return nil, fmt.Errorf("no candidate models provided")
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	kept, dropped := s.applySLOFilter(selCtx.CandidateModels)
	if len(kept) == 0 {
		return s.applyNoCandidatePolicy(selCtx, dropped)
	}

	signals := s.gatherSignals(kept)
	mins, maxs := signalExtrema(signals)
	allScores := make(map[string]float64, len(kept))

	var bestIdx int
	bestScore := math.Inf(-1)
	secondBest := math.Inf(-1)
	for i, sig := range signals {
		score := s.scoreCandidate(sig, mins, maxs)
		allScores[sig.model] = score
		if score > bestScore {
			secondBest = bestScore
			bestScore = score
			bestIdx = i
		} else if score > secondBest {
			secondBest = score
		}
	}

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
		"multi_factor: weights{q=%.2f l=%.2f c=%.2f L=%.2f} latency_p%d, kept=%d, dropped=%d",
		s.config.Weights.Quality, s.config.Weights.Latency,
		s.config.Weights.Cost, s.config.Weights.Load,
		s.config.LatencyPercentile, len(kept), len(dropped),
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

func (s *MultiFactorSelector) gatherSignals(candidates []config.ModelRef) []signalSet {
	out := make([]signalSet, 0, len(candidates))
	for _, c := range candidates {
		sig := signalSet{model: c.Model}
		if params, ok := s.modelParams[c.Model]; ok {
			if params.QualityScore > 0 {
				sig.quality = params.QualityScore
				sig.hasQ = true
			}
			if params.Pricing.PromptPer1M > 0 {
				sig.cost = params.Pricing.PromptPer1M
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

func (s *MultiFactorSelector) applySLOFilter(candidates []config.ModelRef) (kept, dropped []config.ModelRef) {
	for _, c := range candidates {
		if reason, drop := s.exceedsSLO(c.Model); drop {
			logging.Debugf("[MultiFactor] dropping %s by SLO: %s", c.Model, reason)
			dropped = append(dropped, c)
			continue
		}
		kept = append(kept, c)
	}
	return kept, dropped
}

func (s *MultiFactorSelector) exceedsSLO(model string) (string, bool) {
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
		if params, ok := s.modelParams[model]; ok && params.Pricing.PromptPer1M > slo.MaxCostPer1M {
			return fmt.Sprintf("cost=$%.2f>$%.2f per 1M", params.Pricing.PromptPer1M, slo.MaxCostPer1M), true
		}
	}
	if slo.MaxInflight > 0 {
		if got := s.getInflight(model); got > slo.MaxInflight {
			return fmt.Sprintf("inflight=%d>%d", got, slo.MaxInflight), true
		}
	}
	return "", false
}

func (s *MultiFactorSelector) applyNoCandidatePolicy(selCtx *SelectionContext, dropped []config.ModelRef) (*SelectionResult, error) {
	switch strings.ToLower(s.config.OnNoCandidates) {
	case "fail":
		return nil, fmt.Errorf("multi_factor: all %d candidates excluded by SLO", len(dropped))
	case "first":
		c := selCtx.CandidateModels[0]
		return s.noCandidateResult(c, "all_candidates_excluded_by_slo:first"), nil
	default:
		c := s.cheapestCandidate(selCtx.CandidateModels)
		return s.noCandidateResult(c, "all_candidates_excluded_by_slo:cheapest"), nil
	}
}

func (s *MultiFactorSelector) cheapestCandidate(candidates []config.ModelRef) config.ModelRef {
	best := candidates[0]
	bestCost := math.Inf(1)
	for _, c := range candidates {
		cost := math.Inf(1)
		if params, ok := s.modelParams[c.Model]; ok && params.Pricing.PromptPer1M > 0 {
			cost = params.Pricing.PromptPer1M
		}
		if cost < bestCost {
			bestCost = cost
			best = c
		}
	}
	return best
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

	if w.Quality > 0 && maxs.hasQ {
		score += w.Quality * normalizeDirect(sig.quality, mins.quality, maxs.quality, sig.hasQ)
	}
	if w.Latency > 0 && maxs.hasLat {
		score += w.Latency * normalizeInverted(sig.latency, mins.latency, maxs.latency, sig.hasLat)
	}
	if w.Cost > 0 && maxs.hasCost {
		score += w.Cost * normalizeInverted(sig.cost, mins.cost, maxs.cost, sig.hasCost)
	}
	if w.Load > 0 {
		score += w.Load * normalizeInverted(sig.load, mins.load, maxs.load, true)
	}
	return score
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
