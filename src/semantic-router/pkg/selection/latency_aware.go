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
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/latency"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// LatencyAwareConfig configures the latency-aware selector.
// This is currently a placeholder for future selector-level tuning options.
type LatencyAwareConfig struct{}

// DefaultLatencyAwareConfig returns the default LatencyAware configuration.
func DefaultLatencyAwareConfig() *LatencyAwareConfig {
	return &LatencyAwareConfig{}
}

// LatencyAwareSelector selects models based on TPOT/TTFT percentile statistics.
// Lower percentile values are considered better (faster latency).
type LatencyAwareSelector struct {
	config *LatencyAwareConfig
}

// NewLatencyAwareSelector creates a new latency-aware selector.
func NewLatencyAwareSelector(cfg *LatencyAwareConfig) *LatencyAwareSelector {
	if cfg == nil {
		cfg = DefaultLatencyAwareConfig()
	}
	return &LatencyAwareSelector{
		config: cfg,
	}
}

// Method returns the selection method type.
func (s *LatencyAwareSelector) Method() SelectionMethod {
	return MethodLatencyAware
}

type latencyCandidateScore struct {
	modelRef *config.ModelRef
	tpot     float64
	ttft     float64
}

var ErrLatencyAwarePercentileInvalid = errors.New("latency-aware percentile must be between 1 and 100")

// Select chooses the best model based on configured TPOT/TTFT percentiles.
// If percentile data is missing for all candidates, uses the first candidate as
// the default candidate.
func (s *LatencyAwareSelector) Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
	_ = ctx

	if err := ValidateSelectionContext(selCtx); err != nil {
		return nil, err
	}
	if err := validateLatencyAwarePercentiles(selCtx); err != nil {
		return nil, err
	}

	hasTPOT := selCtx.LatencyAwareTPOTPercentile > 0
	hasTTFT := selCtx.LatencyAwareTTFTPercentile > 0
	if !hasTPOT && !hasTTFT {
		logging.Warnf("[LatencyAwareSelector] Missing TPOT/TTFT percentile configuration, using first candidate as default")
		return s.defaultToFirst(selCtx, "Latency-aware percentile config missing; using first candidate as default"), nil
	}

	candidates, minTPOT, minTTFT := collectLatencyCandidates(selCtx, hasTPOT, hasTTFT)
	if len(candidates) == 0 {
		logging.Warnf("[LatencyAwareSelector] No latency percentile data for candidates=%v, using first candidate as default", getModelNames(selCtx.CandidateModels))
		return s.defaultToFirst(selCtx, "No latency stats available; using first candidate as default"), nil
	}

	scored := scoreLatencyCandidates(candidates, latencyScoreOptions{
		hasTPOT: hasTPOT,
		hasTTFT: hasTTFT,
		minTPOT: minTPOT,
		minTTFT: minTTFT,
	})
	if !scored.ok {
		logging.Warnf("[LatencyAwareSelector] Failed to score candidates=%v, using first candidate as default", getModelNames(selCtx.CandidateModels))
		return s.defaultToFirst(selCtx, "Latency-aware scoring failed; using first candidate as default"), nil
	}

	confidence := latencyConfidence(scored.bestScore, scored.secondBestScore)
	reasoning := latencyReasoning(hasTPOT, hasTTFT, selCtx.LatencyAwareTPOTPercentile, selCtx.LatencyAwareTTFTPercentile)
	logging.Infof("[LatencyAwareSelector] Candidates=%v -> selected=%s (score=%.4f, confidence=%.2f)",
		getModelNames(selCtx.CandidateModels), scored.best.modelRef.Model, scored.bestScore, confidence)

	return &SelectionResult{
		SelectedModel: scored.best.modelRef.Model,
		LoRAName:      scored.best.modelRef.LoRAName,
		Score:         scored.bestScore,
		Confidence:    confidence,
		Method:        MethodLatencyAware,
		Reasoning:     reasoning,
		AllScores:     scored.allScores,
	}, nil
}

func validateLatencyAwarePercentiles(selCtx *SelectionContext) error {
	for _, percentile := range []struct {
		name  string
		value int
	}{
		{name: "tpot_percentile", value: selCtx.LatencyAwareTPOTPercentile},
		{name: "ttft_percentile", value: selCtx.LatencyAwareTTFTPercentile},
	} {
		if percentile.value < 0 || percentile.value > 100 {
			return fmt.Errorf("%w: %s=%d", ErrLatencyAwarePercentileInvalid, percentile.name, percentile.value)
		}
	}
	return nil
}

func collectLatencyCandidates(selCtx *SelectionContext, hasTPOT, hasTTFT bool) ([]latencyCandidateScore, float64, float64) {
	candidates := make([]latencyCandidateScore, 0, len(selCtx.CandidateModels))
	minTPOT := math.MaxFloat64
	minTTFT := math.MaxFloat64

	for i := range selCtx.CandidateModels {
		candidate, ok := latencyCandidateFromRef(&selCtx.CandidateModels[i], selCtx, hasTPOT, hasTTFT)
		if !ok {
			continue
		}
		if hasTPOT && candidate.tpot < minTPOT {
			minTPOT = candidate.tpot
		}
		if hasTTFT && candidate.ttft < minTTFT {
			minTTFT = candidate.ttft
		}
		candidates = append(candidates, candidate)
	}

	return candidates, minTPOT, minTTFT
}

func latencyCandidateFromRef(ref *config.ModelRef, selCtx *SelectionContext, hasTPOT, hasTTFT bool) (latencyCandidateScore, bool) {
	model := strings.TrimSpace(ref.Model)
	if model == "" {
		return latencyCandidateScore{}, false
	}

	candidate := latencyCandidateScore{modelRef: ref}
	if hasTPOT {
		tpotValue, ok := latency.GetTPOTPercentile(model, selCtx.LatencyAwareTPOTPercentile)
		if !ok {
			return latencyCandidateScore{}, false
		}
		candidate.tpot = tpotValue
	}

	if hasTTFT {
		ttftValue, ok := latency.GetTTFTPercentile(model, selCtx.LatencyAwareTTFTPercentile)
		if !ok {
			return latencyCandidateScore{}, false
		}
		candidate.ttft = ttftValue
	}

	return candidate, true
}

type latencyScoreOptions struct {
	hasTPOT bool
	hasTTFT bool
	minTPOT float64
	minTTFT float64
}

type latencyScoreResult struct {
	allScores       map[string]float64
	best            latencyCandidateScore
	bestScore       float64
	secondBestScore float64
	ok              bool
}

func scoreLatencyCandidates(candidates []latencyCandidateScore, opts latencyScoreOptions) latencyScoreResult {
	result := latencyScoreResult{
		allScores:       make(map[string]float64, len(candidates)),
		best:            candidates[0],
		bestScore:       math.MaxFloat64,
		secondBestScore: math.MaxFloat64,
	}

	for _, candidate := range candidates {
		score, ok := normalizedLatencyScore(candidate, opts)
		if !ok {
			continue
		}
		result.allScores[candidate.modelRef.Model] = score
		result.record(candidate, score)
	}
	result.ok = result.bestScore != math.MaxFloat64
	return result
}

func (r *latencyScoreResult) record(candidate latencyCandidateScore, score float64) {
	if score < r.bestScore {
		r.secondBestScore = r.bestScore
		r.bestScore = score
		r.best = candidate
		return
	}
	if score < r.secondBestScore {
		r.secondBestScore = score
	}
}

func normalizedLatencyScore(candidate latencyCandidateScore, opts latencyScoreOptions) (float64, bool) {
	score := 0.0
	parts := 0
	if opts.hasTPOT {
		score += candidate.tpot / positiveDenominator(opts.minTPOT)
		parts++
	}
	if opts.hasTTFT {
		score += candidate.ttft / positiveDenominator(opts.minTTFT)
		parts++
	}
	if parts == 0 {
		return 0, false
	}
	return score / float64(parts), true
}

func positiveDenominator(value float64) float64 {
	if value <= 0 {
		return 1.0
	}
	return value
}

func latencyConfidence(bestScore, secondBestScore float64) float64 {
	if secondBestScore == math.MaxFloat64 || secondBestScore <= 0 {
		return 1.0
	}
	gap := secondBestScore - bestScore
	if gap < 0 {
		gap = 0
	}
	return gap / secondBestScore
}

func latencyReasoning(hasTPOT, hasTTFT bool, tpotPercentile, ttftPercentile int) string {
	if hasTPOT && hasTTFT {
		return fmt.Sprintf("Latency-aware selection using TPOT p%d + TTFT p%d percentiles", tpotPercentile, ttftPercentile)
	}
	if hasTPOT {
		return fmt.Sprintf("Latency-aware selection using TPOT p%d percentile", tpotPercentile)
	}
	return fmt.Sprintf("Latency-aware selection using TTFT p%d percentile", ttftPercentile)
}

func (s *LatencyAwareSelector) defaultToFirst(selCtx *SelectionContext, reason string) *SelectionResult {
	first := selCtx.CandidateModels[0]
	return &SelectionResult{
		SelectedModel: first.Model,
		LoRAName:      first.LoRAName,
		Score:         1.0,
		Confidence:    0.0,
		Method:        MethodLatencyAware,
		Reasoning:     reason,
		AllScores: map[string]float64{
			first.Model: 1.0,
		},
	}
}

// UpdateFeedback is a no-op for latency-aware selection.
func (s *LatencyAwareSelector) UpdateFeedback(ctx context.Context, feedback *Feedback) error {
	_ = ctx
	_ = feedback
	return nil
}
