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

// Select chooses the best model based on configured TPOT/TTFT percentiles.
// If percentile data is missing for all candidates, falls back to the first candidate.
func (s *LatencyAwareSelector) Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
	_ = ctx

	if len(selCtx.CandidateModels) == 0 {
		return nil, fmt.Errorf("no candidate models provided")
	}

	hasTPOT := selCtx.LatencyAwareTPOTPercentile > 0
	hasTTFT := selCtx.LatencyAwareTTFTPercentile > 0
	if !hasTPOT && !hasTTFT {
		logging.Warnf("[LatencyAwareSelector] Missing TPOT/TTFT percentile configuration, using first candidate")
		return s.fallbackToFirst(selCtx, "Latency-aware percentile config missing, using first candidate"), nil
	}

	candidates := make([]latencyCandidateScore, 0, len(selCtx.CandidateModels))
	minTPOT := math.MaxFloat64
	minTTFT := math.MaxFloat64

	for i := range selCtx.CandidateModels {
		ref := &selCtx.CandidateModels[i]
		model := strings.TrimSpace(ref.Model)
		if model == "" {
			continue
		}

		candidate := latencyCandidateScore{
			modelRef: ref,
		}

		if hasTPOT {
			tpotValue, ok := latency.GetTPOTPercentile(model, selCtx.LatencyAwareTPOTPercentile)
			if !ok {
				continue
			}
			candidate.tpot = tpotValue
		}

		if hasTTFT {
			ttftValue, ok := latency.GetTTFTPercentile(model, selCtx.LatencyAwareTTFTPercentile)
			if !ok {
				continue
			}
			candidate.ttft = ttftValue
		}

		if hasTPOT && candidate.tpot < minTPOT {
			minTPOT = candidate.tpot
		}
		if hasTTFT && candidate.ttft < minTTFT {
			minTTFT = candidate.ttft
		}

		candidates = append(candidates, candidate)
	}

	if len(candidates) == 0 {
		logging.Warnf("[LatencyAwareSelector] No latency percentile data for candidates=%v, using first candidate", getModelNames(selCtx.CandidateModels))
		return s.fallbackToFirst(selCtx, "No latency stats available, using first candidate"), nil
	}

	allScores := make(map[string]float64, len(candidates))
	best := candidates[0]
	bestScore := math.MaxFloat64
	secondBestScore := math.MaxFloat64

	for _, candidate := range candidates {
		score := 0.0
		parts := 0

		if hasTPOT {
			denominator := minTPOT
			if denominator <= 0 {
				denominator = 1.0
			}
			score += candidate.tpot / denominator
			parts++
		}

		if hasTTFT {
			denominator := minTTFT
			if denominator <= 0 {
				denominator = 1.0
			}
			score += candidate.ttft / denominator
			parts++
		}

		if parts == 0 {
			continue
		}

		score = score / float64(parts)
		allScores[candidate.modelRef.Model] = score

		if score < bestScore {
			secondBestScore = bestScore
			bestScore = score
			best = candidate
		} else if score < secondBestScore {
			secondBestScore = score
		}
	}

	if bestScore == math.MaxFloat64 {
		logging.Warnf("[LatencyAwareSelector] Failed to score candidates=%v, using first candidate", getModelNames(selCtx.CandidateModels))
		return s.fallbackToFirst(selCtx, "Latency-aware scoring failed, using first candidate"), nil
	}

	confidence := 1.0
	if secondBestScore < math.MaxFloat64 && secondBestScore > 0 {
		gap := secondBestScore - bestScore
		if gap < 0 {
			gap = 0
		}
		confidence = gap / secondBestScore
	}

	reasoning := latencyReasoning(hasTPOT, hasTTFT, selCtx.LatencyAwareTPOTPercentile, selCtx.LatencyAwareTTFTPercentile)
	logging.Infof("[LatencyAwareSelector] Candidates=%v -> selected=%s (score=%.4f, confidence=%.2f)",
		getModelNames(selCtx.CandidateModels), best.modelRef.Model, bestScore, confidence)

	return &SelectionResult{
		SelectedModel: best.modelRef.Model,
		LoRAName:      best.modelRef.LoRAName,
		Score:         bestScore,
		Confidence:    confidence,
		Method:        MethodLatencyAware,
		Reasoning:     reasoning,
		AllScores:     allScores,
	}, nil
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

func (s *LatencyAwareSelector) fallbackToFirst(selCtx *SelectionContext, reason string) *SelectionResult {
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
