package selection

import (
	"math"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func secondsDuration(seconds int) time.Duration {
	if seconds <= 0 {
		return 0
	}
	return time.Duration(seconds) * time.Second
}

func staticBaseResult(selCtx *SelectionContext) *SelectionResult {
	ref := selCtx.CandidateModels[0]
	allScores := make(map[string]float64, len(selCtx.CandidateModels))
	for i, candidate := range selCtx.CandidateModels {
		score := 1.0
		if i > 0 {
			score = 1.0 - float64(i)*0.001
		}
		allScores[candidate.Model] = score
	}
	return &SelectionResult{
		SelectedModel: ref.Model,
		LoRAName:      ref.LoRAName,
		Score:         allScores[ref.Model],
		Confidence:    1.0,
		Method:        MethodStatic,
		Tier:          TierSupported,
		Reasoning:     "static base selection",
		AllScores:     allScores,
	}
}

func ensureScoresForCandidates(result *SelectionResult, candidates []config.ModelRef) {
	if result.AllScores == nil {
		result.AllScores = make(map[string]float64, len(candidates))
	}
	for i, candidate := range candidates {
		if _, ok := result.AllScores[candidate.Model]; ok {
			continue
		}
		score := 0.0
		if result.SelectedModel == candidate.Model {
			score = result.Score
		} else {
			score = -float64(i) * 0.001
		}
		result.AllScores[candidate.Model] = score
	}
}

func cloneScores(in map[string]float64) map[string]float64 {
	if in == nil {
		return nil
	}
	out := make(map[string]float64, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func bestCandidateByScore(candidates []config.ModelRef, scores map[string]float64) (*config.ModelRef, float64) {
	var best *config.ModelRef
	bestScore := math.Inf(-1)
	for i := range candidates {
		score, ok := scores[candidates[i].Model]
		if !ok || !isFinite(score) {
			continue
		}
		if best == nil || score > bestScore {
			best = &candidates[i]
			bestScore = score
		}
	}
	return best, bestScore
}

func adjustedConfidence(selected string, scores map[string]float64) float64 {
	best := scores[selected]
	second := math.Inf(-1)
	for model, score := range scores {
		if model == selected {
			continue
		}
		if score > second {
			second = score
		}
	}
	if math.IsInf(second, -1) {
		return 1
	}
	return clamp01(0.5 + math.Min(0.5, math.Max(0, best-second)))
}

func modelRefForName(candidates []config.ModelRef, model string) *config.ModelRef {
	for i := range candidates {
		if candidates[i].Model == model || candidates[i].LoRAName == model {
			return &candidates[i]
		}
	}
	return nil
}

func maxScore(scores map[string]float64) float64 {
	max := math.Inf(-1)
	for _, score := range scores {
		if score > max {
			max = score
		}
	}
	if math.IsInf(max, -1) {
		return 0
	}
	return max
}
