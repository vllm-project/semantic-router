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
	scores := make(CandidateScores, len(selCtx.CandidateModels))
	for i, ref := range selCtx.CandidateModels {
		scores[i] = CandidateScore{Candidate: ref, Score: 1 - float64(i)*0.001}
	}
	return (&SelectionResult{
		Score: scores[0].Score, Confidence: 1, Method: MethodStatic,
		Tier: TierSupported, Reasoning: "static base selection",
	}).WithCandidate(selCtx.CandidateModels[0]).WithScores(scores)
}

func ensureScoresForCandidates(result *SelectionResult, candidates []config.ModelRef) {
	scores := result.ScoresFor(candidates)
	for i, ref := range candidates {
		if _, ok := scores.Get(ref); ok {
			continue
		}
		// This is the existing chooser-only policy fallback, not benchmark evidence.
		score := -float64(i) * 0.001
		if result.SelectedCandidate != nil && CandidateIdentity(*result.SelectedCandidate) == CandidateIdentity(ref) ||
			result.SelectedCandidate == nil && result.SelectedModel == ref.Model {
			score = result.Score
		}
		scores = append(scores, CandidateScore{Candidate: ref, Score: score})
	}
	*result = *result.WithScores(scores)
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

func candidateScoreConfidence(selected int, scores CandidateScores) float64 {
	second := math.Inf(-1)
	for i, row := range scores {
		if i != selected && row.Score > second {
			second = row.Score
		}
	}
	if math.IsInf(second, -1) {
		return 1
	}
	return clamp01(0.5 + math.Min(0.5, math.Max(0, scores[selected].Score-second)))
}
