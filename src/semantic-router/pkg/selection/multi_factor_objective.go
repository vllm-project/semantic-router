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
	"math"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func normalizeMultiFactorObjective(objective *MultiFactorObjective) {
	if objective.Strategy == "" {
		objective.Strategy = config.MultiFactorObjectiveWeighted
	}
	if objective.Strategy != config.MultiFactorObjectiveLexicographic {
		objective.Strategy = config.MultiFactorObjectiveWeighted
		objective.Priorities = nil
		return
	}
	if len(objective.Priorities) == 0 {
		objective.Priorities = []MultiFactorPriority{
			{Factor: config.MultiFactorFactorQuality},
			{Factor: config.MultiFactorFactorCost},
			{Factor: config.MultiFactorFactorLatency},
			{Factor: config.MultiFactorFactorLoad},
		}
	}
	for index := range objective.Priorities {
		objective.Priorities[index].Tolerance = math.Max(0, math.Min(1, objective.Priorities[index].Tolerance))
	}
}

func (s *MultiFactorSelector) chooseCandidate(
	signals []signalSet,
	mins, maxs extrema,
) (int, map[string]float64, float64, float64) {
	var bestIndex int
	var allScores map[string]float64
	var candidateScores []float64
	var bestScore, secondBest float64
	if s.config.Objective.Strategy == config.MultiFactorObjectiveLexicographic {
		bestIndex, allScores, candidateScores, bestScore, secondBest = s.chooseLexicographic(signals)
	} else {
		bestIndex, allScores, candidateScores, bestScore, secondBest = s.chooseWeighted(signals, mins, maxs)
	}
	if s.qualityRelevant() && signals[bestIndex].hasQ {
		bestRank := candidateRank{score: candidateScores[bestIndex], evidence: signals[bestIndex].evidence}
		for index, signal := range signals {
			if signal.hasQ {
				rank := candidateRank{score: candidateScores[index], evidence: signal.evidence}
				if rank.Compare(bestRank) > 0 {
					bestIndex, bestRank = index, rank
				}
			}
		}
	}
	return bestIndex, allScores, bestScore, secondBest
}

func (s *MultiFactorSelector) chooseWeighted(
	signals []signalSet,
	mins, maxs extrema,
) (int, map[string]float64, []float64, float64, float64) {
	allScores := make(map[string]float64, len(signals))
	candidateScores := make([]float64, len(signals))
	bestIndex := 0
	bestScore := math.Inf(-1)
	secondBest := math.Inf(-1)
	for index, signal := range signals {
		score := s.scoreCandidate(signal, mins, maxs)
		allScores[signal.scoreKey] = score
		candidateScores[index] = score
		if score > bestScore {
			secondBest = bestScore
			bestScore = score
			bestIndex = index
		} else if score > secondBest {
			secondBest = score
		}
	}
	return bestIndex, allScores, candidateScores, bestScore, secondBest
}

func (s *MultiFactorSelector) chooseLexicographic(
	signals []signalSet,
) (int, map[string]float64, []float64, float64, float64) {
	active := make([]int, len(signals))
	for index := range signals {
		active[index] = index
	}
	allScores := make(map[string]float64, len(signals))
	candidateScores := make([]float64, len(signals))
	denominator := float64(len(s.config.Objective.Priorities) + 1)
	for stage, priority := range s.config.Objective.Priorities {
		best := 0.0
		bestSet := false
		higherIsBetter := false
		values := make(map[int]float64, len(active))
		for _, index := range active {
			value, available, higher := factorValue(signals[index], priority.Factor)
			if !available {
				continue
			}
			values[index] = value
			higherIsBetter = higher
			if !bestSet || (higher && value > best) || (!higher && value < best) {
				best = value
				bestSet = true
			}
		}
		if len(values) == 0 {
			continue
		}
		retained := make([]int, 0, len(active))
		for _, index := range active {
			value, available := values[index]
			if available && withinRelativeTolerance(value, best, higherIsBetter, priority.Tolerance) {
				retained = append(retained, index)
				continue
			}
			allScores[signals[index].scoreKey] = float64(stage) / denominator
			candidateScores[index] = float64(stage) / denominator
		}
		active = retained
		if len(active) == 1 {
			break
		}
	}
	for _, index := range active {
		allScores[signals[index].scoreKey] = 1
		candidateScores[index] = 1
	}
	winner := active[0]
	secondBest := math.Inf(-1)
	for index := range signals {
		if index == winner {
			continue
		}
		secondBest = math.Max(secondBest, candidateScores[index])
	}
	return winner, allScores, candidateScores, 1, secondBest
}

func factorValue(signal signalSet, factor string) (float64, bool, bool) {
	switch factor {
	case config.MultiFactorFactorQuality:
		return signal.quality, signal.hasQ, true
	case config.MultiFactorFactorLatency:
		return signal.latency, signal.hasLat, false
	case config.MultiFactorFactorCost:
		return signal.cost, signal.hasCost, false
	case config.MultiFactorFactorLoad:
		return signal.load, true, false
	default:
		return 0, false, false
	}
}

func withinRelativeTolerance(value, best float64, higherIsBetter bool, tolerance float64) bool {
	margin := math.Abs(best) * tolerance
	if higherIsBetter {
		return value >= best-margin
	}
	return value <= best+margin
}

func multiFactorObjectiveDescription(objective MultiFactorObjective) string {
	if objective.Strategy != config.MultiFactorObjectiveLexicographic {
		return config.MultiFactorObjectiveWeighted
	}
	parts := make([]string, 0, len(objective.Priorities))
	for _, priority := range objective.Priorities {
		parts = append(parts, priority.Factor+"±"+formatTolerance(priority.Tolerance))
	}
	return config.MultiFactorObjectiveLexicographic + "[" + strings.Join(parts, ",") + "]"
}

func formatTolerance(value float64) string {
	if value == 0 {
		return "0"
	}
	return strings.TrimRight(strings.TrimRight(fmtFloat(value), "0"), ".")
}

func fmtFloat(value float64) string {
	return strconv.FormatFloat(value, 'f', 3, 64)
}
