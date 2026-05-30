package classification

import (
	"math"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func applySignalGroupToMatches(
	signalType string,
	group config.ProjectionPartition,
	groupMembers []string,
	matched []string,
	confidences map[string]float64,
	results *SignalResults,
) []string {
	memberSet := make(map[string]struct{}, len(groupMembers))
	for _, member := range groupMembers {
		memberSet[member] = struct{}{}
	}

	var contenders []string
	for _, name := range matched {
		if _, ok := memberSet[name]; ok {
			if _, hasScore := confidences[signalConfidenceKey(signalType, name)]; hasScore {
				contenders = append(contenders, name)
			}
		}
	}

	if len(contenders) == 0 {
		return applySignalGroupDefaultFallback(signalType, group, memberSet, matched, results)
	}

	if len(contenders) == 1 {
		return matched
	}

	winner, winnerScore := selectSignalGroupWinner(signalType, group, contenders, confidences)
	appendPartitionWinnerTrace(results, signalType, group, contenders, confidences, winner, winnerScore)

	filtered := make([]string, 0, len(matched)-len(contenders)+1)
	for _, name := range matched {
		if _, ok := memberSet[name]; !ok || name == winner {
			filtered = append(filtered, name)
			continue
		}
		delete(confidences, signalConfidenceKey(signalType, name))
	}

	confidences[signalConfidenceKey(signalType, winner)] = winnerScore

	logging.Debugf(
		"[Signal Groups] %s group %q reduced contenders %v to winner %q (score=%.4f, semantics=%s)",
		signalType,
		group.Name,
		contenders,
		winner,
		winnerScore,
		group.Semantics,
	)

	return filtered
}

func applySignalGroupDefaultFallback(
	signalType string,
	group config.ProjectionPartition,
	memberSet map[string]struct{},
	matched []string,
	results *SignalResults,
) []string {
	if group.Default == "" {
		return matched
	}
	if _, ok := memberSet[group.Default]; !ok {
		return matched
	}
	for _, name := range matched {
		if name == group.Default {
			return matched
		}
	}

	logging.Debugf(
		"[Signal Groups] %s group %q synthesized default member %q because no group members matched",
		signalType,
		group.Name,
		group.Default,
	)
	appendPartitionDefaultTrace(results, signalType, group)
	return append(matched, group.Default)
}

func selectSignalGroupWinner(
	signalType string,
	group config.ProjectionPartition,
	contenders []string,
	confidences map[string]float64,
) (string, float64) {
	scores := make([]float64, 0, len(contenders))
	winnerIndex := 0
	bestRawScore := -1.0

	for i, contender := range contenders {
		score := confidences[signalConfidenceKey(signalType, contender)]
		scores = append(scores, score)
		if score > bestRawScore {
			bestRawScore = score
			winnerIndex = i
		}
	}

	if !strings.EqualFold(group.Semantics, "softmax_exclusive") {
		return contenders[winnerIndex], scores[winnerIndex]
	}

	normalized := softmaxScores(scores, group.Temperature)
	return contenders[winnerIndex], normalized[winnerIndex]
}

func softmaxScores(scores []float64, temperature float64) []float64 {
	if len(scores) == 0 {
		return nil
	}
	if temperature <= 0 {
		temperature = 1.0
	}

	maxScore := scores[0]
	for _, score := range scores[1:] {
		if score > maxScore {
			maxScore = score
		}
	}

	expScores := make([]float64, len(scores))
	sum := 0.0
	for i, score := range scores {
		expScore := math.Exp((score - maxScore) / temperature)
		expScores[i] = expScore
		sum += expScore
	}

	if sum == 0 {
		return expScores
	}

	for i := range expScores {
		expScores[i] /= sum
	}

	return expScores
}

func signalConfidenceKey(signalType, name string) string {
	return signalType + ":" + name
}
