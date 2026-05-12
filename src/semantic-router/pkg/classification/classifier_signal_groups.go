package classification

import (
	"math"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/projectiontrace"
)

func (c *Classifier) applySignalGroups(results *SignalResults) *SignalResults {
	if results == nil || len(c.Config.Projections.Partitions) == 0 {
		return results
	}

	results.MatchedDomainRules = c.applySignalGroupsForType(
		config.SignalTypeDomain,
		results.MatchedDomainRules,
		results.SignalConfidences,
		results,
	)
	results.MatchedEmbeddingRules = c.applySignalGroupsForType(
		config.SignalTypeEmbedding,
		results.MatchedEmbeddingRules,
		results.SignalConfidences,
		results,
	)

	return results
}

func (c *Classifier) applySignalOutputPolicies(results *SignalResults) *SignalResults {
	if results == nil {
		return nil
	}
	results.MatchedEmbeddingRules = c.limitMatchedSignalsByTopK(
		config.SignalTypeEmbedding,
		results.MatchedEmbeddingRules,
		results.SignalConfidences,
	)
	return results
}

func (c *Classifier) applySignalGroupsForType(
	signalType string,
	matched []string,
	confidences map[string]float64,
	results *SignalResults,
) []string {
	result := matched
	for _, group := range c.Config.Projections.Partitions {
		members := c.signalGroupMembersForType(group, signalType)
		if len(members) <= 1 {
			continue
		}
		result = applySignalGroupToMatches(signalType, group, members, result, confidences, results)
	}

	return result
}

func (c *Classifier) signalGroupMembersForType(group config.ProjectionPartition, signalType string) []string {
	exists := make(map[string]struct{})

	switch signalType {
	case config.SignalTypeDomain:
		for _, category := range c.Config.Categories {
			exists[category.Name] = struct{}{}
		}
	case config.SignalTypeEmbedding:
		for _, rule := range c.Config.EmbeddingRules {
			exists[rule.Name] = struct{}{}
		}
	default:
		return nil
	}

	var members []string
	for _, member := range group.Members {
		if _, ok := exists[member]; ok {
			members = append(members, member)
		}
	}

	return members
}

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

func appendPartitionTraceEntry(results *SignalResults, entry projectiontrace.PartitionResolution) {
	if results == nil {
		return
	}
	if results.ProjectionTrace == nil {
		results.ProjectionTrace = &projectiontrace.Trace{SchemaVersion: projectiontrace.SchemaVersion}
	}
	results.ProjectionTrace.Partitions = append(results.ProjectionTrace.Partitions, entry)
}

func appendPartitionDefaultTrace(results *SignalResults, signalType string, group config.ProjectionPartition) {
	appendPartitionTraceEntry(results, projectiontrace.PartitionResolution{
		GroupName:   group.Name,
		SignalType:  signalType,
		Semantics:   group.Semantics,
		Temperature: group.Temperature,
		Winner:      group.Default,
		DefaultUsed: true,
	})
}

func appendPartitionWinnerTrace(
	results *SignalResults,
	signalType string,
	group config.ProjectionPartition,
	contenders []string,
	confidences map[string]float64,
	winner string,
	winnerScore float64,
) {
	raw := make([]float64, len(contenders))
	for i, name := range contenders {
		raw[i] = confidences[signalConfidenceKey(signalType, name)]
	}
	entry := projectiontrace.PartitionResolution{
		GroupName:      group.Name,
		SignalType:     signalType,
		Semantics:      group.Semantics,
		Temperature:    group.Temperature,
		Winner:         winner,
		WinnerScore:    winnerScore,
		RawWinnerScore: rawScoreForName(signalType, winner, contenders, raw),
	}
	softmax := strings.EqualFold(group.Semantics, "softmax_exclusive")
	var norm []float64
	if softmax {
		norm = softmaxScores(raw, group.Temperature)
	}
	for i, name := range contenders {
		pc := projectiontrace.PartitionContender{Name: name, RawScore: raw[i]}
		if softmax && i < len(norm) {
			ns := norm[i]
			pc.NormalizedScore = &ns
		}
		entry.Contenders = append(entry.Contenders, pc)
	}
	if softmax && len(norm) > 0 {
		entry.Margin = topTwoMargin(norm)
	} else {
		entry.Margin = topTwoMargin(raw)
	}
	appendPartitionTraceEntry(results, entry)
}

func rawScoreForName(signalType, winner string, contenders []string, raw []float64) float64 {
	for i, name := range contenders {
		if name == winner && i < len(raw) {
			return raw[i]
		}
	}
	return 0
}

func topTwoMargin(values []float64) float64 {
	if len(values) < 2 {
		return 0
	}
	sorted := append([]float64(nil), values...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] > sorted[j] })
	return sorted[0] - sorted[1]
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

func (c *Classifier) limitMatchedSignalsByTopK(
	signalType string,
	matched []string,
	confidences map[string]float64,
) []string {
	if signalType != config.SignalTypeEmbedding || len(matched) == 0 {
		return matched
	}

	topK := 1
	if c != nil && c.Config != nil && c.Config.EmbeddingConfig.TopK != nil {
		topK = *c.Config.EmbeddingConfig.TopK
	}
	if topK == 0 || len(matched) <= topK {
		return matched
	}

	type rankedSignal struct {
		name  string
		score float64
	}
	ranked := make([]rankedSignal, 0, len(matched))
	for _, name := range matched {
		ranked = append(ranked, rankedSignal{
			name:  name,
			score: confidences[signalConfidenceKey(signalType, name)],
		})
	}

	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].score == ranked[j].score {
			return ranked[i].name < ranked[j].name
		}
		return ranked[i].score > ranked[j].score
	})

	keep := make(map[string]struct{}, topK)
	limited := make([]string, 0, topK)
	for _, entry := range ranked[:topK] {
		keep[entry.name] = struct{}{}
		limited = append(limited, entry.name)
	}

	for _, entry := range ranked[topK:] {
		delete(confidences, signalConfidenceKey(signalType, entry.name))
	}

	return limited
}
