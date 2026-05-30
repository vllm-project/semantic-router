package classification

import (
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

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
