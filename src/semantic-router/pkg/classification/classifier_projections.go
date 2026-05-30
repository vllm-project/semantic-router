package classification

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

func (c *Classifier) applyProjections(results *SignalResults) *SignalResults {
	if results == nil {
		return nil
	}
	if len(c.Config.Projections.Scores) == 0 || len(c.Config.Projections.Mappings) == 0 {
		return results
	}

	if results.ProjectionScores == nil {
		results.ProjectionScores = make(map[string]float64)
	}
	if results.SignalConfidences == nil {
		results.SignalConfidences = make(map[string]float64)
	}

	orderedScores := topologicalScoreOrder(c.Config.Projections.Scores, c.Config.Projections.Mappings)

	mappingBySource := make(map[string][]config.ProjectionMapping, len(c.Config.Projections.Mappings))
	for _, mapping := range c.Config.Projections.Mappings {
		mappingBySource[mapping.Source] = append(mappingBySource[mapping.Source], mapping)
	}

	for _, score := range orderedScores {
		results.ProjectionScores[score.Name] = projectionScoreValue(score, results)

		for _, mapping := range mappingBySource[score.Name] {
			scoreValue := results.ProjectionScores[mapping.Source]
			applyProjectionMapping(mapping, scoreValue, results)
		}
	}

	results.ProjectionTrace = mergeProjectionTrace(results, c.Config.Projections)
	return results
}
