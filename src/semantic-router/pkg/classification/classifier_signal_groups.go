package classification

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

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
