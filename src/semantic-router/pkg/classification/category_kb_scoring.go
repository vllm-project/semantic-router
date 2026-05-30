package classification

import (
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (c *KnowledgeBaseClassifier) computeLabelScores(queryEmb []float32) map[string]prototypeBankScore {
	labelScores := make(map[string]prototypeBankScore, len(c.labels))
	for labelName, data := range c.labels {
		if data.Prototype == nil {
			continue
		}
		labelScores[labelName] = data.Prototype.score(queryEmb, defaultPrototypeScoreOptions(c.rule.PrototypeScoring))
	}
	return labelScores
}

func (c *KnowledgeBaseClassifier) effectiveThreshold(label string) float64 {
	if threshold, ok := c.rule.LabelThresholds[label]; ok {
		return float64(threshold)
	}
	return float64(c.rule.Threshold)
}

func (c *KnowledgeBaseClassifier) buildMatchedLabels(labelScores map[string]float64) []string {
	matched := make([]string, 0, len(labelScores))
	for label, score := range labelScores {
		if score >= c.effectiveThreshold(label) {
			matched = append(matched, label)
		}
	}
	sort.Strings(matched)
	return matched
}

func (c *KnowledgeBaseClassifier) computeGroupScores(labelScores map[string]float64) map[string]float64 {
	groupScores := make(map[string]float64, len(c.rule.Groups))
	for group, labels := range c.rule.Groups {
		best := 0.0
		for _, label := range labels {
			if score := labelScores[label]; score > best {
				best = score
			}
		}
		groupScores[group] = best
	}
	return groupScores
}

func (c *KnowledgeBaseClassifier) collectMatchedGroups(matchedLabels []string) []string {
	if len(c.rule.Groups) == 0 || len(matchedLabels) == 0 {
		return nil
	}
	labelSet := make(map[string]struct{}, len(matchedLabels))
	for _, label := range matchedLabels {
		labelSet[label] = struct{}{}
	}
	groups := make([]string, 0, len(c.rule.Groups))
	for group, labels := range c.rule.Groups {
		for _, label := range labels {
			if _, ok := labelSet[label]; ok {
				groups = append(groups, group)
				break
			}
		}
	}
	sort.Strings(groups)
	return groups
}

func bestScoredName(scores map[string]float64) (string, float64) {
	if len(scores) == 0 {
		return "", 0
	}
	names := make([]string, 0, len(scores))
	for name := range scores {
		names = append(names, name)
	}
	sort.Strings(names)
	bestName := ""
	bestScore := 0.0
	for _, name := range names {
		score := scores[name]
		if bestName == "" || score > bestScore {
			bestName = name
			bestScore = score
		}
	}
	return bestName, bestScore
}

func (c *KnowledgeBaseClassifier) computeMetricValues(labelScores, groupScores map[string]float64, bestScore, bestMatchedScore float64) map[string]float64 {
	values := map[string]float64{
		config.KBMetricBestScore:        bestScore,
		config.KBMetricBestMatchedScore: bestMatchedScore,
	}
	for _, metric := range c.rule.Metrics {
		if metric.Type != config.KBMetricTypeGroupMargin {
			continue
		}
		values[metric.Name] = groupScores[metric.PositiveGroup] - groupScores[metric.NegativeGroup]
	}
	return values
}

func (c *KnowledgeBaseClassifier) buildResultFromLabelScores(labelBankScores map[string]prototypeBankScore) *KBClassifyResult {
	labelScores := make(map[string]float64, len(labelBankScores))
	labelBestScores := make(map[string]float64, len(labelBankScores))
	labelSupportScores := make(map[string]float64, len(labelBankScores))
	for label, score := range labelBankScores {
		labelScores[label] = score.Score
		labelBestScores[label] = score.Best
		labelSupportScores[label] = score.Support
	}

	bestLabel, bestScore := bestScoredName(labelScores)
	runnerUpScore := 0.0
	if bestLabel != "" {
		for label, score := range labelScores {
			if label == bestLabel {
				continue
			}
			if score > runnerUpScore {
				runnerUpScore = score
			}
		}
	}

	matchedLabels := c.buildMatchedLabels(labelScores)
	matchedLabelScores := make(map[string]float64, len(matchedLabels))
	for _, label := range matchedLabels {
		matchedLabelScores[label] = labelScores[label]
	}
	bestMatchedLabel, bestMatchedScore := bestScoredName(matchedLabelScores)

	groupScores := c.computeGroupScores(labelScores)
	bestGroup, _ := bestScoredName(groupScores)
	matchedGroups := c.collectMatchedGroups(matchedLabels)
	matchedGroupScores := make(map[string]float64, len(matchedGroups))
	for _, group := range matchedGroups {
		matchedGroupScores[group] = groupScores[group]
	}
	bestMatchedGroup, _ := bestScoredName(matchedGroupScores)

	return &KBClassifyResult{
		BestLabel:             bestLabel,
		BestSimilarity:        bestScore,
		BestLabelMargin:       bestScore - runnerUpScore,
		BestMatchedLabel:      bestMatchedLabel,
		BestMatchedSimilarity: bestMatchedScore,
		BestGroup:             bestGroup,
		BestMatchedGroup:      bestMatchedGroup,
		MatchedLabels:         matchedLabels,
		MatchedGroups:         matchedGroups,
		LabelConfidences:      labelScores,
		LabelBestScores:       labelBestScores,
		LabelSupportScores:    labelSupportScores,
		GroupScores:           groupScores,
		MetricValues:          c.computeMetricValues(labelScores, groupScores, bestScore, bestMatchedScore),
	}
}
