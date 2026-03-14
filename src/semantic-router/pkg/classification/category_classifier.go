package classification

import (
	"strings"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// matchDomainCategories returns the domain categories that exceed the configured
// threshold, using entropy analysis to decide between top-1 and multi-category output.
func (c *Classifier) matchDomainCategories(
	domainResult candle_binding.ClassResultWithProbs,
	topCategoryName string,
) []entropy.CategoryProbability {
	threshold := c.Config.CategoryModel.Threshold
	topMatch := domainResult.Confidence >= threshold && topCategoryName != ""

	if len(domainResult.Probabilities) == 0 {
		if topMatch {
			return []entropy.CategoryProbability{
				{Category: topCategoryName, Probability: domainResult.Confidence},
			}
		}
		return nil
	}

	entropyResult := entropy.AnalyzeEntropy(domainResult.Probabilities)
	logging.Infof("[Signal Computation] Domain entropy analysis: entropy=%.3f, normalized=%.3f, uncertainty=%s",
		entropyResult.Entropy, entropyResult.NormalizedEntropy, entropyResult.UncertaintyLevel)

	categoryNames := make([]string, len(domainResult.Probabilities))
	for i := range domainResult.Probabilities {
		if name, ok := c.CategoryMapping.GetCategoryFromIndex(i); ok {
			categoryNames[i] = c.translateMMLUToGeneric(name)
		}
	}

	var matched []entropy.CategoryProbability
	switch entropyResult.UncertaintyLevel {
	case "very_low", "low":
		if topMatch {
			matched = []entropy.CategoryProbability{
				{Category: topCategoryName, Probability: domainResult.Confidence},
			}
		}
	default:
		for i, prob := range domainResult.Probabilities {
			if prob >= threshold && categoryNames[i] != "" {
				matched = append(matched, entropy.CategoryProbability{
					Category:    categoryNames[i],
					Probability: prob,
				})
			}
		}
	}

	logging.Infof("[Signal Computation] Domain signal matched %d categories (uncertainty=%s)",
		len(matched), entropyResult.UncertaintyLevel)
	return matched
}

func (c *Classifier) buildCategoryNameMappings() {
	c.MMLUToGeneric = make(map[string]string)
	c.GenericToMMLU = make(map[string][]string)

	knownMMLU := make(map[string]bool)
	if c.CategoryMapping != nil {
		for _, label := range c.CategoryMapping.IdxToCategory {
			knownMMLU[strings.ToLower(label)] = true
		}
	}

	for _, cat := range c.Config.Categories {
		if len(cat.MMLUCategories) > 0 {
			for _, mmlu := range cat.MMLUCategories {
				key := strings.ToLower(mmlu)
				c.MMLUToGeneric[key] = cat.Name
				c.GenericToMMLU[cat.Name] = append(c.GenericToMMLU[cat.Name], mmlu)
			}
		} else {
			nameLower := strings.ToLower(cat.Name)
			if knownMMLU[nameLower] {
				c.MMLUToGeneric[nameLower] = cat.Name
				c.GenericToMMLU[cat.Name] = append(c.GenericToMMLU[cat.Name], cat.Name)
			}
		}
	}
}

// translateMMLUToGeneric translates an MMLU-Pro category to a generic category if mapping exists
func (c *Classifier) translateMMLUToGeneric(mmluCategory string) string {
	if mmluCategory == "" {
		return ""
	}
	if c.MMLUToGeneric == nil {
		return mmluCategory
	}
	if generic, ok := c.MMLUToGeneric[strings.ToLower(mmluCategory)]; ok {
		return generic
	}
	return mmluCategory
}
