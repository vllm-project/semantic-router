package classification

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// otherDomainLabel is the classifier's catch-all label. A label no domain rule
// lists counts as it, so the rule that lists it is the domain fallback.
const otherDomainLabel = "other"

// matchDomainCategories returns the declared domain rules whose labels exceed
// the configured threshold, using entropy analysis to decide between top-1 and
// multi-category output. topLabel is the classifier's top label.
func (c *Classifier) matchDomainCategories(
	domainResult tasks.ClassResultWithProbs,
	topLabel string,
) []entropy.CategoryProbability {
	threshold := c.Config.CategoryModel.Threshold
	topRule, topDeclared := c.domainRuleForLabel(topLabel)
	topMatch := domainResult.Confidence >= threshold && topDeclared

	if len(domainResult.Probabilities) == 0 {
		if topMatch {
			return []entropy.CategoryProbability{
				{Category: topRule, Probability: domainResult.Confidence},
			}
		}
		return nil
	}

	entropyResult := entropy.AnalyzeEntropy(domainResult.Probabilities)
	logging.Debugf("[Signal Computation] Domain entropy analysis: entropy=%.3f, normalized=%.3f, uncertainty=%s",
		entropyResult.Entropy, entropyResult.NormalizedEntropy, entropyResult.UncertaintyLevel)

	var matched []entropy.CategoryProbability
	switch entropyResult.UncertaintyLevel {
	case "very_low", "low":
		if topMatch {
			matched = []entropy.CategoryProbability{
				{Category: topRule, Probability: domainResult.Confidence},
			}
		}
	default:
		matched = c.domainRulesAboveThreshold(domainResult.Probabilities, threshold)
	}

	logging.Debugf("[Signal Computation] Domain signal matched %d categories (uncertainty=%s)",
		len(matched), entropyResult.UncertaintyLevel)
	return matched
}

// domainRulesAboveThreshold returns each declared rule with a label at or
// above threshold, once, with the highest of its labels' probabilities.
func (c *Classifier) domainRulesAboveThreshold(probabilities []float32, threshold float32) []entropy.CategoryProbability {
	var matched []entropy.CategoryProbability
	position := map[string]int{}
	for i, prob := range probabilities {
		if prob < threshold {
			continue
		}
		label, ok := c.CategoryMapping.GetCategoryFromIndex(i)
		if !ok {
			continue
		}
		rule, declared := c.domainRuleForLabel(label)
		if !declared {
			continue
		}
		if at, seen := position[rule]; seen {
			matched[at].Probability = max(matched[at].Probability, prob)
			continue
		}
		position[rule] = len(matched)
		matched = append(matched, entropy.CategoryProbability{Category: rule, Probability: prob})
	}
	return matched
}

// domainRuleForLabel returns the declared domain rule a classifier label
// matches: the rule that lists it in mmlu_categories, or the rule named after
// it. A label no rule lists counts as "other"; without a rule for that, it
// matches nothing.
func (c *Classifier) domainRuleForLabel(label string) (string, bool) {
	key := strings.ToLower(strings.TrimSpace(label))
	if key == "" {
		return "", false
	}
	if rule, ok := c.MMLUToGeneric[key]; ok {
		return rule, true
	}
	rule, ok := c.MMLUToGeneric[otherDomainLabel]
	return rule, ok
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
