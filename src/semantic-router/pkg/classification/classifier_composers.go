package classification

import (
	"slices"
	"strings"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// applySignalComposers applies composer filters to signals that depend on other signals
// This is executed after all signals are computed in parallel
func (c *Classifier) applySignalComposers(results *SignalResults) *SignalResults {
	// Filter complexity signals by composer conditions
	if len(results.MatchedComplexityRules) > 0 && len(c.Config.ComplexityRules) > 0 {
		results.MatchedComplexityRules = c.filterComplexityByComposer(
			results.MatchedComplexityRules,
			results,
		)
	}

	// Future: Add other signals' composer filtering here
	// if len(results.MatchedXxxRules) > 0 { ... }

	return results
}

// filterComplexityByComposer filters complexity rules based on their composer conditions
func (c *Classifier) filterComplexityByComposer(
	matchedRules []string,
	allSignals *SignalResults,
) []string {
	filtered := []string{}

	for _, matched := range matchedRules {
		// Parse rule name (e.g., "code_complexity:hard" -> "code_complexity")
		parts := strings.Split(matched, ":")
		if len(parts) != 2 {
			logging.Warnf("Invalid complexity rule format: %s", matched)
			continue
		}
		ruleName := parts[0]

		// Find the corresponding rule config
		var rule *config.ComplexityRule
		for i := range c.Config.ComplexityRules {
			if c.Config.ComplexityRules[i].Name == ruleName {
				rule = &c.Config.ComplexityRules[i]
				break
			}
		}

		if rule == nil {
			logging.Warnf("Complexity rule config not found: %s", ruleName)
			continue
		}

		// If no composer, keep the result (no filtering)
		if rule.Composer == nil {
			filtered = append(filtered, matched)
			logging.Debugf("Complexity rule '%s' has no composer, keeping result", matched)
			continue
		}

		// Evaluate composer conditions
		if c.evaluateComposer(rule.Composer, allSignals) {
			filtered = append(filtered, matched)
			logging.Infof("Complexity rule '%s' passed composer filter", matched)
		} else {
			logging.Infof("Complexity rule '%s' filtered out by composer", matched)
		}
	}

	return filtered
}

// evaluateComposer evaluates a composer rule tree against signal results.
// Returns true when the tree matches (allowing the complexity rule through the filter).
// A nil composer always returns true (no filter applied).
func (c *Classifier) evaluateComposer(
	composer *config.RuleNode,
	signals *SignalResults,
) bool {
	if composer == nil {
		return true
	}
	return c.evalComposerNode(*composer, signals)
}

// evalComposerNode recursively evaluates a RuleNode against signal results.
func (c *Classifier) evalComposerNode(
	node config.RuleNode,
	signals *SignalResults,
) bool {
	if node.IsLeaf() {
		return c.evalComposerLeaf(node.Type, node.Name, signals)
	}

	switch strings.ToUpper(node.Operator) {
	case "OR":
		for _, child := range node.Conditions {
			if c.evalComposerNode(child, signals) {
				return true
			}
		}
		return false
	case "NOT":
		// Strictly unary: negate the single child's result.
		if len(node.Conditions) != 1 {
			logging.Warnf("Composer NOT operator requires exactly 1 child, got %d — treating as false", len(node.Conditions))
			return false
		}
		return !c.evalComposerNode(node.Conditions[0], signals)
	default: // AND
		for _, child := range node.Conditions {
			if !c.evalComposerNode(child, signals) {
				return false
			}
		}
		return true
	}
}

// evalComposerLeaf evaluates a single signal reference against signal results.
func (c *Classifier) evalComposerLeaf(
	typ, name string,
	signals *SignalResults,
) bool {
	switch typ {
	case "keyword":
		return slices.Contains(signals.MatchedKeywordRules, name)
	case "embedding":
		return slices.Contains(signals.MatchedEmbeddingRules, name)
	case "domain":
		return slices.Contains(signals.MatchedDomainRules, name)
	case "fact_check":
		return slices.Contains(signals.MatchedFactCheckRules, name)
	case "user_feedback":
		return slices.Contains(signals.MatchedUserFeedbackRules, name)
	case "preference":
		return slices.Contains(signals.MatchedPreferenceRules, name)
	case "language":
		return slices.Contains(signals.MatchedLanguageRules, name)
	case "context":
		return slices.Contains(signals.MatchedContextRules, name)
	case "structure":
		return slices.Contains(signals.MatchedStructureRules, name)
	case "modality":
		return slices.Contains(signals.MatchedModalityRules, name)
	case "taxonomy":
		return slices.Contains(signals.MatchedTaxonomyRules, name)
	default:
		logging.Warnf("Unknown composer condition type: %s", typ)
		return false
	}
}

// GetQueryEmbedding returns the embedding vector for a query text as float64
// This is used by model selection algorithms for similarity-based selection
// Returns float64 for compatibility with numerical operations
func (c *Classifier) GetQueryEmbedding(text string) []float64 {
	if text == "" {
		return nil
	}

	// Use the candle binding to get the embedding
	// GetEmbedding returns ([]float32, error) with auto-detected dimension
	embedding32, err := candle_binding.GetEmbedding(text, 0)
	if err != nil {
		logging.Debugf("Failed to get query embedding: %v", err)
		return nil
	}

	// Convert float32 to float64 for numerical operations
	embedding64 := make([]float64, len(embedding32))
	for i, v := range embedding32 {
		embedding64[i] = float64(v)
	}

	return embedding64
}
