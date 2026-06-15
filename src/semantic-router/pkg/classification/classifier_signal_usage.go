package classification

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// getUsedSignals analyzes all decisions and returns which signals (type:name) are actually used.
// This allows us to skip evaluation of unused signals for performance optimization.
// Returns a map with keys in format "type:name" (e.g., "keyword:math_keywords").
func (c *Classifier) getUsedSignals() map[string]bool {
	usedSignals := make(map[string]bool)

	for _, decision := range c.Config.Decisions {
		c.analyzeRuleCombination(decision.Rules, usedSignals)
	}
	c.expandProjectionDependencies(usedSignals)

	return usedSignals
}

// collectSignalKeys adds signal keys for a slice of items using a name-extraction function.
func collectSignalKeys[T any](signals map[string]bool, signalType string, items []T, getName func(T) string) {
	for _, item := range items {
		signals[strings.ToLower(signalType+":"+getName(item))] = true
	}
}

// getAllSignalTypes returns a map containing all configured signal types.
// This is used when forceEvaluateAll is true to evaluate all signals regardless of decision usage.
func (c *Classifier) getAllSignalTypes() map[string]bool {
	allSignals := make(map[string]bool)

	collectSignalKeys(allSignals, config.SignalTypeKeyword, c.Config.KeywordRules, func(r config.KeywordRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeEmbedding, c.Config.EmbeddingRules, func(r config.EmbeddingRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeDomain, c.Config.Categories, func(r config.Category) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeFactCheck, c.Config.FactCheckRules, func(r config.FactCheckRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeUserFeedback, c.Config.UserFeedbackRules, func(r config.UserFeedbackRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeReask, c.Config.ReaskRules, func(r config.ReaskRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypePreference, c.Config.PreferenceRules, func(r config.PreferenceRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeLanguage, c.Config.LanguageRules, func(r config.LanguageRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeContext, c.Config.ContextRules, func(r config.ContextRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeStructure, c.Config.StructureRules, func(r config.StructureRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeComplexity, c.Config.ComplexityRules, func(r config.ComplexityRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeModality, c.Config.ModalityRules, func(r config.ModalityRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeAuthz, c.Config.GetRoleBindings(), func(rb config.RoleBinding) string { return rb.Role })
	collectSignalKeys(allSignals, config.SignalTypeJailbreak, c.Config.JailbreakRules, func(r config.JailbreakRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypePII, c.Config.PIIRules, func(r config.PIIRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeKB, c.Config.KBRules, func(r config.KBSignalRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeConversation, c.Config.ConversationRules, func(r config.ConversationRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeEvent, c.Config.EventRules, func(r config.EventRule) string { return r.Name })
	for _, mapping := range c.Config.Projections.Mappings {
		for _, output := range mapping.Outputs {
			allSignals[strings.ToLower(config.SignalTypeProjection+":"+output.Name)] = true
		}
	}
	c.expandProjectionDependencies(allSignals)

	return allSignals
}

// analyzeRuleCombination recursively traverses a rule tree to collect all referenced signals.
func (c *Classifier) analyzeRuleCombination(node config.RuleNode, usedSignals map[string]bool) {
	if node.IsLeaf() {
		t := strings.ToLower(strings.TrimSpace(node.Type))
		n := strings.ToLower(strings.TrimSpace(node.Name))
		usedSignals[t+":"+n] = true
		return
	}
	for _, child := range node.Conditions {
		c.analyzeRuleCombination(child, usedSignals)
	}
}

func (c *Classifier) expandProjectionDependencies(usedSignals map[string]bool) {
	if len(c.Config.Projections.Scores) == 0 || len(c.Config.Projections.Mappings) == 0 {
		return
	}

	scoreByName := make(map[string]config.ProjectionScore, len(c.Config.Projections.Scores))
	for _, score := range c.Config.Projections.Scores {
		scoreByName[score.Name] = score
	}

	sourceByOutput := make(map[string]string)
	for _, mapping := range c.Config.Projections.Mappings {
		for _, output := range mapping.Outputs {
			sourceByOutput[strings.ToLower(output.Name)] = mapping.Source
		}
	}

	for key := range usedSignals {
		if !strings.HasPrefix(key, config.SignalTypeProjection+":") {
			continue
		}
		outputName := strings.TrimPrefix(key, config.SignalTypeProjection+":")
		scoreName, ok := sourceByOutput[outputName]
		if !ok {
			continue
		}
		c.expandScoreInputs(scoreName, scoreByName, sourceByOutput, usedSignals, make(map[string]bool))
	}
}

func (c *Classifier) expandScoreInputs(
	scoreName string,
	scoreByName map[string]config.ProjectionScore,
	sourceByOutput map[string]string,
	usedSignals map[string]bool,
	visited map[string]bool,
) {
	if visited[scoreName] {
		return
	}
	visited[scoreName] = true

	score, ok := scoreByName[scoreName]
	if !ok {
		return
	}
	for _, input := range score.Inputs {
		if strings.EqualFold(input.Type, config.ProjectionInputKBMetric) {
			usedSignals[strings.ToLower(config.SignalTypeKB+":"+input.KB)] = true
			continue
		}
		if strings.EqualFold(input.Type, config.SignalTypeProjection) {
			dep := input.Name
			if strings.EqualFold(strings.TrimSpace(input.ValueSource), "confidence") {
				if src, ok := sourceByOutput[strings.ToLower(dep)]; ok {
					dep = src
				}
			}
			c.expandScoreInputs(dep, scoreByName, sourceByOutput, usedSignals, visited)
			continue
		}
		usedSignals[strings.ToLower(input.Type+":"+input.Name)] = true
	}
}

// isSignalTypeUsed checks if any signal of the given type is used in decisions.
func isSignalTypeUsed(usedSignals map[string]bool, signalType string) bool {
	normalizedType := strings.ToLower(strings.TrimSpace(signalType))
	prefix := normalizedType + ":"

	for key := range usedSignals {
		if strings.HasPrefix(strings.ToLower(strings.TrimSpace(key)), prefix) {
			return true
		}
	}
	return false
}
