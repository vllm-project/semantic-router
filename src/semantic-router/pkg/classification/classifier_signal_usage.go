package classification

import (
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// getUsedSignals analyzes this classifier's isolated routing profile and
// returns which signals (type:name) are actually used. This allows us to skip
// evaluation of unused local signals.
// Returns a map with keys in format "type:name" (e.g., "keyword:math_keywords").
func (c *Classifier) getUsedSignals() map[string]bool {
	return c.getUsedSignalsForDecisions(c.Config.AllRoutingDecisions())
}

// getUsedSignalsForDecisions computes usage for an explicit local decision
// subset, such as a direct-looper algorithm view.
func (c *Classifier) getUsedSignalsForDecisions(decisions []config.Decision) map[string]bool {
	usedSignals := make(map[string]bool)

	for i := range decisions {
		c.analyzeRuleCombination(decisions[i].Rules, usedSignals)
		for _, condition := range c.Config.ResponseGuardRequestConditions(decisions[i].Plugins) {
			c.analyzeRuleCombination(condition, usedSignals)
		}
	}
	// A request that matches no decision runs the default plugins' guard.
	for _, condition := range c.Config.ResponseGuardRequestConditions(c.Config.DefaultPlugins) {
		c.analyzeRuleCombination(condition, usedSignals)
	}
	// Replay's privacy check reads the PII rules after routing (ReplayPromptEvidence), never here.
	c.expandTransitiveSignalDependencies(usedSignals)

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
	collectSignalKeys(allSignals, config.SignalTypeDomain, c.Config.Categories, func(r config.Category) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeFactCheck, c.Config.FactCheckRules, func(r config.FactCheckRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeUserFeedback, c.Config.UserFeedbackRules, func(r config.UserFeedbackRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeLanguage, c.Config.LanguageRules, func(r config.LanguageRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeContext, c.Config.ContextRules, func(r config.ContextRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeComplexity, c.Config.ComplexityRules, func(r config.ComplexityRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeModality, c.Config.ModalityRules, func(r config.ModalityRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeAuthz, c.Config.GetRoleBindings(), func(rb config.RoleBinding) string { return rb.Role })
	collectSignalKeys(allSignals, config.SignalTypeJailbreak, c.Config.JailbreakRules, func(r config.JailbreakRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeSafety, c.Config.SafetyRules, func(r config.SafetyRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypePII, c.Config.PIIRules, func(r config.PIIRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeTurnRelation, c.Config.TurnRelationRules, func(r config.TurnRelationRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeConversation, c.Config.ConversationRules, func(r config.ConversationRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeMetadata, c.Config.MetadataRules, func(r config.MetadataRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeInputModality, c.Config.InputModalityRules, func(r config.InputModalityRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeSystemOne, c.Config.SystemOneRules, func(r config.SystemOneRule) string { return r.Name })
	for _, mapping := range c.Config.Projections.Mappings {
		for _, output := range mapping.Outputs {
			allSignals[strings.ToLower(config.SignalTypeProjection+":"+output.Name)] = true
		}
	}
	c.expandTransitiveSignalDependencies(allSignals)

	return allSignals
}

func (c *Classifier) expandTransitiveSignalDependencies(usedSignals map[string]bool) {
	for {
		before := len(usedSignals)
		c.expandProjectionDependencies(usedSignals)
		c.expandPartitionMembers(usedSignals)
		if len(usedSignals) == before {
			return
		}
	}
}

// expandPartitionMembers marks every member of a partition used once one is:
// the partition picks its winner, or its default, among all of them.
func (c *Classifier) expandPartitionMembers(usedSignals map[string]bool) {
	key := func(member string) string { return strings.ToLower(config.SignalTypeDomain + ":" + member) }
	for _, partition := range c.Config.Projections.Partitions {
		if !slices.ContainsFunc(partition.Members, func(member string) bool { return usedSignals[key(member)] }) {
			continue
		}
		for _, member := range partition.Members {
			usedSignals[key(member)] = true
		}
	}
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
		if inputType := strings.ToLower(strings.TrimSpace(input.Type)); inputType == config.SignalTypeSystemOne || config.IsApplicationSignalType(inputType) {
			// "<rule>:<option>" reads one option's probability; the rule
			// itself must still be asked.
			rule, _, _ := strings.Cut(input.Name, ":")
			usedSignals[strings.ToLower(inputType+":"+rule)] = true
		}
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
