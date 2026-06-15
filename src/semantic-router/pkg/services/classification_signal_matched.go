package services

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var matchedSignalResolvers = map[string]func(*MatchedSignals) *[]string{
	config.SignalTypeKeyword:      func(target *MatchedSignals) *[]string { return &target.Keywords },
	config.SignalTypeEmbedding:    func(target *MatchedSignals) *[]string { return &target.Embeddings },
	config.SignalTypeDomain:       func(target *MatchedSignals) *[]string { return &target.Domains },
	config.SignalTypeFactCheck:    func(target *MatchedSignals) *[]string { return &target.FactCheck },
	config.SignalTypeUserFeedback: func(target *MatchedSignals) *[]string { return &target.UserFeedback },
	config.SignalTypeReask:        func(target *MatchedSignals) *[]string { return &target.Reask },
	config.SignalTypePreference:   func(target *MatchedSignals) *[]string { return &target.Preferences },
	config.SignalTypeLanguage:     func(target *MatchedSignals) *[]string { return &target.Language },
	config.SignalTypeContext:      func(target *MatchedSignals) *[]string { return &target.Context },
	config.SignalTypeStructure:    func(target *MatchedSignals) *[]string { return &target.Structure },
	config.SignalTypeComplexity:   func(target *MatchedSignals) *[]string { return &target.Complexity },
	config.SignalTypeModality:     func(target *MatchedSignals) *[]string { return &target.Modality },
	config.SignalTypeAuthz:        func(target *MatchedSignals) *[]string { return &target.Authz },
	config.SignalTypeJailbreak:    func(target *MatchedSignals) *[]string { return &target.Jailbreak },
	config.SignalTypePII:          func(target *MatchedSignals) *[]string { return &target.PII },
	config.SignalTypeKB:           func(target *MatchedSignals) *[]string { return &target.KB },
	config.SignalTypeConversation: func(target *MatchedSignals) *[]string { return &target.Conversation },
	config.SignalTypeEvent:        func(target *MatchedSignals) *[]string { return &target.Event },
	config.SignalTypeProjection:   func(target *MatchedSignals) *[]string { return &target.Projection },
}

func buildMatchedSignals(signals *classification.SignalResults) *MatchedSignals {
	if signals == nil {
		return &MatchedSignals{}
	}

	return &MatchedSignals{
		Keywords:     signals.MatchedKeywordRules,
		Embeddings:   signals.MatchedEmbeddingRules,
		Domains:      signals.MatchedDomainRules,
		FactCheck:    signals.MatchedFactCheckRules,
		UserFeedback: signals.MatchedUserFeedbackRules,
		Reask:        signals.MatchedReaskRules,
		Preferences:  signals.MatchedPreferenceRules,
		Language:     signals.MatchedLanguageRules,
		Context:      signals.MatchedContextRules,
		Structure:    signals.MatchedStructureRules,
		Complexity:   signals.MatchedComplexityRules,
		Modality:     signals.MatchedModalityRules,
		Authz:        signals.MatchedAuthzRules,
		Jailbreak:    signals.MatchedJailbreakRules,
		PII:          signals.MatchedPIIRules,
		KB:           signals.MatchedKBRules,
		Conversation: signals.MatchedConversationRules,
		Event:        signals.MatchedEventRules,
		Projection:   signals.MatchedProjectionRules,
	}
}

// extractUsedSignalsFromDecision extracts all signals used in a decision's rule configuration.
// This includes ALL signals defined in the decision rules, not just the ones that matched.
func (s *ClassificationService) extractUsedSignalsFromDecision(decision *config.Decision) *MatchedSignals {
	usedSignals := &MatchedSignals{}
	s.extractSignalsFromRuleCombination(decision.Rules, usedSignals)
	return usedSignals
}

// extractSignalsFromRuleCombination recursively extracts signals from a rule combination.
func (s *ClassificationService) extractSignalsFromRuleCombination(rules config.RuleCombination, usedSignals *MatchedSignals) {
	if rules.IsLeaf() {
		appendSignalToMatchedSignals(usedSignals, rules.Type, rules.Name)
	}
	for _, condition := range rules.Conditions {
		s.extractSignalsFromRuleCombination(condition, usedSignals)
	}
}

func appendSignalName(target *[]string, signalName string) {
	if signalName == "" {
		return
	}
	if !contains(*target, signalName) {
		*target = append(*target, signalName)
	}
}

// contains checks if a string slice contains a specific string.
func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// getUnmatchedSignals returns all configured signals that were not matched.
func (s *ClassificationService) getUnmatchedSignals(signals *classification.SignalResults) *MatchedSignals {
	unmatched := &MatchedSignals{}

	if s.classifier == nil || s.config == nil {
		return unmatched
	}
	cfg := s.classifier.Config
	collectUnmatchedRuleNames(&unmatched.Keywords, cfg.KeywordRules, signals.MatchedKeywordRules, func(rule config.KeywordRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.Embeddings, cfg.EmbeddingRules, signals.MatchedEmbeddingRules, func(rule config.EmbeddingRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.Domains, cfg.Categories, signals.MatchedDomainRules, func(category config.Category) string { return category.Name })
	collectUnmatchedRuleNames(&unmatched.FactCheck, cfg.FactCheckRules, signals.MatchedFactCheckRules, func(rule config.FactCheckRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.UserFeedback, cfg.UserFeedbackRules, signals.MatchedUserFeedbackRules, func(rule config.UserFeedbackRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.Reask, cfg.ReaskRules, signals.MatchedReaskRules, func(rule config.ReaskRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.Preferences, cfg.PreferenceRules, signals.MatchedPreferenceRules, func(rule config.PreferenceRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.Language, cfg.LanguageRules, signals.MatchedLanguageRules, func(rule config.LanguageRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.Context, cfg.ContextRules, signals.MatchedContextRules, func(rule config.ContextRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.Structure, cfg.StructureRules, signals.MatchedStructureRules, func(rule config.StructureRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.Complexity, cfg.ComplexityRules, signals.MatchedComplexityRules, func(rule config.ComplexityRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.Modality, cfg.ModalityRules, signals.MatchedModalityRules, func(rule config.ModalityRule) string { return rule.Name })
	collectUnmatchedAuthzRules(&unmatched.Authz, cfg.GetRoleBindings(), signals.MatchedAuthzRules)
	collectUnmatchedRuleNames(&unmatched.Jailbreak, cfg.JailbreakRules, signals.MatchedJailbreakRules, func(rule config.JailbreakRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.PII, cfg.PIIRules, signals.MatchedPIIRules, func(rule config.PIIRule) string { return rule.Name })
	collectUnmatchedProjectionOutputs(&unmatched.Projection, cfg.Projections.Mappings, signals.MatchedProjectionRules)

	return unmatched
}

func appendSignalToMatchedSignals(target *MatchedSignals, signalType string, signalName string) {
	resolver, ok := matchedSignalResolvers[strings.ToLower(strings.TrimSpace(signalType))]
	if !ok {
		return
	}
	appendSignalName(resolver(target), strings.TrimSpace(signalName))
}

func collectUnmatchedRuleNames[T any](target *[]string, rules []T, matched []string, nameFn func(T) string) {
	matchedSet := makeStringSet(matched)
	for _, rule := range rules {
		name := strings.TrimSpace(nameFn(rule))
		if name == "" || matchedSet[name] {
			continue
		}
		*target = append(*target, name)
	}
}

func collectUnmatchedAuthzRules(target *[]string, rules []config.RoleBinding, matched []string) {
	matchedSet := makeStringSet(matched)
	for _, rule := range rules {
		if matchedSet[rule.Role] || matchedSet[rule.Name] {
			continue
		}
		name := strings.TrimSpace(rule.Name)
		if name == "" {
			name = strings.TrimSpace(rule.Role)
		}
		if name != "" {
			*target = append(*target, name)
		}
	}
}

func collectUnmatchedProjectionOutputs(
	target *[]string,
	mappings []config.ProjectionMapping,
	matched []string,
) {
	matchedSet := makeStringSet(matched)
	for _, mapping := range mappings {
		for _, output := range mapping.Outputs {
			if output.Name == "" || matchedSet[output.Name] {
				continue
			}
			*target = append(*target, output.Name)
		}
	}
}

func makeStringSet(values []string) map[string]bool {
	set := make(map[string]bool, len(values))
	for _, value := range values {
		set[value] = true
	}
	return set
}
