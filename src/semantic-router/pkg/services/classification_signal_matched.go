package services

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var matchedSignalResolvers = map[string]func(*MatchedSignals) *[]string{
	config.SignalTypeKeyword:       func(target *MatchedSignals) *[]string { return &target.Keywords },
	config.SignalTypeDomain:        func(target *MatchedSignals) *[]string { return &target.Domains },
	config.SignalTypeFactCheck:     func(target *MatchedSignals) *[]string { return &target.FactCheck },
	config.SignalTypeUserFeedback:  func(target *MatchedSignals) *[]string { return &target.UserFeedback },
	config.SignalTypeTurnRelation:  func(target *MatchedSignals) *[]string { return &target.TurnRelation },
	config.SignalTypeLanguage:      func(target *MatchedSignals) *[]string { return &target.Language },
	config.SignalTypeContext:       func(target *MatchedSignals) *[]string { return &target.Context },
	config.SignalTypeComplexity:    func(target *MatchedSignals) *[]string { return &target.Complexity },
	config.SignalTypeModality:      func(target *MatchedSignals) *[]string { return &target.Modality },
	config.SignalTypeAuthz:         func(target *MatchedSignals) *[]string { return &target.Authz },
	config.SignalTypeJailbreak:     func(target *MatchedSignals) *[]string { return &target.Jailbreak },
	config.SignalTypeSafety:        func(target *MatchedSignals) *[]string { return &target.Safety },
	config.SignalTypePII:           func(target *MatchedSignals) *[]string { return &target.PII },
	config.SignalTypeConversation:  func(target *MatchedSignals) *[]string { return &target.Conversation },
	config.SignalTypeMetadata:      func(target *MatchedSignals) *[]string { return &target.Metadata },
	config.SignalTypeInputModality: func(target *MatchedSignals) *[]string { return &target.InputModality },
	config.SignalTypeSystemOne:     func(target *MatchedSignals) *[]string { return &target.SystemOne },
	config.SignalTypeProjection:    func(target *MatchedSignals) *[]string { return &target.Projection },
}

func buildMatchedSignals(signals *classification.SignalResults) *MatchedSignals {
	if signals == nil {
		return &MatchedSignals{}
	}

	return &MatchedSignals{
		Keywords:      signals.MatchedKeywordRules,
		Domains:       signals.MatchedDomainRules,
		FactCheck:     signals.MatchedFactCheckRules,
		UserFeedback:  signals.MatchedUserFeedbackRules,
		TurnRelation:  signals.MatchedTurnRelationRules,
		Language:      signals.MatchedLanguageRules,
		Context:       signals.MatchedContextRules,
		Complexity:    signals.MatchedComplexityRules,
		Modality:      signals.MatchedModalityRules,
		Authz:         signals.MatchedAuthzRules,
		Jailbreak:     signals.MatchedJailbreakRules,
		Safety:        signals.MatchedSafetyRules,
		PII:           signals.MatchedPIIRules,
		Conversation:  signals.MatchedConversationRules,
		Metadata:      signals.MatchedMetadataRules,
		InputModality: signals.MatchedInputModalityRules,
		SystemOne:     signals.MatchedSystemOneRules,
		Projection:    signals.MatchedProjectionRules,
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
func getUnmatchedSignals(
	signals *classification.SignalResults,
	classifier *classification.Classifier,
) *MatchedSignals {
	unmatched := &MatchedSignals{}

	if classifier == nil || classifier.Config == nil {
		return unmatched
	}
	cfg := classifier.Config
	collectUnmatchedRuleNames(&unmatched.Keywords, cfg.KeywordRules, signals.MatchedKeywordRules, func(rule config.KeywordRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.Language, cfg.LanguageRules, signals.MatchedLanguageRules, func(rule config.LanguageRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.Context, cfg.ContextRules, signals.MatchedContextRules, func(rule config.ContextRule) string { return rule.Name })
	collectUnmatchedAuthzRules(&unmatched.Authz, cfg.GetRoleBindings(), signals.MatchedAuthzRules)
	collectUnmatchedProjectionOutputs(&unmatched.Projection, cfg.Projections.Mappings, signals.MatchedProjectionRules)
	collectUnmatchedRuleNames(&unmatched.Metadata, cfg.MetadataRules, signals.MatchedMetadataRules, func(rule config.MetadataRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.InputModality, cfg.InputModalityRules, signals.MatchedInputModalityRules, func(rule config.InputModalityRule) string { return rule.Name })
	systemOne := make([]string, len(cfg.SystemOneRules))
	for index, rule := range cfg.SystemOneRules {
		systemOne[index] = rule.Name
	}
	collectUnreferencedRules(&unmatched.SystemOne, systemOne, signals.MatchedSystemOneRules)
	for _, rule := range cfg.ApplicationRules() {
		if rule.Rule.EffectiveStage() != config.SystemOneStageRequest {
			continue
		}
		if target, ok := matchedSignalResolvers[rule.Type]; ok {
			collectUnreferencedRules(target(unmatched), []string{rule.Rule.Name}, matchedReferences(signals, rule.Type))
		}
	}

	return unmatched
}

// matchedReferences is the matched-reference list of a systemone or
// application signal type.
func matchedReferences(signals *classification.SignalResults, signalType string) []string {
	resolver, ok := matchedSignalResolvers[signalType]
	if !ok {
		return nil
	}
	return *resolver(buildMatchedSignals(signals))
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

// collectUnreferencedRules lists the systemone or application rules with no
// matched reference, the rule's own or "rule:label".
func collectUnreferencedRules(target *[]string, rules []string, matched []string) {
	for _, rule := range rules {
		hit := false
		for _, name := range matched {
			if name == rule || strings.HasPrefix(name, rule+":") {
				hit = true
				break
			}
		}
		if !hit {
			*target = append(*target, rule)
		}
	}
}
