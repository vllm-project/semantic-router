package config

import (
	"fmt"
	"sort"
	"strings"
)

// complexityDifficultyLevels enumerates the difficulty levels a complexity
// rule can emit at runtime. It mirrors classifyComplexityDifficulty in
// pkg/classification/complexity_rule_scoring.go, which always tags a rule
// result as "<rule>:<level>" (e.g. "needs_reasoning:hard"). The config
// package cannot import the classification package (import cycle), so the set
// is duplicated here; keep the two in sync.
var complexityDifficultyLevels = map[string]struct{}{
	"hard":   {},
	"easy":   {},
	"medium": {},
}

// validateComplexityContracts checks that every decision condition of type
// "complexity" references a declared complexity rule using the required
// "<rule>:<difficulty>" form.
//
// The decision engine matches complexity conditions by literal membership
// against the classifier's emitted "<rule>:<difficulty>" strings
// (pkg/decision/engine.go uses slices.Contains), so a bare rule name — or one
// with an unknown difficulty — never matches at runtime and becomes a
// silently inert route. Config validation previously accepted the bare form
// because the projection-input validator strips the ":" suffix before
// checking the base name, leaving the mismatch undetected. Enforcing the
// suffix here turns that latent misconfiguration into a clear load-time error
// (issue #2324).
func validateComplexityContracts(cfg *RouterConfig) error {
	declared := collectComplexityRuleNames(cfg.ComplexityRules)
	for _, decision := range cfg.Decisions {
		if err := validateDecisionComplexityReferences(decision.Name, &decision.Rules, declared); err != nil {
			return err
		}
	}
	return nil
}

func validateDecisionComplexityReferences(decisionName string, node *RuleNode, declared map[string]struct{}) error {
	if node == nil {
		return nil
	}

	if node.Type == SignalTypeComplexity {
		if err := validateComplexityConditionName(decisionName, node.Name, declared); err != nil {
			return err
		}
	}

	for i := range node.Conditions {
		if err := validateDecisionComplexityReferences(decisionName, &node.Conditions[i], declared); err != nil {
			return err
		}
	}

	return nil
}

func validateComplexityConditionName(decisionName, name string, declared map[string]struct{}) error {
	// Split on the last ":" so complexity rule names that themselves contain a
	// colon still resolve to the trailing difficulty token.
	idx := strings.LastIndex(name, ":")
	if idx <= 0 || idx == len(name)-1 {
		return fmt.Errorf(
			"decision %q references complexity %q, but a complexity condition must name a difficulty level as "+
				"\"<rule>:<%s>\"; a bare rule name never matches at runtime because the classifier always emits "+
				"\"<rule>:<difficulty>\"",
			decisionName, name, strings.Join(sortedComplexityDifficultyLevels(), "|"),
		)
	}

	rule, difficulty := name[:idx], name[idx+1:]
	if _, ok := declared[rule]; !ok {
		return fmt.Errorf(
			"decision %q references complexity rule %q, but no routing.signals.complexity entry declares that name",
			decisionName, rule,
		)
	}
	if _, ok := complexityDifficultyLevels[difficulty]; !ok {
		return fmt.Errorf(
			"decision %q references complexity %q with unsupported difficulty %q; valid levels: %s",
			decisionName, name, difficulty, strings.Join(sortedComplexityDifficultyLevels(), ", "),
		)
	}

	return nil
}

func sortedComplexityDifficultyLevels() []string {
	levels := make([]string, 0, len(complexityDifficultyLevels))
	for level := range complexityDifficultyLevels {
		levels = append(levels, level)
	}
	sort.Strings(levels)
	return levels
}
