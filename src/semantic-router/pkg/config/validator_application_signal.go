package config

import (
	"fmt"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// validateApplicationSignalContracts checks a scope's application signal
// rules: a unique, well-formed name per type, the stage and input each type
// takes, labels and ignore_labels among the preset's options, the predicate,
// and a deployment that answers the compiled question.
func validateApplicationSignalContracts(cfg *RouterConfig) (err error) {
	defer func() { err = cfg.inRecipe(err) }()
	seen := map[string]bool{}
	for _, spec := range cfg.applicationRuleSpecs() {
		path := "signals." + ApplicationCollections[spec.signalType] + "[" + spec.name + "]"
		if !validRuleName(spec.name) {
			return fmt.Errorf("signals.%s: every rule needs a name, trimmed and without ':'", ApplicationCollections[spec.signalType])
		}
		if seen[spec.signalType+"\x00"+spec.name] {
			return fmt.Errorf("%s: duplicate name %q", path, spec.name)
		}
		seen[spec.signalType+"\x00"+spec.name] = true
		if err := validateApplicationRuleSpec(spec); err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		rule := cfg.compileApplicationRule(spec)
		if err := ValidateSystemOneRuleContract(rule.Rule); err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		if err := validateQuestionDeployment(cfg, rule.Rule.Deployment, spec.signalType); err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		deployment := cfg.SystemOneRuleDeployment(rule.Rule)
		questionTypes, modalities := cfg.deploymentAnswers(deployment)
		if err := CheckApplicationAnswerer(cfg.ApplicationRulePath(spec.signalType, spec.name), cfg.ScopePath(""), deployment, rule, questionTypes, cfg.ModelDeployments[deployment].DeclaredPresets(), modalities); err != nil {
			return err
		}
	}
	return nil
}

// validateApplicationRuleSpec checks what one rule writes before it is
// compiled: its stage, its input, and labels and ignore_labels that name
// options of its preset.
func validateApplicationRuleSpec(spec applicationRuleSpec) error {
	switch spec.stage {
	case "", presets.StageRequest, presets.StageResponse:
	default:
		return fmt.Errorf("stage must be %s or %s", presets.StageRequest, presets.StageResponse)
	}
	switch UnknownPolicy(spec.onUnknown) {
	case "", RuleOnUnknownMatch, RuleOnUnknownNoMatch:
	default:
		return fmt.Errorf("on_unknown must be %s or %s", RuleOnUnknownMatch, RuleOnUnknownNoMatch)
	}
	if spec.onUnknown != "" && spec.stage == presets.StageResponse {
		return fmt.Errorf("on_unknown applies to a request-stage rule; the response_guard plugin's on_unknown decides a response-stage rule")
	}
	if spec.input.PreviousTurns < 0 || spec.input.PreviousTurns > MaxSystemOnePreviousTurns {
		return fmt.Errorf("input.previous_turns must be within [1, %d] when set", MaxSystemOnePreviousTurns)
	}
	preset, _ := presets.Lookup(spec.preset, spec.stage)
	options := preset.OptionKeys()
	for _, field := range []struct {
		name string
		keys []string
	}{{"labels", spec.labels}, {"ignore_labels", spec.ignoreLabels}} {
		for _, key := range field.keys {
			if slices.Contains(options, key) {
				continue
			}
			if spec.signalType == SignalTypeDomain && field.name == "labels" && len(spec.labels) == 1 && key == spec.name {
				return fmt.Errorf("the rule's name %q is not a domain, so the rule names the domains it matches: set labels to some of %s%s",
					key, strings.Join(options, ", "), formatDomainSuggestion(key))
			}
			return fmt.Errorf("%s: %q is not a %s label; the labels are %s", field.name, key, spec.signalType, strings.Join(options, ", "))
		}
	}
	return validateNumericPredicateContract(spec.predicate)
}

// validateApplicationLeaf checks a condition on an application signal: it
// names a declared rule, without a label when the rule matches itself, or
// with an option of the rule's preset that the rule counts.
func validateApplicationLeaf(cfg *RouterConfig, decisionName string, node *RuleNode) error {
	signalType := strings.ToLower(strings.TrimSpace(node.Type))
	rule, ok := cfg.ApplicationRuleByName(signalType, node.Name)
	if !ok {
		return fmt.Errorf("decision '%s': %s condition references unknown signal %q; declare it in the recipe's signals.%s", decisionName, signalType, node.Name, ApplicationCollections[signalType])
	}
	if node.Label == "" {
		if !rule.Rule.MatchesItself() {
			return fmt.Errorf("decision '%s': %s condition %q names a rule without labels; name the option as the condition's label, or give the rule labels", decisionName, signalType, node.Name)
		}
		return nil
	}
	if !slices.Contains(rule.Rule.Question.OptionKeys(), node.Label) {
		return fmt.Errorf("decision '%s': %s condition %q names label %q; the labels are %s", decisionName, signalType, node.Name, node.Label, strings.Join(rule.Rule.Question.OptionKeys(), ", "))
	}
	if slices.Contains(rule.Rule.IgnoreLabels, node.Label) {
		return fmt.Errorf("decision '%s': %s condition %q names label %q, which the rule's ignore_labels never count", decisionName, signalType, node.Name, node.Label)
	}
	return nil
}
