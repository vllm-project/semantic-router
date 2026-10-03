package config

import (
	"fmt"
	"strings"
)

// validateDecisionSignalContracts checks decision signals, the decision
// algorithm and the model_runtime deployments they name.
func validateDecisionSignalContracts(cfg *RouterConfig) error {
	seen := make(map[string]struct{}, len(cfg.DecisionRules))
	for index, rule := range cfg.DecisionRules {
		if err := ValidateDecisionSignalRuleContract(rule); err != nil {
			return fmt.Errorf("routing.signals.decision[%d]: %w", index, err)
		}
		if _, exists := seen[rule.Name]; exists {
			return fmt.Errorf("routing.signals.decision[%d]: duplicate name %q", index, rule.Name)
		}
		seen[rule.Name] = struct{}{}
		if err := validateModelRuntimeReference(cfg, rule.Deployment); err != nil {
			return fmt.Errorf("routing.signals.decision[%s]: %w", rule.Name, err)
		}
	}
	for _, decision := range cfg.Decisions {
		algorithm := decision.Algorithm
		if algorithm == nil || !strings.EqualFold(strings.TrimSpace(algorithm.Type), DecisionAlgorithmDecision) || algorithm.Decision == nil {
			continue
		}
		if err := validateModelRuntimeReference(cfg, algorithm.Decision.Deployment); err != nil {
			return fmt.Errorf("decision '%s', algorithm.decision: %w", decision.Name, err)
		}
	}
	return nil
}

// ValidateDecisionSignalRuleContract validates one decision signal without the
// surrounding configuration; the DSL compiler shares it.
func ValidateDecisionSignalRuleContract(rule DecisionSignalRule) error {
	if strings.TrimSpace(rule.Name) == "" || strings.TrimSpace(rule.Name) != rule.Name || strings.Contains(rule.Name, ":") {
		return fmt.Errorf("name is required, trimmed and without ':'")
	}
	if strings.TrimSpace(rule.Deployment) == "" {
		return fmt.Errorf("deployment is required")
	}
	if rule.TimeoutMs < 0 || rule.TimeoutMs > MaxDecisionTimeoutMs {
		return fmt.Errorf("timeout_ms must be within [1, %d] when set", MaxDecisionTimeoutMs)
	}
	if err := validateDecisionQuestion(rule.Question); err != nil {
		return fmt.Errorf("question: %w", err)
	}
	if rule.Question.Type == DecisionQuestionScore && rule.Predicate == nil {
		return fmt.Errorf("a score question requires a predicate on its expected level")
	}
	if err := validateNumericPredicateContract(rule.Predicate); err != nil {
		return err
	}
	return nil
}

func validateDecisionQuestion(question DecisionQuestion) error {
	if strings.TrimSpace(question.Instructions) == "" {
		return fmt.Errorf("instructions are required")
	}
	switch question.Type {
	case DecisionQuestionChoice:
		if len(question.Levels) > 0 {
			return fmt.Errorf("levels apply only to score questions")
		}
		if len(question.Choices) < MinDecisionChoices || len(question.Choices) > MaxDecisionChoices {
			return fmt.Errorf("a choice question needs %d..%d choices", MinDecisionChoices, MaxDecisionChoices)
		}
		return validateDecisionChoiceKeys(question.Choices, nil)
	case DecisionQuestionNoul:
		if len(question.Levels) > 0 {
			return fmt.Errorf("levels apply only to score questions")
		}
		return validateDecisionChoiceKeys(question.Choices, map[string]struct{}{"false": {}, "true": {}})
	case DecisionQuestionScore:
		if len(question.Choices) > 0 {
			return fmt.Errorf("a score question takes ordered levels, not choices")
		}
		if len(question.Levels) < MinDecisionLevels || len(question.Levels) > MaxDecisionLevels {
			return fmt.Errorf("a score question needs %d..%d levels", MinDecisionLevels, MaxDecisionLevels)
		}
		for index, level := range question.Levels {
			if strings.TrimSpace(level) == "" {
				return fmt.Errorf("level %d needs a description", index)
			}
		}
		return nil
	default:
		return fmt.Errorf("type must be choice, noul or score")
	}
}

func validateDecisionChoiceKeys(choices []DecisionChoice, allowed map[string]struct{}) error {
	seen := make(map[string]struct{}, len(choices))
	for _, choice := range choices {
		if strings.TrimSpace(choice.Key) == "" || strings.TrimSpace(choice.Key) != choice.Key {
			return fmt.Errorf("choice keys must be nonempty and trimmed")
		}
		if allowed != nil {
			if _, ok := allowed[choice.Key]; !ok {
				return fmt.Errorf("a noul question accepts only the false and true choices")
			}
		}
		if _, exists := seen[choice.Key]; exists {
			return fmt.Errorf("duplicate choice key %q", choice.Key)
		}
		seen[choice.Key] = struct{}{}
	}
	return nil
}

func validateModelRuntimeReference(cfg *RouterConfig, name string) error {
	deployment, exists := cfg.ModelDeployments[name]
	if !exists {
		return fmt.Errorf("deployment %q is not declared in global.model_catalog.deployments", name)
	}
	if !deployment.IsModelRuntime() {
		return fmt.Errorf("deployment %q must use provider %s", name, ModelRuntimeProvider)
	}
	return nil
}

func decisionSignalRuleByName(rules []DecisionSignalRule, name string) *DecisionSignalRule {
	for index := range rules {
		if rules[index].Name == name {
			return &rules[index]
		}
	}
	return nil
}

func validateDecisionModelLeaf(cfg *RouterConfig, decisionName string, node *RuleNode) error {
	rule := decisionSignalRuleByName(cfg.DecisionRules, node.Name)
	if rule == nil {
		return fmt.Errorf("decision '%s': decision condition references unknown signal %q", decisionName, node.Name)
	}
	if rule.Question.Type != DecisionQuestionChoice {
		if node.Label != "" {
			return fmt.Errorf("decision '%s': decision condition %q is a %s question and takes no label", decisionName, node.Name, rule.Question.Type)
		}
		return nil
	}
	if node.Label == "" || !stringSliceContains(rule.Question.OptionKeys(), node.Label) {
		return fmt.Errorf("decision '%s': choice condition %q requires a declared choice key as its label", decisionName, node.Name)
	}
	return nil
}

func validateDecisionSelectorConfig(decisionName string, modelRefs []ModelRef, algorithm *AlgorithmConfig) error {
	cfg := algorithm.Decision
	if cfg == nil {
		return fmt.Errorf("decision '%s': algorithm.type=decision requires algorithm.decision configuration", decisionName)
	}
	path := fmt.Sprintf("decision '%s', algorithm.decision", decisionName)
	if strings.TrimSpace(cfg.Deployment) == "" {
		return fmt.Errorf("%s: deployment is required", path)
	}
	if strings.TrimSpace(cfg.Instructions) == "" {
		return fmt.Errorf("%s: instructions are required", path)
	}
	if cfg.TimeoutMs < 0 || cfg.TimeoutMs > MaxDecisionTimeoutMs {
		return fmt.Errorf("%s: timeout_ms must be within [1, %d] when set", path, MaxDecisionTimeoutMs)
	}
	if len(modelRefs) < MinDecisionChoices || len(modelRefs) > MaxDecisionChoices {
		return fmt.Errorf("%s: needs %d..%d modelRefs to choose from", path, MinDecisionChoices, MaxDecisionChoices)
	}
	models := make(map[string]struct{}, len(modelRefs))
	for _, ref := range modelRefs {
		if _, exists := models[ref.Model]; exists {
			return fmt.Errorf("%s: requires unique modelRefs; duplicate model %q", path, ref.Model)
		}
		models[ref.Model] = struct{}{}
	}
	for candidate := range cfg.Candidates {
		if _, exists := models[candidate]; !exists {
			return fmt.Errorf("%s: candidates names %q, which is not one of the decision's modelRefs", path, candidate)
		}
	}
	return nil
}
