package config

import (
	"fmt"
	"math"
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
		if err := validateDecisionDeployment(cfg, rule.Deployment, "question"); err != nil {
			return fmt.Errorf("routing.signals.decision[%s]: %w", rule.Name, err)
		}
	}
	if err := validateSetLabelAnswerKeys(cfg, cfg.DecisionRules); err != nil {
		return err
	}
	for _, decision := range cfg.Decisions {
		algorithm := decision.Algorithm
		if algorithm == nil || !strings.EqualFold(strings.TrimSpace(algorithm.Type), DecisionAlgorithmDecision) || algorithm.Decision == nil {
			continue
		}
		if err := validateDecisionDeployment(cfg, algorithm.Decision.Deployment, "selector"); err != nil {
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
	if rule.TimeoutMs < 0 || rule.TimeoutMs > MaxDecisionTimeoutMs {
		return fmt.Errorf("timeout_ms must be within [1, %d] when set", MaxDecisionTimeoutMs)
	}
	if rule.PriorUserTurns < 0 || rule.PriorUserTurns > MaxDecisionPriorUserTurns {
		return fmt.Errorf("prior_user_turns must be within [0, %d]", MaxDecisionPriorUserTurns)
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

// validateDecisionDeployment checks the deployment a decision question or
// selector asks: a declared model_runtime deployment, or, when it names none,
// the decision model, which must support the requested task.
func validateDecisionDeployment(cfg *RouterConfig, deployment, asker string) error {
	if deployment != "" {
		return validateModelRuntimeReference(cfg, deployment)
	}
	if _, _, _, err := cfg.DecisionModelDeployment(); err != nil {
		return fmt.Errorf("default decision deployment for %s: %w", asker, err)
	}
	return nil
}

// validateSetLabelAnswerKeys rejects a rule whose name equals the answer key
// ("<rule>.<label>") of a Set question asked of the same deployment: the
// model answers each Set label under that key in the same call.
func validateSetLabelAnswerKeys(cfg *RouterConfig, rules []DecisionSignalRule) error {
	names := make(map[string]string, len(rules))
	for _, rule := range rules {
		names[cfg.DecisionQuestionDeployment(rule)+"\x00"+rule.Name] = rule.Name
	}
	for _, rule := range rules {
		if rule.Question.Type != DecisionQuestionSet {
			continue
		}
		deployment := cfg.DecisionQuestionDeployment(rule)
		for _, label := range rule.Question.Labels {
			if other, exists := names[deployment+"\x00"+rule.Name+"."+label.Key]; exists {
				return fmt.Errorf("routing.signals.decision[%s]: the name collides with the answer key of set question %q's label %q on deployment %q; rename one of them", other, rule.Name, label.Key, deployment)
			}
		}
	}
	return nil
}

func validateDecisionQuestion(question DecisionQuestion) error {
	if strings.TrimSpace(question.Instructions) == "" {
		return fmt.Errorf("instructions are required")
	}
	if question.Type != DecisionQuestionSet && question.Type != DecisionQuestionSpan {
		if len(question.Labels) > 0 {
			return fmt.Errorf("labels apply only to set and span questions")
		}
		if question.Threshold != nil {
			return fmt.Errorf("threshold applies only to set and span questions")
		}
	}
	if question.Head != "" && question.Type != DecisionQuestionSpan {
		return fmt.Errorf("head applies only to span questions")
	}
	switch question.Type {
	case DecisionQuestionSet, DecisionQuestionSpan:
		return validateLabelledDecisionQuestion(question)
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
		return fmt.Errorf("type must be choice, noul, score, set or span")
	}
}

// validateLabelledDecisionQuestion checks a Set or Span question: 1..255
// unique labels, an optional threshold in [0, 1] and, for a span, an
// optional head.
func validateLabelledDecisionQuestion(question DecisionQuestion) error {
	if len(question.Choices) > 0 || len(question.Levels) > 0 {
		return fmt.Errorf("a %s question takes labels, not choices or levels", question.Type)
	}
	if len(question.Labels) < MinDecisionLabels || len(question.Labels) > MaxDecisionLabels {
		return fmt.Errorf("a %s question needs %d..%d labels", question.Type, MinDecisionLabels, MaxDecisionLabels)
	}
	if threshold := question.Threshold; threshold != nil && (math.IsNaN(*threshold) || *threshold < 0 || *threshold > 1) {
		return fmt.Errorf("threshold must be within [0, 1]")
	}
	if question.Head != "" && question.Head != DecisionSpanHeadRouter && question.Head != DecisionSpanHeadBroad {
		return fmt.Errorf("head must be %s or %s", DecisionSpanHeadRouter, DecisionSpanHeadBroad)
	}
	return validateDecisionLabelKeys(question.Labels)
}

func validateDecisionLabelKeys(labels []DecisionChoice) error {
	seen := make(map[string]struct{}, len(labels))
	for _, label := range labels {
		if strings.TrimSpace(label.Key) == "" || strings.TrimSpace(label.Key) != label.Key {
			return fmt.Errorf("label keys must be nonempty and trimmed")
		}
		if _, exists := seen[label.Key]; exists {
			return fmt.Errorf("duplicate label key %q", label.Key)
		}
		seen[label.Key] = struct{}{}
	}
	return nil
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
	return deployment.ValidateDecisionInput(name)
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
	if !rule.Question.Labelled() {
		if node.Label != "" {
			return fmt.Errorf("decision '%s': decision condition %q is a %s question and takes no label", decisionName, node.Name, rule.Question.Type)
		}
		return nil
	}
	if node.Label == "" || !stringSliceContains(rule.Question.OptionKeys(), node.Label) {
		option := "choice key"
		if rule.Question.Type != DecisionQuestionChoice {
			option = "label"
		}
		return fmt.Errorf("decision '%s': %s condition %q requires a declared %s as its label", decisionName, rule.Question.Type, node.Name, option)
	}
	return nil
}

func validateDecisionSelectorConfig(decisionName string, modelRefs []ModelRef, algorithm *AlgorithmConfig) error {
	cfg := algorithm.Decision
	if cfg == nil {
		return fmt.Errorf("decision '%s': algorithm.type=decision requires algorithm.decision configuration", decisionName)
	}
	path := fmt.Sprintf("decision '%s', algorithm.decision", decisionName)
	if strings.TrimSpace(cfg.Deployment) != cfg.Deployment {
		return fmt.Errorf("%s: deployment must name a model_runtime deployment; omit it to ask the decision model", path)
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
