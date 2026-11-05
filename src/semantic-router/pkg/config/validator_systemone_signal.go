package config

import (
	"fmt"
	"math"
	"slices"
	"strings"
)

// validateSystemOneContracts checks systemone rules, the systemone selector
// and the model_runtime or systemone deployments they name.
func validateSystemOneContracts(cfg *RouterConfig) error {
	seen := make(map[string]struct{}, len(cfg.SystemOneRules))
	for index, rule := range cfg.SystemOneRules {
		if err := ValidateSystemOneRuleContract(rule); err != nil {
			path := fmt.Sprintf("routing.signals.systemone[%d]", index)
			if validRuleName(rule.Name) {
				path = SystemOneRulePath(rule.Name)
			}
			return fmt.Errorf("%s: %w", path, err)
		}
		path := SystemOneRulePath(rule.Name)
		if _, exists := seen[rule.Name]; exists {
			return fmt.Errorf("%s: duplicate name %q", path, rule.Name)
		}
		seen[rule.Name] = struct{}{}
		if err := validateQuestionDeployment(cfg, rule.Deployment, "question"); err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		if err := validateSystemOneAnswerer(cfg, cfg.ScopePath("signals.systemone["+rule.Name+"]"), rule); err != nil {
			return err
		}
		if err := validateDeclaredQuestionType(cfg, cfg.SystemOneRuleDeployment(rule), rule.Question.Type); err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
	}
	if err := validateSetLabelAnswerKeys(cfg, cfg.SystemOneRules); err != nil {
		return err
	}
	for _, decision := range cfg.Decisions {
		algorithm := decision.Algorithm
		if algorithm == nil || !strings.EqualFold(strings.TrimSpace(algorithm.Type), DecisionAlgorithmSystemOne) || algorithm.SystemOne == nil {
			continue
		}
		if err := validateQuestionDeployment(cfg, algorithm.SystemOne.Deployment, "selector"); err != nil {
			return fmt.Errorf("decision '%s', algorithm.systemone: %w", decision.Name, err)
		}
		if err := validateDeclaredQuestionType(cfg, cfg.SystemOneSelectorDeployment(*algorithm.SystemOne), SystemOneQuestionChoice); err != nil {
			return fmt.Errorf("decision '%s', algorithm.systemone: %w", decision.Name, err)
		}
	}
	return nil
}

// SystemOneRulePath is where a configuration writes a rule.
func SystemOneRulePath(name string) string {
	return "routing.signals.systemone[" + name + "]"
}

func validRuleName(name string) bool {
	return strings.TrimSpace(name) != "" && strings.TrimSpace(name) == name && !strings.Contains(name, ":")
}

// validateDeclaredQuestionType refuses a question a systemone deployment's
// declared question types leave out. A deployment that declares none is told
// by its API's listing.
func validateDeclaredQuestionType(cfg *RouterConfig, name, questionType string) error {
	deployment, ok := cfg.ModelDeployments[name]
	if !ok || !deployment.IsSystemOne() {
		return nil
	}
	declared := deployment.DeclaredQuestionTypes()
	if len(declared) == 0 || slices.Contains(declared, questionType) {
		return nil
	}
	return fmt.Errorf("deployment %q declares question types %s, so it does not answer a %s question", name, strings.Join(declared, ", "), questionType)
}

// ValidateSystemOneRuleContract validates one systemone rule without the
// surrounding configuration; the DSL compiler shares it.
func ValidateSystemOneRuleContract(rule SystemOneRule) error {
	if !validRuleName(rule.Name) {
		return fmt.Errorf("name is required, trimmed and without ':'")
	}
	if rule.TimeoutMs < 0 || rule.TimeoutMs > MaxSystemOneTimeoutMs {
		return fmt.Errorf("timeout_ms must be within [1, %d] when set", MaxSystemOneTimeoutMs)
	}
	if err := validateSystemOneQuestion(rule.Question); err != nil {
		return fmt.Errorf("question: %w", err)
	}
	if err := validateSystemOneMatching(rule); err != nil {
		return err
	}
	if err := validateSystemOneInput(rule); err != nil {
		return err
	}
	return validateNumericPredicateContract(rule.Predicate)
}

// validateSystemOneMatching checks what a rule matches: a score rule's
// predicate, and labels and ignore_labels that name options of the question.
func validateSystemOneMatching(rule SystemOneRule) error {
	questionType := rule.Question.Type
	if questionType == SystemOneQuestionScore && rule.Predicate == nil {
		return fmt.Errorf("a score question requires a predicate on its expected level")
	}
	if len(rule.Labels) > 0 && !rule.Question.Labelled() {
		return fmt.Errorf("labels name the options a choice, set or span rule matches; a %s rule takes none", questionType)
	}
	if len(rule.IgnoreLabels) > 0 && questionType != SystemOneQuestionSet && questionType != SystemOneQuestionSpan {
		return fmt.Errorf("ignore_labels apply only to set and span rules")
	}
	options := rule.Question.OptionKeys()
	for _, field := range []struct {
		name string
		keys []string
	}{{"labels", rule.Labels}, {"ignore_labels", rule.IgnoreLabels}} {
		for index, key := range field.keys {
			if !slices.Contains(options, key) {
				return fmt.Errorf("%s: %q is not an option of the question", field.name, key)
			}
			if slices.Contains(field.keys[:index], key) {
				return fmt.Errorf("%s lists %q twice", field.name, key)
			}
		}
	}
	for _, key := range rule.IgnoreLabels {
		if slices.Contains(rule.Labels, key) {
			return fmt.Errorf("%q is both in labels and in ignore_labels", key)
		}
	}
	if len(rule.IgnoreLabels) > 0 && len(rule.CountedLabels()) == 0 {
		return fmt.Errorf("ignore_labels leaves no label to count")
	}
	return nil
}

// validateSystemOneInput checks the stage, the input and the coverage a rule
// reads.
func validateSystemOneInput(rule SystemOneRule) error {
	switch rule.Stage {
	case "", SystemOneStageRequest, SystemOneStageResponse:
	default:
		return fmt.Errorf("stage must be %s or %s", SystemOneStageRequest, SystemOneStageResponse)
	}
	input := rule.Input
	if input.PreviousTurns < 0 || input.PreviousTurns > MaxSystemOnePreviousTurns {
		return fmt.Errorf("input.previous_turns must be within [0, %d]", MaxSystemOnePreviousTurns)
	}
	if rule.EffectiveStage() == SystemOneStageRequest && (input.Answer || input.Context) {
		return fmt.Errorf("input.answer and input.context read the model's answer and its grounding context, which a request-stage rule does not have; set stage: response, or read the request's tool results with input.tool_results")
	}
	switch rule.Coverage {
	case "", SystemOneCoverageTruncate, SystemOneCoverageFull:
	default:
		return fmt.Errorf("coverage must be %s or %s", SystemOneCoverageTruncate, SystemOneCoverageFull)
	}
	switch rule.OnUnscanned {
	case "":
	case SystemOneUnscannedMatch, SystemOneUnscannedNoMatch, SystemOneUnscannedUnknown:
		if !rule.FullCoverage() {
			return fmt.Errorf("on_unscanned applies only to a rule with coverage: %s", SystemOneCoverageFull)
		}
	default:
		return fmt.Errorf("on_unscanned must be %s, %s or %s", SystemOneUnscannedMatch, SystemOneUnscannedNoMatch, SystemOneUnscannedUnknown)
	}
	return nil
}

// validateQuestionDeployment checks the deployment a systemone rule or
// selector asks: a declared model_runtime or systemone deployment, or, when
// it names none, the decision model. A model_runtime deployment answers
// questions, so its input is a scan budget or nothing.
func validateQuestionDeployment(cfg *RouterConfig, deployment, asker string) error {
	if deployment != "" {
		return validateDecisionModelReference(cfg, deployment)
	}
	if cfg.RemoteDecisionModel() {
		return nil
	}
	name, resolved, _, err := cfg.DecisionModelDeployment()
	if err != nil {
		return fmt.Errorf("default decision deployment for %s: %w", asker, err)
	}
	if err := resolved.ValidateDecisionInput(name); err != nil {
		return fmt.Errorf("default decision deployment for %s: %w", asker, err)
	}
	return nil
}

// validateSetLabelAnswerKeys rejects a rule whose name equals the answer key
// ("<rule>.<label>") of a Set question asked of the same deployment: the
// model answers each Set label under that key in the same call.
func validateSetLabelAnswerKeys(cfg *RouterConfig, rules []SystemOneRule) error {
	names := make(map[string]string, len(rules))
	for _, rule := range rules {
		names[cfg.SystemOneRuleDeployment(rule)+"\x00"+rule.Name] = rule.Name
	}
	for _, rule := range rules {
		if rule.Question.Type != SystemOneQuestionSet {
			continue
		}
		deployment := cfg.SystemOneRuleDeployment(rule)
		for _, label := range rule.Question.Labels {
			if other, exists := names[deployment+"\x00"+rule.Name+"."+label.Key]; exists {
				return fmt.Errorf("%s: the name collides with the answer key of set question %q's label %q on deployment %q; rename one of them", SystemOneRulePath(other), rule.Name, label.Key, deployment)
			}
		}
	}
	return nil
}

func validateSystemOneQuestion(question SystemOneQuestion) error {
	if strings.TrimSpace(question.Instructions) == "" {
		return fmt.Errorf("instructions are required")
	}
	if question.Type != SystemOneQuestionSet && question.Type != SystemOneQuestionSpan {
		if len(question.Labels) > 0 {
			return fmt.Errorf("labels apply only to set and span questions")
		}
		if question.Threshold != nil {
			return fmt.Errorf("threshold applies only to set and span questions")
		}
	}
	if question.Head != "" && question.Type != SystemOneQuestionSpan {
		return fmt.Errorf("head applies only to span questions")
	}
	switch question.Type {
	case SystemOneQuestionSet, SystemOneQuestionSpan:
		return validateLabelledSystemOneQuestion(question)
	case SystemOneQuestionChoice:
		if len(question.Levels) > 0 {
			return fmt.Errorf("levels apply only to score questions")
		}
		if len(question.Choices) < MinSystemOneChoices || len(question.Choices) > MaxSystemOneChoices {
			return fmt.Errorf("a choice question needs %d..%d choices", MinSystemOneChoices, MaxSystemOneChoices)
		}
		return validateSystemOneChoiceKeys(question.Choices, nil)
	case SystemOneQuestionNoul:
		if len(question.Levels) > 0 {
			return fmt.Errorf("levels apply only to score questions")
		}
		return validateSystemOneChoiceKeys(question.Choices, map[string]struct{}{"false": {}, "true": {}})
	case SystemOneQuestionScore:
		if len(question.Choices) > 0 {
			return fmt.Errorf("a score question takes ordered levels, not choices")
		}
		if len(question.Levels) < MinSystemOneLevels || len(question.Levels) > MaxSystemOneLevels {
			return fmt.Errorf("a score question needs %d..%d levels", MinSystemOneLevels, MaxSystemOneLevels)
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

// validateLabelledSystemOneQuestion checks a Set or Span question: 1..255
// unique labels, an optional threshold in [0, 1] and, for a span, an
// optional head.
func validateLabelledSystemOneQuestion(question SystemOneQuestion) error {
	if len(question.Choices) > 0 || len(question.Levels) > 0 {
		return fmt.Errorf("a %s question takes labels, not choices or levels", question.Type)
	}
	if len(question.Labels) < MinSystemOneLabels || len(question.Labels) > MaxSystemOneLabels {
		return fmt.Errorf("a %s question needs %d..%d labels", question.Type, MinSystemOneLabels, MaxSystemOneLabels)
	}
	if threshold := question.Threshold; threshold != nil && (math.IsNaN(*threshold) || *threshold < 0 || *threshold > 1) {
		return fmt.Errorf("threshold must be within [0, 1]")
	}
	if question.Head != "" && question.Head != SystemOneSpanHeadRouter && question.Head != SystemOneSpanHeadBroad {
		return fmt.Errorf("head must be %s or %s", SystemOneSpanHeadRouter, SystemOneSpanHeadBroad)
	}
	return validateSystemOneLabelKeys(question.Labels)
}

func validateSystemOneLabelKeys(labels []SystemOneOption) error {
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

func validateSystemOneChoiceKeys(choices []SystemOneOption, allowed map[string]struct{}) error {
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

// validateDecisionModelReference checks a deployment a systemone rule or
// selector names: a model_runtime deployment, or a systemone deployment.
func validateDecisionModelReference(cfg *RouterConfig, name string) error {
	if cfg.IsSystemOneDeployment(name) {
		return nil
	}
	return validateModelRuntimeReference(cfg, name)
}

func validateModelRuntimeReference(cfg *RouterConfig, name string) error {
	deployment, exists := cfg.ModelDeployments[name]
	if !exists {
		return fmt.Errorf("deployment %q is not declared in global.model_catalog.deployments", name)
	}
	if !deployment.IsModelRuntime() {
		return fmt.Errorf("deployment %q must use provider %s or %s", name, ModelRuntimeProvider, SystemOneProvider)
	}
	return deployment.ValidateDecisionInput(name)
}

func systemOneRuleByName(rules []SystemOneRule, name string) *SystemOneRule {
	for index := range rules {
		if rules[index].Name == name {
			return &rules[index]
		}
	}
	return nil
}

// validateSystemOneLeaf checks a systemone condition: named without a label
// when the rule matches itself, or with one of its options that counts. A
// response-stage rule is refused with every response-stage signal
// (validateDecisionsReadRequestStageSignals).
func validateSystemOneLeaf(cfg *RouterConfig, decisionName string, node *RuleNode) error {
	rule := systemOneRuleByName(cfg.SystemOneRules, node.Name)
	if rule == nil {
		return fmt.Errorf("decision '%s': systemone condition references unknown signal %q", decisionName, node.Name)
	}
	if node.Label == "" {
		if !rule.MatchesItself() {
			return fmt.Errorf("decision '%s': systemone condition %q names a choice rule without labels; name the option as the condition's label, or give the rule labels", decisionName, node.Name)
		}
		return nil
	}
	if !rule.Question.Labelled() {
		return fmt.Errorf("decision '%s': systemone condition %q is a %s question and takes no label", decisionName, node.Name, rule.Question.Type)
	}
	if !slices.Contains(rule.Question.OptionKeys(), node.Label) {
		return fmt.Errorf("decision '%s': systemone condition %q names label %q, which is not an option of the question", decisionName, node.Name, node.Label)
	}
	if slices.Contains(rule.IgnoreLabels, node.Label) {
		return fmt.Errorf("decision '%s': systemone condition %q names label %q, which the rule's ignore_labels never count", decisionName, node.Name, node.Label)
	}
	return nil
}

func validateSystemOneSelectorConfig(decisionName string, modelRefs []ModelRef, algorithm *AlgorithmConfig) error {
	cfg := algorithm.SystemOne
	if cfg == nil {
		return fmt.Errorf("decision '%s': algorithm.type=systemone requires algorithm.systemone configuration", decisionName)
	}
	path := fmt.Sprintf("decision '%s', algorithm.systemone", decisionName)
	if strings.TrimSpace(cfg.Deployment) != cfg.Deployment {
		return fmt.Errorf("%s: deployment must name a model_runtime or systemone deployment; omit it to ask the decision model", path)
	}
	if strings.TrimSpace(cfg.Instructions) == "" {
		return fmt.Errorf("%s: instructions are required", path)
	}
	if cfg.TimeoutMs < 0 || cfg.TimeoutMs > MaxSystemOneTimeoutMs {
		return fmt.Errorf("%s: timeout_ms must be within [1, %d] when set", path, MaxSystemOneTimeoutMs)
	}
	// A model-free built-in Recipe asset receives its candidates through
	// assignments; the choice range applies once they are bound.
	if len(modelRefs) > 0 && (len(modelRefs) < MinSystemOneChoices || len(modelRefs) > MaxSystemOneChoices) {
		return fmt.Errorf("%s: needs %d..%d modelRefs to choose from", path, MinSystemOneChoices, MaxSystemOneChoices)
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
