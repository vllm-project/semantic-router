package config

import (
	"fmt"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// ApplicationRule is an application signal rule compiled into the systemone
// rule that asks its preset: the preset's question, stage, input and
// coverage, the rule's labels (by default the preset's positive option) and
// predicate (by default the threshold calibrated for the preset on the model
// the rule resolves to), and its deployment: the rule's own, its recipe's
// binding of its type, the global one, else the decision model. Rules of one
// preset asked of one deployment about one state share one question.
type ApplicationRule struct {
	// Type is the application signal type, which conditions name.
	Type string
	// Preset is the preset the rule asks.
	Preset string
	// Rule asks the preset's question under the application rule's name.
	Rule SystemOneRule
	// Uncalibrated marks a rule that takes the neutral predicate because the
	// model it resolves to has no calibrated threshold for the preset.
	Uncalibrated bool
	// FailClosed marks a request-stage jailbreak or pii rule with on_unknown:
	// match, which matches when its deployment gives no answer, so a guard
	// whose model is down does not pass the request.
	FailClosed bool
}

// ApplicationCollections are the collections a configuration writes each
// application signal type's rules in, under routing.signals.
var ApplicationCollections = map[string]string{
	SignalTypeDomain:        "domains",
	SignalTypeFactCheck:     "fact_check",
	SignalTypeUserFeedback:  "user_feedbacks",
	SignalTypeJailbreak:     "jailbreak",
	SignalTypeSafety:        "safety",
	SignalTypePII:           "pii",
	SignalTypeHallucination: "hallucination",
	SignalTypeRefusal:       "refusal",
	SignalTypeTurnRelation:  "turn_relation",
	SignalTypeComplexity:    "complexity",
	SignalTypeModality:      "modality",
}

// IsApplicationSignalType reports whether signalType is an application signal.
func IsApplicationSignalType(signalType string) bool {
	_, ok := ApplicationCollections[signalType]
	return ok
}

// ApplicationRulePath is where a scope's configuration writes a rule.
func (c *RouterConfig) ApplicationRulePath(signalType, name string) string {
	return c.ScopePath("signals." + ApplicationCollections[signalType] + "[" + name + "]")
}

// applicationRuleSpec is what an application rule says about the systemone
// rule it compiles into.
type applicationRuleSpec struct {
	signalType, preset, stage     string
	name, description, deployment string
	labels, ignoreLabels          []string
	predicate                     *NumericPredicate
	input                         SystemOneInput
	onUnscanned, onUnknown        string
	timeoutMs                     int
}

// failsClosed reports whether the compiled rule matches when its deployment
// gives no answer: a request-stage jailbreak or pii rule with on_unknown:
// match.
func (spec applicationRuleSpec) failsClosed() bool {
	guard := spec.signalType == SignalTypeJailbreak || spec.signalType == SignalTypePII
	return guard && spec.stage != SystemOneStageResponse && spec.onUnknown == string(RuleOnUnknownMatch)
}

// ApplicationRules compiles the scope's application signal rules, by type in
// catalog order and in rule order within a type.
func (c *RouterConfig) ApplicationRules() []ApplicationRule {
	if c == nil {
		return nil
	}
	specs := c.applicationRuleSpecs()
	rules := make([]ApplicationRule, 0, len(specs))
	for _, spec := range specs {
		rules = append(rules, c.compileApplicationRule(spec))
	}
	return rules
}

// AsksApplicationRule reports whether a request of this scope asks the rule's
// question: a decision reads its type, a response plugin asks it at the
// response stage, or a masking plugin names the pii rule.
func (c *RouterConfig) AsksApplicationRule(rule ApplicationRule) bool {
	return rule.Rule.EffectiveStage() == SystemOneStageResponse || c.UsesSignalTypeInReachableRouting(rule.Type) ||
		(rule.Type == SignalTypePII && slices.Contains(c.MaskingSignals(), rule.Rule.Name))
}

// applicationRuleSpecs reads what each application rule says, by type in
// catalog order and in rule order within a type.
func (c *RouterConfig) applicationRuleSpecs() []applicationRuleSpec {
	var specs []applicationRuleSpec
	for _, r := range c.Categories {
		labels := r.Labels
		if len(labels) == 0 {
			labels = []string{r.Name}
		}
		specs = append(specs, applicationRuleSpec{
			signalType: SignalTypeDomain, preset: presets.Domain, name: r.Name, description: r.Description,
			deployment: r.Deployment, labels: labels, predicate: r.Predicate, timeoutMs: r.TimeoutMs,
		})
	}
	for _, r := range c.FactCheckRules {
		specs = append(specs, applicationRuleSpec{
			signalType: SignalTypeFactCheck, preset: presets.FactCheck, name: r.Name, description: r.Description,
			deployment: r.Deployment, labels: r.Labels, predicate: r.Predicate, timeoutMs: r.TimeoutMs,
		})
	}
	for _, r := range c.UserFeedbackRules {
		specs = append(specs, applicationRuleSpec{
			signalType: SignalTypeUserFeedback, preset: presets.Feedback, name: r.Name, description: r.Description,
			deployment: r.Deployment, labels: r.Labels, predicate: r.Predicate, timeoutMs: r.TimeoutMs,
		})
	}
	for _, r := range c.JailbreakRules {
		specs = append(specs, applicationRuleSpec{
			signalType: SignalTypeJailbreak, preset: presets.Jailbreak, stage: r.EffectiveStage(), name: r.Name,
			description: r.Description, deployment: r.Deployment, predicate: r.Predicate, input: SystemOneInput{History: r.Input.History},
			onUnscanned: SystemOneUnscannedMatch, onUnknown: r.OnUnknown, timeoutMs: r.TimeoutMs,
		})
	}
	for _, r := range c.SafetyRules {
		preset := presets.Harm
		if len(r.Labels) > 0 {
			preset = presets.HarmCategories
		}
		specs = append(specs, applicationRuleSpec{
			signalType: SignalTypeSafety, preset: preset, stage: r.EffectiveStage(), name: r.Name,
			description: r.Description, deployment: r.Deployment, labels: r.Labels, predicate: r.Predicate, timeoutMs: r.TimeoutMs,
		})
	}
	for _, r := range c.PIIRules {
		specs = append(specs, applicationRuleSpec{
			signalType: SignalTypePII, preset: presets.PII, name: r.Name, description: r.Description,
			deployment: r.Deployment, ignoreLabels: r.IgnoreLabels, predicate: r.Predicate,
			input: SystemOneInput{History: r.Input.History, ToolResults: r.Input.ToolResults, Request: r.Input.Request}, onUnscanned: SystemOneUnscannedMatch,
			onUnknown: r.OnUnknown, timeoutMs: r.TimeoutMs,
		})
	}
	for _, r := range c.HallucinationRules {
		specs = append(specs, applicationRuleSpec{
			signalType: SignalTypeHallucination, preset: presets.Hallucination, stage: presets.StageResponse, name: r.Name,
			description: r.Description, deployment: r.Deployment, predicate: r.Predicate, timeoutMs: r.TimeoutMs,
		})
	}
	for _, r := range c.RefusalRules {
		specs = append(specs, applicationRuleSpec{
			signalType: SignalTypeRefusal, preset: presets.Refusal, stage: presets.StageResponse, name: r.Name,
			description: r.Description, deployment: r.Deployment, predicate: r.Predicate, timeoutMs: r.TimeoutMs,
		})
	}
	for _, r := range c.TurnRelationRules {
		specs = append(specs, applicationRuleSpec{
			signalType: SignalTypeTurnRelation, preset: presets.TurnRelation, name: r.Name, description: r.Description,
			deployment: r.Deployment, labels: r.Labels, input: SystemOneInput{PreviousTurns: r.Input.PreviousTurns}, timeoutMs: r.TimeoutMs,
		})
	}
	for _, r := range c.ComplexityRules {
		specs = append(specs, applicationRuleSpec{
			signalType: SignalTypeComplexity, preset: presets.Complexity, name: r.Name, description: r.Description,
			deployment: r.Deployment, labels: r.Labels, timeoutMs: r.TimeoutMs,
		})
	}
	for _, r := range c.ModalityRules {
		specs = append(specs, applicationRuleSpec{
			signalType: SignalTypeModality, preset: presets.Modality, name: r.Name, description: r.Description,
			deployment: r.Deployment, labels: r.Labels, predicate: r.Predicate, timeoutMs: r.TimeoutMs,
		})
	}
	return specs
}

// ApplicationRuleByName returns the compiled rule of one application signal
// rule.
func (c *RouterConfig) ApplicationRuleByName(signalType, name string) (ApplicationRule, bool) {
	for _, rule := range c.ApplicationRules() {
		if rule.Type == signalType && rule.Rule.Name == name {
			return rule, true
		}
	}
	return ApplicationRule{}, false
}

func (c *RouterConfig) compileApplicationRule(spec applicationRuleSpec) ApplicationRule {
	preset, _ := presets.Lookup(spec.preset, spec.stage)
	input := spec.input
	input.History = input.History || preset.Input.History
	input.Answer, input.Context = preset.Input.Answer, preset.Input.Context
	input.PreviousAnswer = preset.Input.PreviousAnswer
	input.ToolResults = input.ToolResults || preset.Input.ToolResults
	if input.PreviousTurns == 0 {
		input.PreviousTurns = preset.Input.PreviousTurns
	}
	rule := SystemOneRule{
		Name: spec.name, Description: spec.description, Question: PresetSystemOneQuestion(preset),
		Labels: slices.Clone(spec.labels), IgnoreLabels: slices.Clone(spec.ignoreLabels), Predicate: spec.predicate,
		Deployment: c.applicationDeployment(spec.signalType, spec.deployment), Stage: preset.Stage, Input: input,
		TimeoutMs: spec.timeoutMs,
	}
	if len(rule.Labels) == 0 {
		rule.Labels = slices.Clone(preset.Positive)
	}
	if preset.Full {
		rule.Coverage, rule.OnUnscanned = SystemOneCoverageFull, spec.onUnscanned
	}
	compiled := ApplicationRule{Type: spec.signalType, Preset: spec.preset, Rule: rule, FailClosed: spec.failsClosed()}
	if rule.Predicate == nil && calibratedPredicateApplies(preset, rule.Labels) {
		threshold, calibrated := c.calibratedThreshold(c.SystemOneRuleDeployment(rule), spec.preset, preset.Stage)
		if !calibrated {
			threshold, compiled.Uncalibrated = presets.NeutralThreshold, true
		}
		compiled.Rule.Predicate = &NumericPredicate{GTE: &threshold}
	}
	return compiled
}

// PresetSystemOneQuestion is a preset's canonical question as a systemone
// question.
func PresetSystemOneQuestion(preset presets.Preset) SystemOneQuestion {
	question := SystemOneQuestion{Type: preset.Type, Instructions: preset.Instructions}
	options := make([]SystemOneOption, len(preset.Options))
	for index, option := range preset.Options {
		options[index] = SystemOneOption{Key: option.Key, Description: option.Description}
	}
	if preset.Type == presets.Set || preset.Type == presets.Span {
		question.Labels = options
	} else {
		question.Choices = options
	}
	return question
}

// calibratedPredicateApplies reports whether a preset's calibrated threshold
// is the predicate of a rule that sets none: for the domain, feedback,
// modality and pii presets it is a floor on whichever option a rule matches,
// for the others a bound on the preset's positive option. The hallucination
// preset's spans keep the model's own thresholds.
func calibratedPredicateApplies(preset presets.Preset, labels []string) bool {
	switch preset.ID {
	case presets.Domain, presets.Feedback, presets.Modality, presets.PII:
		return true
	case presets.Hallucination:
		return false
	}
	return len(preset.Positive) > 0 && slices.Equal(labels, preset.Positive)
}

// calibratedThreshold is the threshold calibrated for a preset in a stage on
// the model a model_runtime deployment serves; a remote model has none.
func (c *RouterConfig) calibratedThreshold(deployment, preset, stage string) (float64, bool) {
	resolved, ok := c.ModelDeployments[deployment]
	if !ok || !resolved.IsModelRuntime() {
		return 0, false
	}
	spec := registeredModel(resolved.ServedModel(deployment))
	if spec == nil {
		return 0, false
	}
	return presets.Threshold(spec.RepoID, preset, stage)
}

// applicationDeployment is the deployment an application rule names, else
// its recipe's binding of its type, else the global one; empty leaves the
// rule to the decision model.
func (c *RouterConfig) applicationDeployment(signalType, deployment string) string {
	if named := strings.TrimSpace(deployment); named != "" {
		return named
	}
	for _, bindings := range []map[string]ModelBinding{c.recipeBindings(), c.GlobalModelBindings} {
		if binding, ok := bindings[signalType]; ok {
			return binding.Deployment
		}
	}
	return ""
}

// ApplicationTypeDeployment is the deployment that answers a scope's rules of
// an application signal type that name none, and where it is chosen:
// "recipe" or "global" for a binding of the type, "default" for the
// decision model.
func (c *RouterConfig) ApplicationTypeDeployment(signalType string) (string, string) {
	binding, source := c.ApplicationTypeBinding(signalType)
	return binding.Deployment, source
}

// ApplicationTypeBinding is the binding ApplicationTypeDeployment reads, as
// written; the decision model's for "default".
func (c *RouterConfig) ApplicationTypeBinding(signalType string) (ModelBinding, string) {
	if binding, ok := c.recipeBindings()[signalType]; ok {
		return binding, "recipe"
	}
	if binding, ok := c.GlobalModelBindings[signalType]; ok {
		return binding, "global"
	}
	return ModelBinding{Deployment: c.deploymentOrDecisionModel("")}, "default"
}

// PresetThreshold is the threshold calibrated for a preset on the model an
// application signal type's rules that name no deployment resolve to, else
// the neutral one.
func (c *RouterConfig) PresetThreshold(signalType, preset string) float64 {
	deployment, _ := c.ApplicationTypeDeployment(signalType)
	if threshold, ok := c.calibratedThreshold(deployment, preset, presets.StageRequest); ok {
		return threshold
	}
	return presets.NeutralThreshold
}

// uncalibratedPresetWarnings name the application rules a scope asks with
// the neutral predicate.
func (c *RouterConfig) uncalibratedPresetWarnings() []ConfigWarning {
	var warnings []ConfigWarning
	for _, rule := range c.ApplicationRules() {
		if !rule.Uncalibrated {
			continue
		}
		warnings = append(warnings, ConfigWarning{
			Code: CodeUncalibratedPreset, Field: c.ApplicationRulePath(rule.Type, rule.Rule.Name),
			Message: fmt.Sprintf("resolves to deployment %q, whose model has no calibrated threshold for the %s preset; it matches with predicate gte %.2f until the rule sets a predicate",
				c.SystemOneRuleDeployment(rule.Rule), rule.Preset, presets.NeutralThreshold),
		})
	}
	return warnings
}
