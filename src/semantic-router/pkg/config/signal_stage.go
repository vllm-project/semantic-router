package config

import "strings"

// SignalStage is the point in a request's lifecycle at which a signal rule is
// observed.
type SignalStage string

const (
	// SignalStageRequest rules are scored from the request, before a model is
	// selected. Every rule is request-stage unless it says otherwise.
	SignalStageRequest SignalStage = "request"
	// SignalStageResponse rules are scored from the model's output, so they do
	// not exist while the request-stage decision is being made.
	SignalStageResponse SignalStage = "response"
)

// Recognised values for JailbreakRule.Direction. They name the stage the rule
// observes: "request" scores the prompt, "response" scores the model's output.
const (
	SignalDirectionRequest  = string(SignalStageRequest)
	SignalDirectionResponse = string(SignalStageResponse)
)

// Stage reports the stage a jailbreak rule is observed at. An empty direction
// is the request stage, which is what every rule was before direction existed.
func (r JailbreakRule) Stage() SignalStage {
	if r.Direction == SignalDirectionResponse {
		return SignalStageResponse
	}
	return SignalStageRequest
}

// RequestJailbreakRules returns the jailbreak rules scored from the request.
func (c *RouterConfig) RequestJailbreakRules() []JailbreakRule {
	return c.jailbreakRulesAt(SignalStageRequest)
}

// ResponseJailbreakRules returns the jailbreak rules scored from the model's
// output.
func (c *RouterConfig) ResponseJailbreakRules() []JailbreakRule {
	return c.jailbreakRulesAt(SignalStageResponse)
}

func (c *RouterConfig) jailbreakRulesAt(stage SignalStage) []JailbreakRule {
	if c == nil {
		return nil
	}
	var rules []JailbreakRule
	for _, rule := range c.JailbreakRules {
		if rule.Stage() == stage {
			rules = append(rules, rule)
		}
	}
	return rules
}

// SignalStageOf reports the stage a {type, name} condition is observable at.
//
// The stage sits on the rule rather than on the type, so the "type:name" key,
// and with it SignalConfidences, SignalErrors and the condition shape, are
// exactly what they are for a request-stage rule. Only jailbreak rules carry a
// direction today; every other type is request-stage.
func (c *RouterConfig) SignalStageOf(signalType, name string) SignalStage {
	if c == nil || !strings.EqualFold(signalType, SignalTypeJailbreak) {
		return SignalStageRequest
	}
	for _, rule := range c.JailbreakRules {
		if rule.Name == name {
			return rule.Stage()
		}
	}
	return SignalStageRequest
}

// DecisionStage reports the earliest stage at which a decision's rules can be
// evaluated. A decision that reads any response-stage signal cannot be resolved
// while the request is still being routed, so it is a response-stage decision.
//
// This is deliberately not left to on_unknown. That policy means "evaluated and
// could not resolve", which is what SignalErrors records for a detector that
// failed. A signal whose stage has not been reached yet has not been evaluated
// at all, and a rule with on_unknown: match would otherwise fire at the request
// stage on a signal that never ran.
func (c *RouterConfig) DecisionStage(rules *RuleNode) SignalStage {
	if rules != nil && c.ruleNodeReadsResponseSignal(rules) {
		return SignalStageResponse
	}
	return SignalStageRequest
}

func (c *RouterConfig) ruleNodeReadsResponseSignal(node *RuleNode) bool {
	if node == nil {
		return false
	}
	if node.IsLeaf() {
		return c.SignalStageOf(node.Type, node.Name) == SignalStageResponse
	}
	for i := range node.Conditions {
		if c.ruleNodeReadsResponseSignal(&node.Conditions[i]) {
			return true
		}
	}
	return false
}

// DecisionsAtStage returns the decisions evaluated at stage, in configured
// order. A request-stage evaluation must not see a response-stage decision and
// a response-stage evaluation must not re-select a request-stage one.
//
// The input slice is returned as is when every decision is at stage, which is
// every configuration without a response-direction rule, so the decision a
// caller gets back keeps pointing into the slice it passed in.
func (c *RouterConfig) DecisionsAtStage(decisions []Decision, stage SignalStage) []Decision {
	staged := make([]Decision, 0, len(decisions))
	for i := range decisions {
		if c.DecisionStage(&decisions[i].Rules) == stage {
			staged = append(staged, decisions[i])
		}
	}
	if len(staged) == len(decisions) {
		return decisions
	}
	if len(staged) == 0 {
		return nil
	}
	return staged
}
