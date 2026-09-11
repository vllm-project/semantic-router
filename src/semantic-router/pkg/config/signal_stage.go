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
// exactly what they are for a request-stage rule. Jailbreak rules carry a
// direction; hallucination rules check the model's answer and so are always
// response-stage; every other type is request-stage.
func (c *RouterConfig) SignalStageOf(signalType, name string) SignalStage {
	if c == nil {
		return SignalStageRequest
	}
	if strings.EqualFold(signalType, SignalTypeHallucination) {
		return SignalStageResponse
	}
	if !strings.EqualFold(signalType, SignalTypeJailbreak) {
		return SignalStageRequest
	}
	for _, rule := range c.JailbreakRules {
		if rule.Name == name {
			return rule.Stage()
		}
	}
	return SignalStageRequest
}

// decisionReadsResponseSignal reports the first response-direction rule a
// decision's rule tree reads, wherever it sits in the tree, and the projection
// output it is read through when the reference is indirect. Decisions are
// selected while the request is being routed, before the model has answered,
// so such a rule is not a decision input: the selected decision's response
// plugins consume the observation instead. Config validation rejects the
// reference rather than leaving a decision that can never match.
func (c *RouterConfig) decisionReadsResponseSignal(node *RuleNode) (ref responseSignalRef, ok bool) {
	if node == nil {
		return responseSignalRef{}, false
	}
	if node.IsLeaf() {
		return c.leafReadsResponseSignal(node)
	}
	for i := range node.Conditions {
		if ref, ok := c.decisionReadsResponseSignal(&node.Conditions[i]); ok {
			return ref, true
		}
	}
	return responseSignalRef{}, false
}

// responseSignalRef names a response-stage rule a decision reads: its type,
// its name, and the projection output it is read through when the reference
// is indirect.
type responseSignalRef struct {
	Type string
	Name string
	Via  string
}

// leafReadsResponseSignal resolves one condition: a projection through the
// scores behind its output, any other signal by the stage of its rule.
func (c *RouterConfig) leafReadsResponseSignal(node *RuleNode) (ref responseSignalRef, ok bool) {
	if strings.EqualFold(strings.TrimSpace(node.Type), SignalTypeProjection) {
		ref, ok = c.projectionReadsResponseSignal(node.Name)
		ref.Via = node.Name
	} else if c.SignalStageOf(node.Type, node.Name) == SignalStageResponse {
		ref, ok = responseSignalRef{Type: strings.ToLower(strings.TrimSpace(node.Type)), Name: node.Name}, true
	}
	if !ok {
		return responseSignalRef{}, false
	}
	return ref, true
}

// projectionReadsResponseSignal reports the first response-direction rule that
// feeds the projection output a decision reads, through any depth of
// projection scores. A projection is evaluated with the decisions, before the
// model has answered; an input with no result yet takes its configured miss
// value, so a response-direction rule behind a projection would shape the
// decision as a silent miss on every request.
func (c *RouterConfig) projectionReadsResponseSignal(outputName string) (responseSignalRef, bool) {
	sourceByOutput := projectionSourcesByOutput(c.Projections.Mappings)
	scoreByName := projectionScoresByName(c.Projections.Scores)
	scoreName, ok := sourceByOutput[strings.ToLower(strings.TrimSpace(outputName))]
	if !ok {
		return responseSignalRef{}, false
	}
	return c.projectionScoreReadsResponseSignal(scoreName, sourceByOutput, scoreByName, map[string]bool{})
}

func (c *RouterConfig) projectionScoreReadsResponseSignal(
	scoreName string,
	sourceByOutput map[string]string,
	scoreByName map[string]ProjectionScore,
	visiting map[string]bool,
) (responseSignalRef, bool) {
	name := strings.ToLower(strings.TrimSpace(scoreName))
	score, ok := scoreByName[name]
	if !ok || visiting[name] {
		// Unknown scores and cycles are projection validation's to reject;
		// this walk only has to terminate on them.
		return responseSignalRef{}, false
	}
	visiting[name] = true
	defer delete(visiting, name)
	for _, input := range score.Inputs {
		inputType := strings.ToLower(strings.TrimSpace(input.Type))
		switch inputType {
		case ProjectionInputKBMetric:
			continue
		case SignalTypeProjection:
			dependency := strings.ToLower(strings.TrimSpace(input.Name))
			if strings.EqualFold(strings.TrimSpace(input.ValueSource), ProjectionValueSourceConfidence) {
				dependency = sourceByOutput[dependency]
			}
			if ref, ok := c.projectionScoreReadsResponseSignal(dependency, sourceByOutput, scoreByName, visiting); ok {
				return ref, true
			}
		default:
			if c.SignalStageOf(input.Type, input.Name) == SignalStageResponse {
				return responseSignalRef{Type: inputType, Name: input.Name}, true
			}
		}
	}
	return responseSignalRef{}, false
}
