package config

import "sort"

// SignalStage is the point in a request's lifecycle at which a signal type can
// be observed.
type SignalStage string

const (
	// SignalStageRequest signals are computed from the request, before a model
	// is selected. Every signal type is request-stage unless listed below.
	SignalStageRequest SignalStage = "request"
	// SignalStageResponse signals are computed from the model's output, so they
	// do not exist while the request-stage decision is being made.
	SignalStageResponse SignalStage = "response"
)

// responseStageSignalTypes lists the signal types observable only after the
// model has answered. Keeping the stage on the type rather than on each rule
// leaves the "type:name" key, and therefore SignalConfidences, SignalErrors and
// the {type, name} condition shape, exactly as they are.
var responseStageSignalTypes = map[string]struct{}{
	SignalTypeResponseJailbreak: {},
}

// SignalStageOf reports the stage a signal type is observable at.
func SignalStageOf(signalType string) SignalStage {
	if _, ok := responseStageSignalTypes[signalType]; ok {
		return SignalStageResponse
	}
	return SignalStageRequest
}

// IsResponseStageSignal reports whether a signal type is only observable after
// the model has answered.
func IsResponseStageSignal(signalType string) bool {
	return SignalStageOf(signalType) == SignalStageResponse
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
func DecisionStage(rules *RuleNode) SignalStage {
	if rules != nil && ruleNodeReadsResponseSignal(rules) {
		return SignalStageResponse
	}
	return SignalStageRequest
}

func ruleNodeReadsResponseSignal(node *RuleNode) bool {
	if node == nil {
		return false
	}
	if node.IsLeaf() {
		return IsResponseStageSignal(node.Type)
	}
	for i := range node.Conditions {
		if ruleNodeReadsResponseSignal(&node.Conditions[i]) {
			return true
		}
	}
	return false
}

// responseStageSignalTypeNames lists the response-stage types in a stable order
// for diagnostics.
func responseStageSignalTypeNames() []string {
	names := make([]string, 0, len(responseStageSignalTypes))
	for name := range responseStageSignalTypes {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
