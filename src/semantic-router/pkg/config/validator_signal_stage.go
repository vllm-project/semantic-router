package config

import (
	"fmt"
)

// validateSignalStageContracts checks what a response-direction signal rule
// implies for the decisions around it.
func validateSignalStageContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	return validateDecisionsReadRequestStageSignals(cfg)
}

// validateDecisionsReadRequestStageSignals rejects a decision that reads a
// response-stage rule (a jailbreak rule with direction: response, or any
// hallucination rule), directly in its rule tree or through a projection it
// reads. Decisions and projections are evaluated while the request is being
// routed and the model has not answered, so the rule could only ever read as
// unknown there, and a projection would turn that into its configured miss
// value; the selected decision's response_guard reads the rule once the
// response exists. Caught at load rather than as a decision that silently
// never matches.
func validateDecisionsReadRequestStageSignals(cfg *RouterConfig) error {
	for _, decision := range cfg.AllRoutingDecisions() {
		ref, ok := cfg.decisionReadsResponseSignal(&decision.Rules)
		if !ok {
			continue
		}
		through := ""
		if ref.Via != "" {
			through = fmt.Sprintf(" through projection %q", ref.Via)
		}
		return fmt.Errorf("decision %q reads %s rule %q%s, which is observed at the response stage; a response-stage rule is read by the selected decision's response_guard, not by decision rules", decision.Name, ref.Type, ref.Name, through)
	}
	return nil
}
