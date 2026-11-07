package config

import (
	"fmt"
	"strings"
)

// Native requests provide a complete task document, not a Chat conversation or
// trusted identity envelope. Reject unavailable contexts rather than evaluating
// their rules against invented zero values. KB additionally needs request-bound
// inference accounting before it can participate in the shared native budget.
func validateNativeProfileSurfaces(profile RoutingProfile) error {
	if profile.CandidateRequirements != nil {
		return fmt.Errorf("native execution does not support candidate_requirements; declare stages for the supported task")
	}
	if profile.Fallback != nil && profile.Fallback.Enabled {
		return fmt.Errorf("native execution uses declared stages and algorithm.budget; Chat routing.fallback must be disabled")
	}
	for _, score := range profile.Projections.Scores {
		for _, input := range score.Inputs {
			if err := validateNativeSignalReference(input.Type); err != nil {
				return fmt.Errorf("projection score %q: %w", score.Name, err)
			}
		}
	}
	for _, decision := range profile.Decisions {
		if err := validateNativeRuleContext(decision.Rules); err != nil {
			return fmt.Errorf("decision %q: %w", decision.Name, err)
		}
	}
	return nil
}

func validateNativeRuleContext(node RuleNode) error {
	if err := validateNativeSignalReference(node.Type); err != nil {
		return err
	}
	for _, child := range node.Conditions {
		if err := validateNativeRuleContext(child); err != nil {
			return err
		}
	}
	return nil
}

func validateNativeSignalReference(kind string) error {
	switch strings.ToLower(strings.TrimSpace(kind)) {
	case SignalTypeAuthz, SignalTypeMetadata, SignalTypeConversation, SignalTypeTurnRelation,
		SignalTypeUserFeedback, SignalTypeInputModality:
		return fmt.Errorf("signal %q requires request context that native System One does not yet provide", kind)
	}
	return nil
}

func validateNativeDecisionSurfaces(decision Decision) error {
	if len(decision.Plugins) > 0 || decision.Action != nil || decision.Fallback != nil || decision.Reliability != nil ||
		decision.OutputContract != "" || decision.OutputContractSpec != nil ||
		decision.Adaptations.Mode != "" || decision.Adaptations.Adaptation != nil || decision.Adaptations.Protection != nil ||
		len(decision.CandidateIterations) > 0 || decision.Retention != nil {
		return fmt.Errorf("decision %q: Chat plugins, actions, fallback, reliability, output contracts, adaptations, candidateIterations and retention are unsupported for native execution", decision.Name)
	}
	for _, ref := range decision.ModelRefs {
		if ref.Weight != 0 || ref.UseReasoning != nil || ref.ReasoningMode != "" || ref.ReasoningEffort != "" || ref.ReasoningDescription != "" {
			return fmt.Errorf("decision %q: native modelRefs declare model aliases only; weighting and Chat reasoning controls are unsupported", decision.Name)
		}
	}
	return nil
}
