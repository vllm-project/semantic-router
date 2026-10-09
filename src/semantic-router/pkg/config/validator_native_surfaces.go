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
		return fmt.Errorf("native execution uses declared stages and routing.budget; Chat routing.fallback must be disabled")
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
	case SignalTypeAuthz, SignalTypeMetadata, SignalTypeConversation, SignalTypeReask,
		SignalTypeUserFeedback, SignalTypeInputModality:
		return fmt.Errorf("signal %q requires request context that native System One does not yet provide", kind)
	case SignalTypeKB, ProjectionInputKBMetric:
		return fmt.Errorf("signal %q does not yet participate in native request deadlines and call accounting", kind)
	}
	return nil
}

func validateNativeDecisionSurfaces(decision Decision) error {
	if len(decision.Plugins) > 0 || decision.Action != nil || decision.Fallback != nil || decision.Reliability != nil ||
		decision.OutputContract != "" || decision.OutputContractSpec != nil ||
		decision.Adaptations.Mode != "" || decision.Adaptations.Adaptation != nil || decision.Adaptations.Protection != nil ||
		len(decision.CandidateIterations) > 0 || len(decision.Emits) > 0 {
		return fmt.Errorf("decision %q: Chat plugins, actions, fallback, reliability, output contracts, adaptations, candidateIterations and emits are unsupported for native execution", decision.Name)
	}
	for _, ref := range decision.ModelRefs {
		if ref.Weight != 0 || ref.UseReasoning != nil || ref.ReasoningMode != "" || ref.ReasoningEffort != "" || ref.ReasoningDescription != "" {
			return fmt.Errorf("decision %q: native modelRefs declare model aliases only; weighting and Chat reasoning controls are unsupported", decision.Name)
		}
	}
	return nil
}

// Alternative preference/MCP adapters do not yet share native request
// cancellation and inference accounting. Explicit decision task bindings keep
// ordinary native preference available without inheriting those adapters.
func validateNativeSignalBackends(cfg *RouterConfig, profile RoutingProfile) error {
	if nativeReferencesSignal(profile, SignalTypeDomain) && cfg.MCPCategoryModel.Enabled {
		return fmt.Errorf("native domain signals do not support the MCP classifier; use a model deployment binding")
	}
	if !nativeReferencesSignal(profile, SignalTypePreference) {
		return nil
	}
	if !cfg.PreferenceUsesDecisionTask() {
		return fmt.Errorf("native preference requires a decision task deployment; contrastive and external preference adapters do not support native request accounting")
	}
	bindings := cfg.EffectiveModelBindings(cfg.Signals, cfg.ModelBindings)
	if binding, explicit := bindings["preference"]; explicit {
		deployment, ok := cfg.ModelDeployments[binding.Deployment]
		if !ok || !deployment.IsModelRuntime() || binding.Head != "" || binding.OperatingPoint != nil {
			return fmt.Errorf("native preference requires a model_runtime decision task binding without a classifier head or operating point")
		}
		return nil
	}
	_, deployment, ok, err := cfg.DecisionModelDeployment()
	if err != nil || !ok || !deployment.IsModelRuntime() {
		return fmt.Errorf("native preference requires an available decision model deployment")
	}
	return nil
}

func nativeReferencesSignal(profile RoutingProfile, kind string) bool {
	var references func(RuleNode) bool
	references = func(node RuleNode) bool {
		if node.Type == kind {
			return true
		}
		for _, child := range node.Conditions {
			if references(child) {
				return true
			}
		}
		return false
	}
	for _, decision := range profile.Decisions {
		if references(decision.Rules) {
			return true
		}
	}
	for _, score := range profile.Projections.Scores {
		for _, input := range score.Inputs {
			if input.Type == kind {
				return true
			}
		}
	}
	return false
}
