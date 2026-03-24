package config

import (
	"fmt"
	"strings"
)

func validateMetaRoutingContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	return ValidateMetaRoutingConfig(cfg.MetaRouting)
}

// ValidateMetaRoutingConfig validates the public routing.meta contract.
func ValidateMetaRoutingConfig(meta MetaRoutingConfig) error {
	if !meta.Enabled() {
		if meta.MaxPasses > 0 || meta.TriggerPolicy != nil || len(meta.AllowedActions) > 0 {
			return fmt.Errorf("routing.meta.mode is required when routing.meta config is set")
		}
		return nil
	}

	mode := strings.TrimSpace(strings.ToLower(meta.Mode))
	if !IsSupportedMetaRoutingMode(mode) {
		return fmt.Errorf(
			"routing.meta.mode %q must be one of %s",
			meta.Mode,
			strings.Join(SupportedMetaRoutingModes(), ", "),
		)
	}

	if meta.MaxPasses < 0 {
		return fmt.Errorf("routing.meta.max_passes must be >= 0, got %d", meta.MaxPasses)
	}

	if meta.TriggerPolicy != nil {
		if err := validateMetaTriggerPolicy(meta.TriggerPolicy); err != nil {
			return err
		}
	}

	for i, action := range meta.AllowedActions {
		if err := validateMetaRefinementAction(action, i); err != nil {
			return err
		}
	}

	return nil
}

func validateMetaTriggerPolicy(policy *MetaTriggerPolicy) error {
	if policy == nil {
		return nil
	}

	if err := validateMetaUnitInterval(policy.DecisionMarginBelow, "routing.meta.trigger_policy.decision_margin_below"); err != nil {
		return err
	}
	if err := validateMetaUnitInterval(policy.ProjectionBoundaryWithin, "routing.meta.trigger_policy.projection_boundary_within"); err != nil {
		return err
	}

	for i, family := range policy.RequiredFamilies {
		prefix := fmt.Sprintf("routing.meta.trigger_policy.required_families[%d]", i)
		if err := validateMetaRoutingSignalFamily(prefix+".type", family.Type); err != nil {
			return err
		}
		if err := validateMetaUnitInterval(family.MinConfidence, prefix+".min_confidence"); err != nil {
			return err
		}
		if family.MinMatches != nil && *family.MinMatches < 0 {
			return fmt.Errorf("%s.min_matches must be >= 0, got %d", prefix, *family.MinMatches)
		}
	}

	for i, disagreement := range policy.FamilyDisagreements {
		prefix := fmt.Sprintf("routing.meta.trigger_policy.family_disagreements[%d]", i)
		if err := validateMetaRoutingSignalFamily(prefix+".cheap", disagreement.Cheap); err != nil {
			return err
		}
		if err := validateMetaRoutingSignalFamily(prefix+".expensive", disagreement.Expensive); err != nil {
			return err
		}
		if strings.EqualFold(strings.TrimSpace(disagreement.Cheap), strings.TrimSpace(disagreement.Expensive)) {
			return fmt.Errorf("%s must compare two distinct signal families", prefix)
		}
	}

	return nil
}

func validateMetaRefinementAction(action MetaRefinementAction, index int) error {
	prefix := fmt.Sprintf("routing.meta.allowed_actions[%d]", index)
	actionType := strings.TrimSpace(strings.ToLower(action.Type))
	if !IsSupportedMetaRoutingActionType(actionType) {
		return fmt.Errorf(
			"%s.type %q must be one of %s",
			prefix,
			action.Type,
			strings.Join(SupportedMetaRoutingActionTypes(), ", "),
		)
	}

	switch actionType {
	case MetaRoutingActionDisableCompression:
		if len(action.SignalFamilies) > 0 {
			return fmt.Errorf("%s.signal_families is only supported for type=%q", prefix, MetaRoutingActionRerunSignalFamilies)
		}
	case MetaRoutingActionRerunSignalFamilies:
		if len(action.SignalFamilies) == 0 {
			return fmt.Errorf("%s.signal_families must not be empty for type=%q", prefix, MetaRoutingActionRerunSignalFamilies)
		}
		for familyIndex, family := range action.SignalFamilies {
			if err := validateMetaRoutingSignalFamily(
				fmt.Sprintf("%s.signal_families[%d]", prefix, familyIndex),
				family,
			); err != nil {
				return err
			}
		}
	}

	return nil
}

func validateMetaRoutingSignalFamily(field string, signalFamily string) error {
	normalized := strings.TrimSpace(strings.ToLower(signalFamily))
	if IsSupportedSignalType(normalized) {
		return nil
	}
	return fmt.Errorf(
		"%s %q must be one of %s",
		field,
		signalFamily,
		strings.Join(SupportedSignalTypes(), ", "),
	)
}

func validateMetaUnitInterval(value *float64, field string) error {
	if value == nil {
		return nil
	}
	if *value < 0 || *value > 1 {
		return fmt.Errorf("%s must be between 0 and 1, got %v", field, *value)
	}
	return nil
}
