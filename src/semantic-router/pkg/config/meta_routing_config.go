package config

import (
	"sort"
	"strings"
)

const (
	MetaRoutingModeObserve = "observe"
	MetaRoutingModeShadow  = "shadow"
	MetaRoutingModeActive  = "active"
)

const (
	MetaRoutingActionDisableCompression  = "disable_compression"
	MetaRoutingActionRerunSignalFamilies = "rerun_signal_families"
)

var supportedMetaRoutingModes = []string{
	MetaRoutingModeObserve,
	MetaRoutingModeShadow,
	MetaRoutingModeActive,
}

var supportedMetaRoutingActionTypes = []string{
	MetaRoutingActionDisableCompression,
	MetaRoutingActionRerunSignalFamilies,
}

// MetaRoutingConfig configures optional request-phase routing assessment and refinement.
type MetaRoutingConfig struct {
	Mode           string                 `yaml:"mode,omitempty"`
	MaxPasses      int                    `yaml:"max_passes,omitempty"`
	TriggerPolicy  *MetaTriggerPolicy     `yaml:"trigger_policy,omitempty"`
	AllowedActions []MetaRefinementAction `yaml:"allowed_actions,omitempty"`
}

// MetaTriggerPolicy defines the deterministic v1 conditions that can trigger refinement.
type MetaTriggerPolicy struct {
	DecisionMarginBelow      *float64                       `yaml:"decision_margin_below,omitempty"`
	ProjectionBoundaryWithin *float64                       `yaml:"projection_boundary_within,omitempty"`
	PartitionConflict        *bool                          `yaml:"partition_conflict,omitempty"`
	RequiredFamilies         []MetaRequiredSignalFamily     `yaml:"required_families,omitempty"`
	FamilyDisagreements      []MetaSignalFamilyDisagreement `yaml:"family_disagreements,omitempty"`
}

// MetaRequiredSignalFamily declares a signal family that must be present or confident enough.
type MetaRequiredSignalFamily struct {
	Type          string   `yaml:"type"`
	MinConfidence *float64 `yaml:"min_confidence,omitempty"`
	MinMatches    *int     `yaml:"min_matches,omitempty"`
}

// MetaSignalFamilyDisagreement declares a cheap-versus-expensive family pair to compare.
type MetaSignalFamilyDisagreement struct {
	Cheap     string `yaml:"cheap"`
	Expensive string `yaml:"expensive"`
}

// MetaRefinementAction defines one bounded refinement action allowed during reassessment.
type MetaRefinementAction struct {
	Type           string   `yaml:"type"`
	SignalFamilies []string `yaml:"signal_families,omitempty"`
}

func (cfg MetaRoutingConfig) Enabled() bool {
	return strings.TrimSpace(cfg.Mode) != ""
}

func SupportedMetaRoutingModes() []string {
	cloned := append([]string(nil), supportedMetaRoutingModes...)
	sort.Strings(cloned)
	return cloned
}

func IsSupportedMetaRoutingMode(mode string) bool {
	for _, candidate := range supportedMetaRoutingModes {
		if candidate == mode {
			return true
		}
	}
	return false
}

func SupportedMetaRoutingActionTypes() []string {
	cloned := append([]string(nil), supportedMetaRoutingActionTypes...)
	sort.Strings(cloned)
	return cloned
}

func IsSupportedMetaRoutingActionType(actionType string) bool {
	for _, candidate := range supportedMetaRoutingActionTypes {
		if candidate == actionType {
			return true
		}
	}
	return false
}

func copyMetaRoutingConfig(input MetaRoutingConfig) MetaRoutingConfig {
	output := MetaRoutingConfig{
		Mode:      input.Mode,
		MaxPasses: input.MaxPasses,
	}
	if input.TriggerPolicy != nil {
		output.TriggerPolicy = copyMetaTriggerPolicy(input.TriggerPolicy)
	}
	if len(input.AllowedActions) > 0 {
		output.AllowedActions = make([]MetaRefinementAction, 0, len(input.AllowedActions))
		for _, action := range input.AllowedActions {
			output.AllowedActions = append(output.AllowedActions, MetaRefinementAction{
				Type:           action.Type,
				SignalFamilies: cloneStrings(action.SignalFamilies),
			})
		}
	}
	return output
}

func copyMetaTriggerPolicy(input *MetaTriggerPolicy) *MetaTriggerPolicy {
	if input == nil {
		return nil
	}
	output := &MetaTriggerPolicy{
		DecisionMarginBelow:      cloneFloat64Ptr(input.DecisionMarginBelow),
		ProjectionBoundaryWithin: cloneFloat64Ptr(input.ProjectionBoundaryWithin),
		PartitionConflict:        cloneBoolPtr(input.PartitionConflict),
	}
	if len(input.RequiredFamilies) > 0 {
		output.RequiredFamilies = make([]MetaRequiredSignalFamily, 0, len(input.RequiredFamilies))
		for _, family := range input.RequiredFamilies {
			output.RequiredFamilies = append(output.RequiredFamilies, MetaRequiredSignalFamily{
				Type:          family.Type,
				MinConfidence: cloneFloat64Ptr(family.MinConfidence),
				MinMatches:    cloneIntPtr(family.MinMatches),
			})
		}
	}
	if len(input.FamilyDisagreements) > 0 {
		output.FamilyDisagreements = make([]MetaSignalFamilyDisagreement, len(input.FamilyDisagreements))
		copy(output.FamilyDisagreements, input.FamilyDisagreements)
	}
	return output
}

func cloneFloat64Ptr(input *float64) *float64 {
	if input == nil {
		return nil
	}
	value := *input
	return &value
}

func cloneIntPtr(input *int) *int {
	if input == nil {
		return nil
	}
	value := *input
	return &value
}

func cloneBoolPtr(input *bool) *bool {
	if input == nil {
		return nil
	}
	value := *input
	return &value
}
