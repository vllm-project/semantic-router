/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1alpha1

import (
	"encoding/json"
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
)

// FilterConfigHelper provides helper methods for working with filter configurations
type FilterConfigHelper struct{}

// NewFilterConfigHelper creates a new FilterConfigHelper
func NewFilterConfigHelper() *FilterConfigHelper {
	return &FilterConfigHelper{}
}

// MarshalFilterConfig marshals a filter configuration to RawExtension
func (h *FilterConfigHelper) MarshalFilterConfig(config interface{}) (*runtime.RawExtension, error) {
	if config == nil {
		return nil, nil
	}

	data, err := json.Marshal(config)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal filter config: %w", err)
	}

	return &runtime.RawExtension{Raw: data}, nil
}

// UnmarshalPIIDetectionConfig unmarshals a PIIDetectionConfig from RawExtension
func (h *FilterConfigHelper) UnmarshalPIIDetectionConfig(raw *runtime.RawExtension) (*PIIDetectionConfig, error) {
	if raw == nil || len(raw.Raw) == 0 {
		return &PIIDetectionConfig{}, nil
	}

	var config PIIDetectionConfig
	if err := json.Unmarshal(raw.Raw, &config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal PIIDetectionConfig: %w", err)
	}

	return &config, nil
}

// UnmarshalPromptGuardConfig unmarshals a PromptGuardConfig from RawExtension
func (h *FilterConfigHelper) UnmarshalPromptGuardConfig(raw *runtime.RawExtension) (*PromptGuardConfig, error) {
	if raw == nil || len(raw.Raw) == 0 {
		return &PromptGuardConfig{}, nil
	}

	var config PromptGuardConfig
	if err := json.Unmarshal(raw.Raw, &config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal PromptGuardConfig: %w", err)
	}

	return &config, nil
}

// UnmarshalSemanticCacheConfig unmarshals a SemanticCacheConfig from RawExtension
func (h *FilterConfigHelper) UnmarshalSemanticCacheConfig(raw *runtime.RawExtension) (*SemanticCacheConfig, error) {
	if raw == nil || len(raw.Raw) == 0 {
		return &SemanticCacheConfig{}, nil
	}

	var config SemanticCacheConfig
	if err := json.Unmarshal(raw.Raw, &config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal SemanticCacheConfig: %w", err)
	}

	return &config, nil
}

// UnmarshalReasoningControlConfig unmarshals a ReasoningControlConfig from RawExtension
func (h *FilterConfigHelper) UnmarshalReasoningControlConfig(raw *runtime.RawExtension) (*ReasoningControlConfig, error) {
	if raw == nil || len(raw.Raw) == 0 {
		return &ReasoningControlConfig{}, nil
	}

	var config ReasoningControlConfig
	if err := json.Unmarshal(raw.Raw, &config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal ReasoningControlConfig: %w", err)
	}

	return &config, nil
}

// MarshalToolSelectionConfig marshals a ToolSelectionConfig to RawExtension
func (h *FilterConfigHelper) MarshalToolSelectionConfig(config *ToolSelectionConfig) (*runtime.RawExtension, error) {
	if config == nil {
		return &runtime.RawExtension{}, nil
	}

	data, err := json.Marshal(config)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal ToolSelectionConfig: %w", err)
	}

	return &runtime.RawExtension{Raw: data}, nil
}

// UnmarshalToolSelectionConfig unmarshals a ToolSelectionConfig from RawExtension
func (h *FilterConfigHelper) UnmarshalToolSelectionConfig(raw *runtime.RawExtension) (*ToolSelectionConfig, error) {
	if raw == nil || len(raw.Raw) == 0 {
		return &ToolSelectionConfig{}, nil
	}

	var config ToolSelectionConfig
	if err := json.Unmarshal(raw.Raw, &config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal ToolSelectionConfig: %w", err)
	}

	return &config, nil
}

// UnmarshalFilterConfig unmarshals a filter configuration based on the filter type
func (h *FilterConfigHelper) UnmarshalFilterConfig(filterType FilterType, raw *runtime.RawExtension) (interface{}, error) {
	switch filterType {
	case FilterTypePIIDetection:
		return h.UnmarshalPIIDetectionConfig(raw)
	case FilterTypePromptGuard:
		return h.UnmarshalPromptGuardConfig(raw)
	case FilterTypeSemanticCache:
		return h.UnmarshalSemanticCacheConfig(raw)
	case FilterTypeReasoningControl:
		return h.UnmarshalReasoningControlConfig(raw)
	case FilterTypeToolSelection:
		return h.UnmarshalToolSelectionConfig(raw)
	default:
		return nil, fmt.Errorf("unsupported filter type: %s", filterType)
	}
}

// ValidateFilterConfig validates a filter configuration
func (h *FilterConfigHelper) ValidateFilterConfig(filter *Filter) error {
	if filter == nil {
		return fmt.Errorf("filter cannot be nil")
	}

	// Validate filter type
	switch filter.Type {
	case FilterTypePIIDetection, FilterTypePromptGuard, FilterTypeSemanticCache, FilterTypeReasoningControl, FilterTypeToolSelection:
		// Valid filter types
	default:
		return fmt.Errorf("invalid filter type: %s", filter.Type)
	}

	// If config is provided, try to unmarshal it to validate structure
	if filter.Config != nil {
		_, err := h.UnmarshalFilterConfig(filter.Type, filter.Config)
		if err != nil {
			return fmt.Errorf("invalid filter config for type %s: %w", filter.Type, err)
		}
	}

	return nil
}

// CreatePIIDetectionFilter creates a PIIDetection filter with the given configuration
func CreatePIIDetectionFilter(config *PIIDetectionConfig) (*Filter, error) {
	helper := NewFilterConfigHelper()
	rawConfig, err := helper.MarshalFilterConfig(config)
	if err != nil {
		return nil, err
	}

	enabled := true
	return &Filter{
		Type:    FilterTypePIIDetection,
		Config:  rawConfig,
		Enabled: &enabled,
	}, nil
}

// CreatePromptGuardFilter creates a PromptGuard filter with the given configuration
func CreatePromptGuardFilter(config *PromptGuardConfig) (*Filter, error) {
	helper := NewFilterConfigHelper()
	rawConfig, err := helper.MarshalFilterConfig(config)
	if err != nil {
		return nil, err
	}

	enabled := true
	return &Filter{
		Type:    FilterTypePromptGuard,
		Config:  rawConfig,
		Enabled: &enabled,
	}, nil
}

// CreateSemanticCacheFilter creates a SemanticCache filter with the given configuration
func CreateSemanticCacheFilter(config *SemanticCacheConfig) (*Filter, error) {
	helper := NewFilterConfigHelper()
	rawConfig, err := helper.MarshalFilterConfig(config)
	if err != nil {
		return nil, err
	}

	enabled := true
	return &Filter{
		Type:    FilterTypeSemanticCache,
		Config:  rawConfig,
		Enabled: &enabled,
	}, nil
}

// CreateReasoningControlFilter creates a ReasoningControl filter with the given configuration
func CreateReasoningControlFilter(config *ReasoningControlConfig) (*Filter, error) {
	helper := NewFilterConfigHelper()
	rawConfig, err := helper.MarshalFilterConfig(config)
	if err != nil {
		return nil, err
	}

	enabled := true
	return &Filter{
		Type:    FilterTypeReasoningControl,
		Config:  rawConfig,
		Enabled: &enabled,
	}, nil
}

// CreateToolSelectionFilter creates a ToolSelection filter with the given configuration
func CreateToolSelectionFilter(config *ToolSelectionConfig) (*Filter, error) {
	helper := NewFilterConfigHelper()
	rawConfig, err := helper.MarshalFilterConfig(config)
	if err != nil {
		return nil, err
	}

	enabled := true
	return &Filter{
		Type:    FilterTypeToolSelection,
		Config:  rawConfig,
		Enabled: &enabled,
	}, nil
}
