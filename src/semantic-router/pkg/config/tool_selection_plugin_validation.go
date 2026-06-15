package config

import (
	"fmt"
	"strings"
)

func (c *ToolSelectionPluginConfig) Validate() error {
	if c == nil {
		return nil
	}
	if !c.Enabled {
		return nil
	}
	mode, err := normalizeToolSelectionMode(c.Mode)
	if err != nil {
		return err
	}
	if err := c.validateModeConstraints(mode); err != nil {
		return err
	}
	return c.validateAdvancedFiltering()
}

func normalizeToolSelectionMode(mode string) (string, error) {
	trimmed := strings.TrimSpace(mode)
	if trimmed == "" {
		return ToolSelectionModeAdd, nil
	}
	switch trimmed {
	case ToolSelectionModeAdd, ToolSelectionModeFilter:
		return trimmed, nil
	default:
		return "", fmt.Errorf("tool_selection plugin: mode must be %q or %q", ToolSelectionModeAdd, ToolSelectionModeFilter)
	}
}

func (c *ToolSelectionPluginConfig) validateModeConstraints(mode string) error {
	if mode == ToolSelectionModeAdd {
		if c.TopK < 0 {
			return fmt.Errorf("tool_selection plugin: top_k must be >= 0")
		}
		return nil
	}
	if c.PreserveCount < 0 {
		return fmt.Errorf("tool_selection plugin: preserve_count must be >= 0")
	}
	if c.RelevanceThreshold == nil {
		return nil
	}
	if *c.RelevanceThreshold < 0 || *c.RelevanceThreshold > 1 {
		return fmt.Errorf("tool_selection plugin: relevance_threshold must be between 0 and 1")
	}
	return nil
}

func (c *ToolSelectionPluginConfig) validateAdvancedFiltering() error {
	if c.AdvancedFiltering == nil || !c.AdvancedFiltering.Enabled {
		return nil
	}
	if err := validateAdvancedToolFilteringIntFields(c.AdvancedFiltering); err != nil {
		return fmt.Errorf("tool_selection plugin: advanced_filtering: %w", err)
	}
	if err := validateAdvancedToolFilteringCoreFloats(c.AdvancedFiltering); err != nil {
		return fmt.Errorf("tool_selection plugin: advanced_filtering: %w", err)
	}
	if err := validateToolFilteringWeightFloats(c.AdvancedFiltering.Weights); err != nil {
		return fmt.Errorf("tool_selection plugin: advanced_filtering: %w", err)
	}
	if err := validateRetrievalStrategyValue(c.AdvancedFiltering.RetrievalStrategy); err != nil {
		return fmt.Errorf("tool_selection plugin: advanced_filtering: %w", err)
	}
	if err := validateHybridHistorySubconfig(c.AdvancedFiltering.HybridHistory); err != nil {
		return fmt.Errorf("tool_selection plugin: advanced_filtering: %w", err)
	}
	return nil
}
