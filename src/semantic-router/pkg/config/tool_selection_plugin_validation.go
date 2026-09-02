package config

import (
	"fmt"
	"strings"
)

func (c *ToolSelectionPluginConfig) Validate() error {
	if c == nil {
		return nil
	}
	// Sticky config under a disabled plugin is rejected outright rather than
	// silently accepted as dead configuration, even before the early return
	// below for the (much more common) disabled-and-no-sticky-block case.
	if c.Sticky != nil && c.Sticky.Enabled && !c.Enabled {
		return fmt.Errorf("tool_selection plugin: sticky.enabled requires tool_selection to be enabled")
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
	if err := c.validateAdvancedFiltering(); err != nil {
		return err
	}
	return c.validateSticky()
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

// validateSticky enforces sticky's bounds. It assumes the caller already
// rejected sticky.enabled under a disabled plugin (Validate does, above);
// this only runs when c.Enabled is true.
func (c *ToolSelectionPluginConfig) validateSticky() error {
	if c.Sticky == nil || !c.Sticky.Enabled {
		return nil
	}
	if c.Sticky.MaxTools < 0 || c.Sticky.MaxTools > StickyToolSelectionMaxToolsUpperBound {
		return fmt.Errorf(
			"tool_selection plugin: sticky.max_tools must be between 1 and %d",
			StickyToolSelectionMaxToolsUpperBound,
		)
	}
	if c.Sticky.MaxNewToolsPerTurn == nil {
		return nil
	}
	maxTools := c.Sticky.EffectiveMaxTools()
	if v := *c.Sticky.MaxNewToolsPerTurn; v < 0 || v > maxTools {
		return fmt.Errorf(
			"tool_selection plugin: sticky.max_new_tools_per_turn must be between 0 and max_tools (%d)",
			maxTools,
		)
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
