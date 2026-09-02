package config

import (
	"testing"
)

func float32Ptr(v float32) *float32 { return &v }
func intPtr(v int) *int             { return &v }

func TestToolSelectionPluginValidate_FilterModeNilThresholdOK(t *testing.T) {
	c := ToolSelectionPluginConfig{
		Enabled:            true,
		Mode:               ToolSelectionModeFilter,
		PreserveCount:      2,
		RelevanceThreshold: nil,
	}
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestToolSelectionPluginValidate_FilterModeExplicitThreshold_OK(t *testing.T) {
	c := ToolSelectionPluginConfig{
		Enabled:            true,
		Mode:               ToolSelectionModeFilter,
		PreserveCount:      2,
		RelevanceThreshold: float32Ptr(0.42),
	}
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestToolSelectionPluginValidate_ModeInvalid_Err(t *testing.T) {
	c := ToolSelectionPluginConfig{Enabled: true, Mode: "bogus"}
	if err := c.Validate(); err == nil {
		t.Fatal("expected error")
	}
}

func TestToolSelectionPluginValidate_StickyOmitted_OK(t *testing.T) {
	c := ToolSelectionPluginConfig{Enabled: true, Mode: ToolSelectionModeAdd}
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	if c.Sticky.EffectiveMaxTools() != StickyToolSelectionDefaultMaxTools {
		t.Fatalf("effective max_tools = %d", c.Sticky.EffectiveMaxTools())
	}
	if c.Sticky.EffectiveMaxNewToolsPerTurn() != StickyToolSelectionDefaultMaxNewToolsPerTurn {
		t.Fatalf("effective max_new_tools_per_turn = %d", c.Sticky.EffectiveMaxNewToolsPerTurn())
	}
	if !c.Sticky.EffectivePinCalledTools() {
		t.Fatal("effective pin_called_tools should default true")
	}
}

func TestToolSelectionPluginValidate_StickyEnabledUnderDisabledPlugin_Err(t *testing.T) {
	c := ToolSelectionPluginConfig{
		Enabled: false,
		Sticky:  &StickyToolSelectionConfig{Enabled: true},
	}
	if err := c.Validate(); err == nil {
		t.Fatal("expected error: sticky.enabled requires tool_selection.enabled")
	}
}

func TestToolSelectionPluginValidate_StickyEnabledWithDefaults_OK(t *testing.T) {
	c := ToolSelectionPluginConfig{
		Enabled: true,
		Mode:    ToolSelectionModeAdd,
		Sticky:  &StickyToolSelectionConfig{Enabled: true},
	}
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestToolSelectionPluginValidate_StickyMaxToolsOutOfRange_Err(t *testing.T) {
	for _, maxTools := range []int{-1, 129} {
		c := ToolSelectionPluginConfig{
			Enabled: true,
			Mode:    ToolSelectionModeAdd,
			Sticky:  &StickyToolSelectionConfig{Enabled: true, MaxTools: maxTools},
		}
		if err := c.Validate(); err == nil {
			t.Fatalf("max_tools=%d: expected error", maxTools)
		}
	}
}

func TestToolSelectionPluginValidate_StickyMaxNewToolsPerTurnZero_OK(t *testing.T) {
	// 0 is a meaningful explicit value (reuse/pin only, no relevance growth),
	// not "unset" — must not be rejected or silently replaced by the default.
	c := ToolSelectionPluginConfig{
		Enabled: true,
		Mode:    ToolSelectionModeAdd,
		Sticky:  &StickyToolSelectionConfig{Enabled: true, MaxNewToolsPerTurn: intPtr(0)},
	}
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	if got := c.Sticky.EffectiveMaxNewToolsPerTurn(); got != 0 {
		t.Fatalf("effective max_new_tools_per_turn = %d, want 0 (explicit, not defaulted)", got)
	}
}

func TestToolSelectionPluginValidate_StickyMaxNewToolsPerTurnExceedsMaxTools_Err(t *testing.T) {
	c := ToolSelectionPluginConfig{
		Enabled: true,
		Mode:    ToolSelectionModeAdd,
		Sticky: &StickyToolSelectionConfig{
			Enabled:            true,
			MaxTools:           4,
			MaxNewToolsPerTurn: intPtr(5),
		},
	}
	if err := c.Validate(); err == nil {
		t.Fatal("expected error: max_new_tools_per_turn must not exceed max_tools")
	}
}

func TestToolSelectionPluginValidate_StickyPinCalledToolsExplicitFalse_OK(t *testing.T) {
	c := ToolSelectionPluginConfig{
		Enabled: true,
		Mode:    ToolSelectionModeAdd,
		Sticky:  &StickyToolSelectionConfig{Enabled: true, PinCalledTools: boolPtr(false)},
	}
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	if c.Sticky.EffectivePinCalledTools() {
		t.Fatal("effective pin_called_tools should honor explicit false")
	}
}

func TestToolSelectionPluginValidate_StickyDisabledIgnoresBounds(t *testing.T) {
	// sticky.enabled: false must not trigger bound validation even with an
	// otherwise-invalid max_tools — the block is simply inert.
	c := ToolSelectionPluginConfig{
		Enabled: true,
		Mode:    ToolSelectionModeAdd,
		Sticky:  &StickyToolSelectionConfig{Enabled: false, MaxTools: 999},
	}
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
}
