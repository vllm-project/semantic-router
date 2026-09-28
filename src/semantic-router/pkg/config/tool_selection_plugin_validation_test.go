package config

import "testing"

func float32Ptr(v float32) *float32 { return &v }
func stickyIntPtr(v int) *int       { return &v }

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

func TestToolSelectionPluginValidate_StickyEnabled_OK(t *testing.T) {
	c := ToolSelectionPluginConfig{
		Enabled: true,
		Mode:    ToolSelectionModeAdd,
		Sticky:  &StickyToolSelectionConfig{Enabled: true},
	}

	if err := c.Validate(); err != nil {
		t.Fatalf("sticky.enabled should be accepted by config validation: %v", err)
	}
}

func TestToolSelectionPluginValidate_StickyMaxToolsOutOfRange_Err(t *testing.T) {
	for _, maxTools := range []int{-1, 0, 129} {
		c := ToolSelectionPluginConfig{
			Enabled: true,
			Mode:    ToolSelectionModeAdd,
			Sticky:  &StickyToolSelectionConfig{Enabled: true, MaxTools: stickyIntPtr(maxTools)},
		}
		if err := c.Validate(); err == nil {
			t.Fatalf("max_tools=%d: expected error", maxTools)
		}
	}
}

func TestToolSelectionPluginValidate_StickyMaxToolsExplicitZero_Err(t *testing.T) {
	// max_tools: 0 is explicitly out of the valid 1..128 range and must be
	// rejected, not silently treated as "unset -> default 16". This is the
	// same class of unset-vs-explicit-zero ambiguity MaxNewToolsPerTurn
	// already guards against, just for the field whose valid range excludes
	// zero entirely rather than including it.
	c := ToolSelectionPluginConfig{
		Enabled: true,
		Mode:    ToolSelectionModeAdd,
		Sticky:  &StickyToolSelectionConfig{Enabled: true, MaxTools: stickyIntPtr(0)},
	}
	err := c.Validate()
	if err == nil {
		t.Fatal("expected error: sticky.max_tools: 0 is out of the valid 1..128 range")
	}
	const want = "tool_selection plugin: sticky.max_tools must be between 1 and 128"
	if err.Error() != want {
		t.Fatalf("error = %q, want %q", err.Error(), want)
	}
}

// TestToolSelectionPluginValidate_StickyMaxNewToolsPerTurnZero_PreservesExplicitValue
// covers the unset-vs-explicit-zero distinction: EffectiveMaxNewToolsPerTurn
// must not silently default an explicit 0.
func TestToolSelectionPluginValidate_StickyMaxNewToolsPerTurnZero_PreservesExplicitValue(t *testing.T) {
	c := ToolSelectionPluginConfig{
		Enabled: true,
		Mode:    ToolSelectionModeAdd,
		Sticky:  &StickyToolSelectionConfig{Enabled: true, MaxNewToolsPerTurn: stickyIntPtr(0)},
	}

	if err := c.Validate(); err != nil {
		t.Fatalf("sticky.enabled with explicit zero should validate: %v", err)
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
			MaxTools:           stickyIntPtr(4),
			MaxNewToolsPerTurn: stickyIntPtr(5),
		},
	}
	if err := c.Validate(); err == nil {
		t.Fatal("expected error: max_new_tools_per_turn must not exceed max_tools")
	}
}

func TestToolSelectionPluginValidate_StickyPinCalledToolsExplicitFalse_OK(t *testing.T) {
	// Enabled: false here — this test is about EffectivePinCalledTools()'s
	// explicit-false handling.
	c := ToolSelectionPluginConfig{
		Enabled: true,
		Mode:    ToolSelectionModeAdd,
		Sticky:  &StickyToolSelectionConfig{Enabled: false, PinCalledTools: boolPtr(false)},
	}
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	if c.Sticky.EffectivePinCalledTools() {
		t.Fatal("effective pin_called_tools should honor explicit false")
	}
}

// TestToolSelectionPluginValidate_StickyDisabledWithValidBounds_OK and
// TestToolSelectionPluginValidate_StickyDisabledInvalidBounds_Err replace
// the old StickyDisabledIgnoresBounds test: bounds are now validated
// whenever a sticky block is present, even when sticky.enabled is false —
// a disabled-but-malformed block must not be able to slip through
// validation and start failing only once switched on.
func TestToolSelectionPluginValidate_StickyDisabledWithValidBounds_OK(t *testing.T) {
	c := ToolSelectionPluginConfig{
		Enabled: true,
		Mode:    ToolSelectionModeAdd,
		Sticky: &StickyToolSelectionConfig{
			Enabled:            false,
			MaxTools:           stickyIntPtr(16),
			MaxNewToolsPerTurn: stickyIntPtr(2),
			PinCalledTools:     boolPtr(true),
		},
	}
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestToolSelectionPluginValidate_StickyDisabledInvalidBounds_Err(t *testing.T) {
	c := ToolSelectionPluginConfig{
		Enabled: true,
		Mode:    ToolSelectionModeAdd,
		Sticky:  &StickyToolSelectionConfig{Enabled: false, MaxTools: stickyIntPtr(999)},
	}
	if err := c.Validate(); err == nil {
		t.Fatal("expected error: sticky.max_tools should be validated when the sticky block is present")
	}
}

func TestToolSelectionPluginConfigContracts_StickyEnabled_OK(t *testing.T) {
	payload := MustStructuredPayload(&ToolSelectionPluginConfig{
		Enabled: true,
		Mode:    ToolSelectionModeAdd,
		Sticky:  &StickyToolSelectionConfig{Enabled: true},
	})
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Decisions: []Decision{
				{
					Name: "sticky-decision",
					Plugins: []DecisionPlugin{
						{
							Type:          DecisionPluginToolSelection,
							Configuration: payload,
						},
					},
				},
			},
		},
	}

	if err := validateConfigContracts(cfg); err != nil {
		t.Fatalf("sticky-enabled decision should pass config admission: %v", err)
	}
}
