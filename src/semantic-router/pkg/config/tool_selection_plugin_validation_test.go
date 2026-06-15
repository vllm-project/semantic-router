package config

import (
	"testing"
)

func float32Ptr(v float32) *float32 { return &v }

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
