package commands

import (
	"testing"
)

func TestDashboardCommandStructure(t *testing.T) {
	tests := []struct {
		name          string
		expectedUse   string
		expectedShort string
		hasFlags      []string
	}{
		{
			name:          "dashboard command has correct structure",
			expectedUse:   "dashboard",
			expectedShort: "Open router dashboard in browser",
			hasFlags:      []string{"namespace", "no-open"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewDashboardCmd()

			if cmd.Use != tt.expectedUse {
				t.Errorf("expected Use %q, got %q", tt.expectedUse, cmd.Use)
			}

			if cmd.Short != tt.expectedShort {
				t.Errorf("expected Short %q, got %q", tt.expectedShort, cmd.Short)
			}

			// Verify flags exist
			for _, flagName := range tt.hasFlags {
				if cmd.Flags().Lookup(flagName) == nil {
					t.Errorf("expected flag %q not found", flagName)
				}
			}
		})
	}
}

func TestDashboardCommandFlags(t *testing.T) {
	cmd := NewDashboardCmd()

	tests := []struct {
		flagName     string
		flagType     string
		defaultValue string
	}{
		{
			flagName:     "namespace",
			flagType:     "string",
			defaultValue: "default",
		},
		{
			flagName:     "no-open",
			flagType:     "bool",
			defaultValue: "false",
		},
	}

	for _, tt := range tests {
		t.Run("flag_"+tt.flagName, func(t *testing.T) {
			flag := cmd.Flags().Lookup(tt.flagName)
			if flag == nil {
				t.Fatalf("flag %q not found", tt.flagName)
			}

			if flag.Value.Type() != tt.flagType {
				t.Errorf("expected flag type %q, got %q", tt.flagType, flag.Value.Type())
			}

			if flag.DefValue != tt.defaultValue {
				t.Errorf("expected default value %q, got %q", tt.defaultValue, flag.DefValue)
			}
		})
	}
}

func TestMetricsCommandStructure(t *testing.T) {
	tests := []struct {
		name          string
		expectedUse   string
		expectedShort string
		hasFlags      []string
	}{
		{
			name:          "metrics command has correct structure",
			expectedUse:   "metrics",
			expectedShort: "Display router metrics",
			hasFlags:      []string{"since", "watch"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewMetricsCmd()

			if cmd.Use != tt.expectedUse {
				t.Errorf("expected Use %q, got %q", tt.expectedUse, cmd.Use)
			}

			if cmd.Short != tt.expectedShort {
				t.Errorf("expected Short %q, got %q", tt.expectedShort, cmd.Short)
			}

			// Verify flags exist
			for _, flagName := range tt.hasFlags {
				if cmd.Flags().Lookup(flagName) == nil {
					t.Errorf("expected flag %q not found", flagName)
				}
			}
		})
	}
}

func TestMetricsCommandFlags(t *testing.T) {
	cmd := NewMetricsCmd()

	tests := []struct {
		flagName     string
		flagType     string
		defaultValue string
	}{
		{
			flagName:     "since",
			flagType:     "string",
			defaultValue: "5m",
		},
		{
			flagName:     "watch",
			flagType:     "bool",
			defaultValue: "false",
		},
	}

	for _, tt := range tests {
		t.Run("flag_"+tt.flagName, func(t *testing.T) {
			flag := cmd.Flags().Lookup(tt.flagName)
			if flag == nil {
				t.Fatalf("flag %q not found", tt.flagName)
			}

			if flag.Value.Type() != tt.flagType {
				t.Errorf("expected flag type %q, got %q", tt.flagType, flag.Value.Type())
			}

			if flag.DefValue != tt.defaultValue {
				t.Errorf("expected default value %q, got %q", tt.defaultValue, flag.DefValue)
			}
		})
	}
}

func TestDetectActiveDeploymentExists(t *testing.T) {
	// Test that the function exists and can be called
	result := detectActiveDeployment("default")
	// Result can be empty string or a deployment type
	if result != "" && result != "local" && result != "docker" && result != "kubernetes" && result != "helm" {
		t.Errorf("unexpected deployment type: %s", result)
	}
}

func TestOpenBrowserFunction(t *testing.T) {
	// Test that the function exists
	// We can't actually test browser opening, just that the function is callable
	err := openBrowser("http://example.com")
	// Error is expected as we likely don't have display, but function should exist
	_ = err // Just testing function exists
}
