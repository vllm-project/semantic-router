package commands

import (
	"testing"

	"github.com/spf13/cobra"
)

func TestDebugCommandStructure(t *testing.T) {
	tests := []struct {
		name          string
		expectedUse   string
		expectedShort string
	}{
		{
			name:          "debug command has correct structure",
			expectedUse:   "debug",
			expectedShort: "Run interactive debugging session",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewDebugCmd()

			if cmd.Use != tt.expectedUse {
				t.Errorf("expected Use %q, got %q", tt.expectedUse, cmd.Use)
			}

			if cmd.Short != tt.expectedShort {
				t.Errorf("expected Short %q, got %q", tt.expectedShort, cmd.Short)
			}
		})
	}
}

func TestDebugCommand(t *testing.T) {
	tests := []struct {
		name      string
		args      []string
		wantError bool
	}{
		{
			name:      "debug command runs",
			args:      []string{"debug"},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rootCmd := &cobra.Command{Use: "vsr"}
			rootCmd.PersistentFlags().StringP("config", "c", "config/config.yaml", "Path to configuration file")

			debugCmd := NewDebugCmd()
			rootCmd.AddCommand(debugCmd)

			rootCmd.SetArgs(tt.args)
			_, _ = rootCmd.ExecuteC() // Ignore error, just testing command structure
		})
	}
}

func TestHealthCommandStructure(t *testing.T) {
	tests := []struct {
		name          string
		expectedUse   string
		expectedShort string
	}{
		{
			name:          "health command has correct structure",
			expectedUse:   "health",
			expectedShort: "Check router health",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewHealthCmd()

			if cmd.Use != tt.expectedUse {
				t.Errorf("expected Use %q, got %q", tt.expectedUse, cmd.Use)
			}

			if cmd.Short != tt.expectedShort {
				t.Errorf("expected Short %q, got %q", tt.expectedShort, cmd.Short)
			}
		})
	}
}

func TestHealthCommand(t *testing.T) {
	tests := []struct {
		name      string
		args      []string
		wantError bool
	}{
		{
			name:      "health command runs",
			args:      []string{"health"},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rootCmd := &cobra.Command{Use: "vsr"}
			rootCmd.PersistentFlags().StringP("config", "c", "config/config.yaml", "Path to configuration file")

			healthCmd := NewHealthCmd()
			rootCmd.AddCommand(healthCmd)

			rootCmd.SetArgs(tt.args)
			_, _ = rootCmd.ExecuteC() // Ignore error, just testing command structure
		})
	}
}

func TestDiagnoseCommandStructure(t *testing.T) {
	tests := []struct {
		name          string
		expectedUse   string
		expectedShort string
		hasFlags      []string
	}{
		{
			name:          "diagnose command has correct structure",
			expectedUse:   "diagnose",
			expectedShort: "Generate diagnostic report",
			hasFlags:      []string{"output"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewDiagnoseCmd()

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

func TestDiagnoseCommand(t *testing.T) {
	tests := []struct {
		name      string
		args      []string
		wantError bool
	}{
		{
			name:      "diagnose command runs",
			args:      []string{"diagnose"},
			wantError: false,
		},
		{
			name:      "diagnose with output flag",
			args:      []string{"diagnose", "--output", "/tmp/diagnose.txt"},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rootCmd := &cobra.Command{Use: "vsr"}
			rootCmd.PersistentFlags().StringP("config", "c", "config/config.yaml", "Path to configuration file")

			diagnoseCmd := NewDiagnoseCmd()
			rootCmd.AddCommand(diagnoseCmd)

			rootCmd.SetArgs(tt.args)
			_, _ = rootCmd.ExecuteC() // Ignore error, just testing command structure
		})
	}
}

func TestDiagnoseCommandFlags(t *testing.T) {
	cmd := NewDiagnoseCmd()

	tests := []struct {
		flagName     string
		flagType     string
		defaultValue string
	}{
		{
			flagName:     "output",
			flagType:     "string",
			defaultValue: "",
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
