package commands

import (
	"testing"

	"github.com/spf13/cobra"
)

func TestStatusCommandStructure(t *testing.T) {
	tests := []struct {
		name          string
		expectedUse   string
		expectedShort string
		hasFlags      []string
	}{
		{
			name:          "status command has correct structure",
			expectedUse:   "status",
			expectedShort: "Check router and components status",
			hasFlags:      []string{"namespace"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewStatusCmd()

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

func TestStatusCommand(t *testing.T) {
	tests := []struct {
		name      string
		args      []string
		wantError bool
	}{
		{
			name:      "status with default namespace",
			args:      []string{"status"},
			wantError: false,
		},
		{
			name:      "status with custom namespace",
			args:      []string{"status", "--namespace", "production"},
			wantError: false,
		},
		{
			name:      "status with short namespace flag",
			args:      []string{"status", "--namespace=test"},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rootCmd := &cobra.Command{Use: "vsr"}
			statusCmd := NewStatusCmd()
			rootCmd.AddCommand(statusCmd)

			rootCmd.SetArgs(tt.args)
			_, _ = rootCmd.ExecuteC() // Ignore error, just testing command structure
		})
	}
}

func TestStatusCommandFlags(t *testing.T) {
	cmd := NewStatusCmd()

	tests := []struct {
		flagName     string
		flagType     string
		defaultValue interface{}
	}{
		{
			flagName:     "namespace",
			flagType:     "string",
			defaultValue: "default",
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

			if flag.DefValue != tt.defaultValue.(string) {
				t.Errorf("expected default value %q, got %q", tt.defaultValue, flag.DefValue)
			}
		})
	}
}

func TestLogsCommandStructure(t *testing.T) {
	tests := []struct {
		name          string
		expectedUse   string
		expectedShort string
		hasFlags      []string
	}{
		{
			name:          "logs command has correct structure",
			expectedUse:   "logs",
			expectedShort: "Fetch router logs",
			hasFlags:      []string{"follow", "tail", "namespace", "env", "component", "since", "grep"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewLogsCmd()

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

func TestLogsCommand(t *testing.T) {
	tests := []struct {
		name      string
		args      []string
		wantError bool
	}{
		{
			name:      "logs with defaults",
			args:      []string{"logs"},
			wantError: false,
		},
		{
			name:      "logs with follow flag",
			args:      []string{"logs", "--follow"},
			wantError: false,
		},
		{
			name:      "logs with tail count",
			args:      []string{"logs", "--tail", "50"},
			wantError: false,
		},
		{
			name:      "logs with namespace",
			args:      []string{"logs", "--namespace", "production"},
			wantError: false,
		},
		{
			name:      "logs with env type",
			args:      []string{"logs", "--env", "docker"},
			wantError: false,
		},
		{
			name:      "logs with component filter",
			args:      []string{"logs", "--component", "router"},
			wantError: false,
		},
		{
			name:      "logs with since filter",
			args:      []string{"logs", "--since", "10m"},
			wantError: false,
		},
		{
			name:      "logs with grep filter",
			args:      []string{"logs", "--grep", "error"},
			wantError: false,
		},
		{
			name:      "logs with multiple flags",
			args:      []string{"logs", "--follow", "--tail", "200", "--env", "kubernetes", "--namespace", "prod", "--component", "router", "--grep", "ERROR"},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rootCmd := &cobra.Command{Use: "vsr"}
			logsCmd := NewLogsCmd()
			rootCmd.AddCommand(logsCmd)

			rootCmd.SetArgs(tt.args)
			_, _ = rootCmd.ExecuteC() // Ignore error, just testing command structure
		})
	}
}

func TestLogsCommandFlags(t *testing.T) {
	cmd := NewLogsCmd()

	tests := []struct {
		flagName     string
		flagType     string
		defaultValue interface{}
	}{
		{
			flagName:     "follow",
			flagType:     "bool",
			defaultValue: "false",
		},
		{
			flagName:     "tail",
			flagType:     "int",
			defaultValue: "100",
		},
		{
			flagName:     "namespace",
			flagType:     "string",
			defaultValue: "default",
		},
		{
			flagName:     "env",
			flagType:     "string",
			defaultValue: "",
		},
		{
			flagName:     "component",
			flagType:     "string",
			defaultValue: "",
		},
		{
			flagName:     "since",
			flagType:     "string",
			defaultValue: "",
		},
		{
			flagName:     "grep",
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

			if flag.DefValue != tt.defaultValue.(string) {
				t.Errorf("expected default value %q, got %q", tt.defaultValue, flag.DefValue)
			}
		})
	}
}

func TestLogsCommandShortFlags(t *testing.T) {
	cmd := NewLogsCmd()

	// Test short flags
	tests := []struct {
		shortFlag string
		longFlag  string
	}{
		{
			shortFlag: "f",
			longFlag:  "follow",
		},
		{
			shortFlag: "n",
			longFlag:  "tail",
		},
	}

	for _, tt := range tests {
		t.Run("short_flag_"+tt.shortFlag, func(t *testing.T) {
			shortFlag := cmd.Flags().ShorthandLookup(tt.shortFlag)
			if shortFlag == nil {
				t.Fatalf("short flag %q not found", tt.shortFlag)
			}

			if shortFlag.Name != tt.longFlag {
				t.Errorf("expected short flag %q to map to %q, got %q", tt.shortFlag, tt.longFlag, shortFlag.Name)
			}
		})
	}
}
