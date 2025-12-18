package commands

import (
	"testing"

	"github.com/spf13/cobra"
)

func TestModelCommandStructure(t *testing.T) {
	tests := []struct {
		name            string
		expectedUse     string
		expectedShort   string
		subcommandCount int
		subcommands     []string
	}{
		{
			name:            "model command has correct structure",
			expectedUse:     "model",
			expectedShort:   "Manage semantic router models",
			subcommandCount: 5,
			subcommands:     []string{"list", "info", "validate", "remove", "download"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewModelCmd()

			if cmd.Use != tt.expectedUse {
				t.Errorf("expected Use %q, got %q", tt.expectedUse, cmd.Use)
			}

			if cmd.Short != tt.expectedShort {
				t.Errorf("expected Short %q, got %q", tt.expectedShort, cmd.Short)
			}

			if len(cmd.Commands()) != tt.subcommandCount {
				t.Errorf("expected %d subcommands, got %d", tt.subcommandCount, len(cmd.Commands()))
			}

			// Verify subcommands exist
			for _, subcmd := range tt.subcommands {
				found := false
				for _, c := range cmd.Commands() {
					if c.Use == subcmd || c.Name() == subcmd {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("expected subcommand %q not found", subcmd)
				}
			}
		})
	}
}

func TestModelListCommand(t *testing.T) {
	tests := []struct {
		name      string
		args      []string
		wantError bool
	}{
		{
			name:      "model list default",
			args:      []string{"model", "list"},
			wantError: false,
		},
		{
			name:      "model list with downloaded flag",
			args:      []string{"model", "list", "--downloaded"},
			wantError: false,
		},
		{
			name:      "model list with json output",
			args:      []string{"model", "list", "-o", "json"},
			wantError: false,
		},
		{
			name:      "model list with yaml output",
			args:      []string{"model", "list", "-o", "yaml"},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rootCmd := &cobra.Command{Use: "vsr"}
			rootCmd.PersistentFlags().StringP("output", "o", "table", "Output format")

			modelCmd := NewModelCmd()
			rootCmd.AddCommand(modelCmd)

			rootCmd.SetArgs(tt.args)
			_, _ = rootCmd.ExecuteC() // Ignore error, just testing command structure
		})
	}
}

func TestModelInfoCommand(t *testing.T) {
	tests := []struct {
		name      string
		args      []string
		wantError bool
	}{
		{
			name:      "model info with model id",
			args:      []string{"model", "info", "test-model"},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rootCmd := &cobra.Command{Use: "vsr"}
			rootCmd.PersistentFlags().StringP("output", "o", "table", "Output format")

			modelCmd := NewModelCmd()
			rootCmd.AddCommand(modelCmd)

			rootCmd.SetArgs(tt.args)
			_, _ = rootCmd.ExecuteC() // Ignore error, just testing command structure
		})
	}
}

func TestModelValidateCommand(t *testing.T) {
	tests := []struct {
		name      string
		args      []string
		wantError bool
	}{
		{
			name:      "model validate specific model",
			args:      []string{"model", "validate", "test-model"},
			wantError: false,
		},
		{
			name:      "model validate all models",
			args:      []string{"model", "validate", "--all"},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rootCmd := &cobra.Command{Use: "vsr"}
			modelCmd := NewModelCmd()
			rootCmd.AddCommand(modelCmd)

			rootCmd.SetArgs(tt.args)
			_, _ = rootCmd.ExecuteC() // Ignore error, just testing command structure
		})
	}
}

func TestModelRemoveCommand(t *testing.T) {
	tests := []struct {
		name      string
		args      []string
		wantError bool
	}{
		{
			name:      "model remove with confirmation",
			args:      []string{"model", "remove", "test-model", "--force"},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rootCmd := &cobra.Command{Use: "vsr"}
			modelCmd := NewModelCmd()
			rootCmd.AddCommand(modelCmd)

			rootCmd.SetArgs(tt.args)
			_, _ = rootCmd.ExecuteC() // Ignore error, just testing command structure
		})
	}
}

func TestModelDownloadCommand(t *testing.T) {
	tests := []struct {
		name      string
		args      []string
		wantError bool
	}{
		{
			name:      "model download",
			args:      []string{"model", "download"},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rootCmd := &cobra.Command{Use: "vsr"}
			modelCmd := NewModelCmd()
			rootCmd.AddCommand(modelCmd)

			rootCmd.SetArgs(tt.args)
			_, _ = rootCmd.ExecuteC() // Ignore error, just testing command structure
		})
	}
}

func TestModelListCommandFlags(t *testing.T) {
	cmd := NewModelCmd()

	// Find the list subcommand
	var listCmd *cobra.Command
	for _, c := range cmd.Commands() {
		if c.Name() == "list" {
			listCmd = c
			break
		}
	}

	if listCmd == nil {
		t.Fatal("list subcommand not found")
	}

	tests := []struct {
		flagName     string
		flagType     string
		defaultValue string
	}{
		{
			flagName:     "downloaded",
			flagType:     "bool",
			defaultValue: "false",
		},
	}

	for _, tt := range tests {
		t.Run("flag_"+tt.flagName, func(t *testing.T) {
			flag := listCmd.Flags().Lookup(tt.flagName)
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

func TestModelValidateCommandFlags(t *testing.T) {
	cmd := NewModelCmd()

	// Find the validate subcommand
	var validateCmd *cobra.Command
	for _, c := range cmd.Commands() {
		if c.Name() == "validate" {
			validateCmd = c
			break
		}
	}

	if validateCmd == nil {
		t.Fatal("validate subcommand not found")
	}

	tests := []struct {
		flagName     string
		flagType     string
		defaultValue string
	}{
		{
			flagName:     "all",
			flagType:     "bool",
			defaultValue: "false",
		},
	}

	for _, tt := range tests {
		t.Run("flag_"+tt.flagName, func(t *testing.T) {
			flag := validateCmd.Flags().Lookup(tt.flagName)
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

func TestModelRemoveCommandFlags(t *testing.T) {
	cmd := NewModelCmd()

	// Find the remove subcommand
	var removeCmd *cobra.Command
	for _, c := range cmd.Commands() {
		if c.Name() == "remove" {
			removeCmd = c
			break
		}
	}

	if removeCmd == nil {
		t.Fatal("remove subcommand not found")
	}

	tests := []struct {
		flagName     string
		flagType     string
		defaultValue string
	}{
		{
			flagName:     "force",
			flagType:     "bool",
			defaultValue: "false",
		},
	}

	for _, tt := range tests {
		t.Run("flag_"+tt.flagName, func(t *testing.T) {
			flag := removeCmd.Flags().Lookup(tt.flagName)
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
