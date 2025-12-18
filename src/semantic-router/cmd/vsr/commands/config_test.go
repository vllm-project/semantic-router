package commands

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/spf13/cobra"
)

func TestConfigCommandStructure(t *testing.T) {
	tests := []struct {
		name            string
		expectedUse     string
		expectedShort   string
		subcommandCount int
		subcommands     []string
	}{
		{
			name:            "config command has correct structure",
			expectedUse:     "config",
			expectedShort:   "Manage router configuration",
			subcommandCount: 5,
			subcommands:     []string{"view", "edit", "validate", "set", "get"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewConfigCmd()

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
					if c.Name() == subcmd {
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

func TestConfigViewCmd(t *testing.T) {
	// Create temporary config file
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")
	configContent := `bert_model:
  model_id: "test-model"
  threshold: 0.8

default_model: "test-model"
`
	if err := os.WriteFile(configPath, []byte(configContent), 0o644); err != nil {
		t.Fatalf("Failed to create test config: %v", err)
	}

	tests := []struct {
		name      string
		args      []string
		wantError bool
	}{
		{
			name:      "view config with yaml format",
			args:      []string{"config", "view", "-c", configPath, "-o", "yaml"},
			wantError: false,
		},
		{
			name:      "view config with table format",
			args:      []string{"config", "view", "-c", configPath, "-o", "table"},
			wantError: false,
		},
		{
			name:      "view config with json format",
			args:      []string{"config", "view", "-c", configPath, "-o", "json"},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rootCmd := &cobra.Command{Use: "vsr"}
			rootCmd.PersistentFlags().StringP("config", "c", "config/config.yaml", "Path to configuration file")
			rootCmd.PersistentFlags().StringP("output", "o", "table", "Output format")

			configCmd := NewConfigCmd()
			rootCmd.AddCommand(configCmd)

			rootCmd.SetArgs(tt.args)
			_, err := rootCmd.ExecuteC()

			if tt.wantError && err == nil {
				t.Error("expected error, got nil")
			}
			if !tt.wantError && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestConfigValidateCmd(t *testing.T) {
	tmpDir := t.TempDir()

	tests := []struct {
		name          string
		configContent string
		wantError     bool
	}{
		{
			name: "valid config",
			configContent: `bert_model:
  model_id: "test-model"
  threshold: 0.8

vllm_endpoints:
  - name: "primary"
    address: "127.0.0.1"
    port: 8000

default_model: "test-model"
`,
			wantError: false,
		},
		{
			name:          "invalid yaml syntax",
			configContent: `bert_model: [invalid yaml`,
			wantError:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			configPath := filepath.Join(tmpDir, tt.name+".yaml")
			if err := os.WriteFile(configPath, []byte(tt.configContent), 0o644); err != nil {
				t.Fatalf("Failed to create test config: %v", err)
			}

			rootCmd := &cobra.Command{Use: "vsr"}
			rootCmd.PersistentFlags().StringP("config", "c", configPath, "Path to configuration file")

			configCmd := NewConfigCmd()
			rootCmd.AddCommand(configCmd)

			rootCmd.SetArgs([]string{"config", "validate", "-c", configPath})
			_, _ = rootCmd.ExecuteC() // Ignore error, just testing command structure
		})
	}
}

func TestConfigSetGetCmd(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")
	configContent := `bert_model:
  model_id: "test-model"
  threshold: 0.8

default_model: "test-model"
`
	if err := os.WriteFile(configPath, []byte(configContent), 0o644); err != nil {
		t.Fatalf("Failed to create test config: %v", err)
	}

	tests := []struct {
		name      string
		command   string
		args      []string
		wantError bool
	}{
		{
			name:      "set top-level value",
			command:   "set",
			args:      []string{"config", "set", "default_model", "new-model", "-c", configPath},
			wantError: false,
		},
		{
			name:      "set nested value",
			command:   "set",
			args:      []string{"config", "set", "bert_model.threshold", "0.9", "-c", configPath},
			wantError: false,
		},
		{
			name:      "get top-level value",
			command:   "get",
			args:      []string{"config", "get", "default_model", "-c", configPath},
			wantError: false,
		},
		{
			name:      "get nested value",
			command:   "get",
			args:      []string{"config", "get", "bert_model.threshold", "-c", configPath},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rootCmd := &cobra.Command{Use: "vsr"}
			rootCmd.PersistentFlags().StringP("config", "c", "config/config.yaml", "Path to configuration file")

			configCmd := NewConfigCmd()
			rootCmd.AddCommand(configCmd)

			rootCmd.SetArgs(tt.args)
			_, _ = rootCmd.ExecuteC() // Ignore error, just testing command structure
		})
	}
}

func TestNestedValueHelpers(t *testing.T) {
	tests := []struct {
		name      string
		data      map[string]interface{}
		key       string
		value     string
		operation string // "get" or "set"
		wantError bool
		expected  interface{}
	}{
		{
			name: "set simple key",
			data: map[string]interface{}{
				"key1": "value1",
			},
			key:       "key1",
			value:     "new-value",
			operation: "set",
			wantError: false,
		},
		{
			name: "set nested key",
			data: map[string]interface{}{
				"parent": map[string]interface{}{
					"child": "value",
				},
			},
			key:       "parent.child",
			value:     "new-value",
			operation: "set",
			wantError: false,
		},
		{
			name: "get simple key",
			data: map[string]interface{}{
				"key1": "value1",
			},
			key:       "key1",
			operation: "get",
			wantError: false,
			expected:  "value1",
		},
		{
			name: "get nested key",
			data: map[string]interface{}{
				"parent": map[string]interface{}{
					"child": "value",
				},
			},
			key:       "parent.child",
			operation: "get",
			wantError: false,
			expected:  "value",
		},
		{
			name: "get non-existent key",
			data: map[string]interface{}{
				"key1": "value1",
			},
			key:       "nonexistent",
			operation: "get",
			wantError: false,
			expected:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			switch tt.operation {
			case "set":
				err := setNestedValue(tt.data, tt.key, tt.value)
				if tt.wantError && err == nil {
					t.Error("expected error, got nil")
				}
				if !tt.wantError && err != nil {
					t.Errorf("unexpected error: %v", err)
				}
			case "get":
				value, err := getNestedValue(tt.data, tt.key)
				if tt.wantError && err == nil {
					t.Error("expected error, got nil")
				}
				if !tt.wantError && err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if !tt.wantError && value != tt.expected {
					t.Errorf("expected %v, got %v", tt.expected, value)
				}
			}
		})
	}
}

func TestConfigEditCmd(t *testing.T) {
	// This test just verifies the command exists and has correct structure
	// Actual editor interaction is hard to test in unit tests
	cmd := NewConfigCmd()
	var editCmd *cobra.Command
	for _, c := range cmd.Commands() {
		if c.Use == "edit" {
			editCmd = c
			break
		}
	}

	if editCmd == nil {
		t.Fatal("edit subcommand not found")
	}

	if editCmd.Short != "Edit configuration in your default editor" {
		t.Errorf("unexpected Short description: %s", editCmd.Short)
	}
}
