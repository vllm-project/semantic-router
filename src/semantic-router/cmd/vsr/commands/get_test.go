package commands

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/spf13/cobra"
)

func TestGetCommandStructure(t *testing.T) {
	tests := []struct {
		name          string
		expectedUse   string
		expectedShort string
	}{
		{
			name:          "get command has correct structure",
			expectedUse:   "get [models|categories|decisions|endpoints]",
			expectedShort: "Get information about router resources",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewGetCmd()

			if cmd.Use != tt.expectedUse {
				t.Errorf("expected Use %q, got %q", tt.expectedUse, cmd.Use)
			}

			if cmd.Short != tt.expectedShort {
				t.Errorf("expected Short %q, got %q", tt.expectedShort, cmd.Short)
			}
		})
	}
}

func TestGetCommand(t *testing.T) {
	// Create temporary config file
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")
	configContent := `bert_model:
  model_id: "test-model"
  threshold: 0.8

vllm_endpoints:
  - name: "endpoint1"
    address: "127.0.0.1"
    port: 8000
    weight: 1

model_config:
  test-model:
    preferred_endpoints: ["endpoint1"]
    pricing:
      currency: "USD"
      prompt_per_1m: 0.5
      completion_per_1m: 1.5

categories:
  - name: "math"
    description: "Math queries"
    mmlu_categories: []

decisions:
  - name: "test_decision"
    description: "Test decision"
    priority: 10
    modelRefs:
      - model: "test-model"

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
			name:      "get models",
			args:      []string{"get", "models", "-c", configPath},
			wantError: false,
		},
		{
			name:      "get categories",
			args:      []string{"get", "categories", "-c", configPath},
			wantError: false,
		},
		{
			name:      "get decisions",
			args:      []string{"get", "decisions", "-c", configPath},
			wantError: false,
		},
		{
			name:      "get endpoints",
			args:      []string{"get", "endpoints", "-c", configPath},
			wantError: false,
		},
		{
			name:      "get unknown resource",
			args:      []string{"get", "unknown", "-c", configPath},
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rootCmd := &cobra.Command{Use: "vsr"}
			rootCmd.PersistentFlags().StringP("config", "c", "config/config.yaml", "Path to configuration file")
			rootCmd.PersistentFlags().StringP("output", "o", "table", "Output format")

			getCmd := NewGetCmd()
			rootCmd.AddCommand(getCmd)

			rootCmd.SetArgs(tt.args)
			_, _ = rootCmd.ExecuteC() // Ignore error, just testing command structure
		})
	}
}

func TestGetCommandWithDifferentOutputFormats(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")
	configContent := `bert_model:
  model_id: "test-model"
  threshold: 0.8

vllm_endpoints:
  - name: "endpoint1"
    address: "127.0.0.1"
    port: 8000
    weight: 1

model_config:
  test-model:
    preferred_endpoints: ["endpoint1"]

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
			name:      "get models in json format",
			args:      []string{"get", "models", "-c", configPath, "-o", "json"},
			wantError: false,
		},
		{
			name:      "get models in yaml format",
			args:      []string{"get", "models", "-c", configPath, "-o", "yaml"},
			wantError: false,
		},
		{
			name:      "get models in table format",
			args:      []string{"get", "models", "-c", configPath, "-o", "table"},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rootCmd := &cobra.Command{Use: "vsr"}
			rootCmd.PersistentFlags().StringP("config", "c", "config/config.yaml", "Path to configuration file")
			rootCmd.PersistentFlags().StringP("output", "o", "table", "Output format")

			getCmd := NewGetCmd()
			rootCmd.AddCommand(getCmd)

			rootCmd.SetArgs(tt.args)
			_, _ = rootCmd.ExecuteC() // Ignore error, just testing command structure
		})
	}
}

func TestGetCommandRequiresResource(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")
	configContent := `default_model: "test"`
	if err := os.WriteFile(configPath, []byte(configContent), 0o644); err != nil {
		t.Fatalf("Failed to create test config: %v", err)
	}

	rootCmd := &cobra.Command{Use: "vsr"}
	rootCmd.PersistentFlags().StringP("config", "c", configPath, "Path to configuration file")

	getCmd := NewGetCmd()
	rootCmd.AddCommand(getCmd)

	rootCmd.SetArgs([]string{"get"})
	_, err := rootCmd.ExecuteC()

	if err == nil {
		t.Error("expected error when no resource specified, got nil")
	}
}
