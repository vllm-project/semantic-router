package commands

import (
	"bytes"
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

func TestUpgradeCommand(t *testing.T) {
	tests := []struct {
		name        string
		args        []string
		expectError bool
		errorMsg    string
	}{
		{
			name:        "upgrade without environment",
			args:        []string{},
			expectError: true,
			errorMsg:    "accepts 1 arg(s)",
		},
		{
			name:        "upgrade local",
			args:        []string{"local"},
			expectError: false,
		},
		{
			name:        "upgrade docker",
			args:        []string{"docker"},
			expectError: false,
		},
		{
			name:        "upgrade kubernetes",
			args:        []string{"kubernetes"},
			expectError: false,
		},
		{
			name:        "upgrade helm",
			args:        []string{"helm"},
			expectError: false,
		},
		{
			name:        "upgrade with too many args",
			args:        []string{"docker", "extra"},
			expectError: true,
			errorMsg:    "accepts 1 arg(s)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewUpgradeCmd()

			rootCmd := &cobra.Command{Use: "vsr"}
			rootCmd.PersistentFlags().StringP("config", "c", "config.yaml", "config file")
			rootCmd.AddCommand(cmd)

			rootCmd.SetArgs(append([]string{"upgrade"}, tt.args...))

			buf := new(bytes.Buffer)
			rootCmd.SetOut(buf)
			rootCmd.SetErr(buf)

			err := rootCmd.Execute()

			if tt.expectError {
				if err == nil {
					t.Errorf("Expected error but got none")
				} else if tt.errorMsg != "" && !strings.Contains(err.Error(), tt.errorMsg) {
					t.Errorf("Expected error containing %q, got %q", tt.errorMsg, err.Error())
				}
			}
		})
	}
}

func TestUpgradeCommandFlags(t *testing.T) {
	tests := []struct {
		name          string
		args          []string
		expectedFlags map[string]string
	}{
		{
			name: "default flags",
			args: []string{"docker"},
			expectedFlags: map[string]string{
				"with-observability": "true",
				"namespace":          "default",
				"force":              "false",
				"wait":               "false",
				"timeout":            "5m",
			},
		},
		{
			name: "with force flag",
			args: []string{"docker", "--force"},
			expectedFlags: map[string]string{
				"force": "true",
			},
		},
		{
			name: "with wait flag",
			args: []string{"kubernetes", "--wait"},
			expectedFlags: map[string]string{
				"wait": "true",
			},
		},
		{
			name: "with custom timeout",
			args: []string{"kubernetes", "--timeout", "10m"},
			expectedFlags: map[string]string{
				"timeout": "10m",
			},
		},
		{
			name: "with custom namespace",
			args: []string{"kubernetes", "--namespace", "production"},
			expectedFlags: map[string]string{
				"namespace": "production",
			},
		},
		{
			name: "without observability",
			args: []string{"docker", "--with-observability=false"},
			expectedFlags: map[string]string{
				"with-observability": "false",
			},
		},
		{
			name: "kubernetes with all options",
			args: []string{"kubernetes", "--namespace", "prod", "--wait", "--timeout", "15m", "--force"},
			expectedFlags: map[string]string{
				"namespace": "prod",
				"wait":      "true",
				"timeout":   "15m",
				"force":     "true",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewUpgradeCmd()

			rootCmd := &cobra.Command{Use: "vsr"}
			rootCmd.PersistentFlags().StringP("config", "c", "config.yaml", "config file")
			rootCmd.AddCommand(cmd)

			fullArgs := append([]string{"upgrade"}, tt.args...)
			rootCmd.SetArgs(fullArgs)

			// Parse command
			_, err := rootCmd.ExecuteC()
			_ = err // Ignore execution errors

			for flagName, expectedValue := range tt.expectedFlags {
				flag := cmd.Flags().Lookup(flagName)
				if flag == nil {
					t.Errorf("Flag %q not found", flagName)
					continue
				}
				if flag.Value.String() != expectedValue {
					t.Errorf("Flag %q: expected %q, got %q", flagName, expectedValue, flag.Value.String())
				}
			}
		})
	}
}

func TestUpgradeCommandTimeoutParsing(t *testing.T) {
	tests := []struct {
		name        string
		timeout     string
		expectError bool
	}{
		{
			name:        "valid timeout - minutes",
			timeout:     "5m",
			expectError: false,
		},
		{
			name:        "valid timeout - seconds",
			timeout:     "300s",
			expectError: false,
		},
		{
			name:        "valid timeout - hours",
			timeout:     "1h",
			expectError: false,
		},
		{
			name:        "invalid timeout - no unit",
			timeout:     "300",
			expectError: true,
		},
		{
			name:        "invalid timeout - wrong unit",
			timeout:     "5x",
			expectError: true,
		},
		{
			name:        "invalid timeout - empty",
			timeout:     "",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewUpgradeCmd()

			rootCmd := &cobra.Command{Use: "vsr"}
			rootCmd.PersistentFlags().StringP("config", "c", "config.yaml", "config file")
			rootCmd.AddCommand(cmd)

			// Force flag to skip confirmation
			args := []string{"upgrade", "docker", "--force", "--timeout", tt.timeout}
			rootCmd.SetArgs(args)

			buf := new(bytes.Buffer)
			rootCmd.SetOut(buf)
			rootCmd.SetErr(buf)

			err := rootCmd.Execute()

			if tt.expectError {
				if err == nil {
					t.Errorf("Expected error for timeout %q but got none", tt.timeout)
				} else if !strings.Contains(err.Error(), "invalid timeout") {
					t.Errorf("Expected 'invalid timeout' error, got: %v", err)
				}
			}
		})
	}
}

func TestUpgradeCommandHelp(t *testing.T) {
	cmd := NewUpgradeCmd()

	if cmd.Use != "upgrade [local|docker|kubernetes|helm]" {
		t.Errorf("Expected Use to include environment options, got: %s", cmd.Use)
	}

	if cmd.Short == "" {
		t.Error("Short description should not be empty")
	}

	if cmd.Long == "" {
		t.Error("Long description should not be empty")
	}

	// Check that Long contains examples
	if !strings.Contains(cmd.Long, "Examples:") {
		t.Error("Long description should contain examples")
	}

	// Check that all environments are mentioned
	for _, env := range []string{"local", "docker", "kubernetes", "helm"} {
		if !strings.Contains(cmd.Long, env) {
			t.Errorf("Long description should mention %s environment", env)
		}
	}
}
