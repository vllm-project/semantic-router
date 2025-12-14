package commands

import (
	"bytes"
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

func TestDeployCommand(t *testing.T) {
	tests := []struct {
		name        string
		args        []string
		expectError bool
		errorMsg    string
	}{
		{
			name:        "deploy without environment",
			args:        []string{},
			expectError: true,
			errorMsg:    "accepts 1 arg(s)",
		},
		{
			name:        "deploy with valid environment - local",
			args:        []string{"local"},
			expectError: false,
		},
		{
			name:        "deploy with valid environment - docker",
			args:        []string{"docker"},
			expectError: false,
		},
		{
			name:        "deploy with valid environment - kubernetes",
			args:        []string{"kubernetes"},
			expectError: false,
		},
		{
			name:        "deploy with invalid environment",
			args:        []string{"invalid"},
			expectError: false, // Command parsing succeeds, execution would fail
		},
		{
			name:        "deploy with too many args",
			args:        []string{"docker", "extra"},
			expectError: true,
			errorMsg:    "accepts 1 arg(s)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewDeployCmd()

			// Create a root command to attach flags properly
			rootCmd := &cobra.Command{Use: "vsr"}
			rootCmd.PersistentFlags().StringP("config", "c", "config.yaml", "config file")
			rootCmd.AddCommand(cmd)

			// Set args
			rootCmd.SetArgs(append([]string{"deploy"}, tt.args...))

			// Capture output
			buf := new(bytes.Buffer)
			rootCmd.SetOut(buf)
			rootCmd.SetErr(buf)

			// Execute
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

func TestDeployCommandFlags(t *testing.T) {
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
				"dry-run":            "false",
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
			name: "dry-run enabled",
			args: []string{"kubernetes", "--dry-run"},
			expectedFlags: map[string]string{
				"dry-run": "true",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewDeployCmd()

			rootCmd := &cobra.Command{Use: "vsr"}
			rootCmd.PersistentFlags().StringP("config", "c", "config.yaml", "config file")
			rootCmd.AddCommand(cmd)

			// Set args and parse
			fullArgs := append([]string{"deploy"}, tt.args...)
			rootCmd.SetArgs(fullArgs)

			// Parse command (this will parse the subcommand flags)
			_, err := rootCmd.ExecuteC()
			// Ignore execution errors, we're just testing flag parsing
			_ = err

			// Check flags
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

func TestUndeployCommand(t *testing.T) {
	tests := []struct {
		name        string
		args        []string
		expectError bool
		errorMsg    string
	}{
		{
			name:        "undeploy without environment",
			args:        []string{},
			expectError: true,
			errorMsg:    "accepts 1 arg(s)",
		},
		{
			name:        "undeploy local",
			args:        []string{"local"},
			expectError: false,
		},
		{
			name:        "undeploy docker",
			args:        []string{"docker"},
			expectError: false,
		},
		{
			name:        "undeploy kubernetes",
			args:        []string{"kubernetes"},
			expectError: false,
		},
		{
			name:        "undeploy with too many args",
			args:        []string{"docker", "extra"},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewUndeployCmd()

			rootCmd := &cobra.Command{Use: "vsr"}
			rootCmd.AddCommand(cmd)
			rootCmd.SetArgs(append([]string{"undeploy"}, tt.args...))

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

func TestUndeployCommandFlags(t *testing.T) {
	tests := []struct {
		name          string
		args          []string
		expectedFlags map[string]string
	}{
		{
			name: "default flags",
			args: []string{"docker"},
			expectedFlags: map[string]string{
				"namespace": "default",
				"volumes":   "false",
				"wait":      "false",
			},
		},
		{
			name: "with volumes flag",
			args: []string{"docker", "--volumes"},
			expectedFlags: map[string]string{
				"volumes": "true",
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
			name: "with custom namespace and wait",
			args: []string{"kubernetes", "--namespace", "prod", "--wait"},
			expectedFlags: map[string]string{
				"namespace": "prod",
				"wait":      "true",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewUndeployCmd()

			rootCmd := &cobra.Command{Use: "vsr"}
			rootCmd.AddCommand(cmd)

			fullArgs := append([]string{"undeploy"}, tt.args...)
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

func TestStartStopRestartCommands(t *testing.T) {
	tests := []struct {
		name    string
		cmdFunc func() *cobra.Command
		cmdName string
	}{
		{
			name:    "start command",
			cmdFunc: NewStartCmd,
			cmdName: "start",
		},
		{
			name:    "stop command",
			cmdFunc: NewStopCmd,
			cmdName: "stop",
		},
		{
			name:    "restart command",
			cmdFunc: NewRestartCmd,
			cmdName: "restart",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := tt.cmdFunc()

			if cmd.Use != tt.cmdName {
				t.Errorf("Expected Use=%q, got %q", tt.cmdName, cmd.Use)
			}

			// These commands should run without error (they just show warnings)
			rootCmd := &cobra.Command{Use: "vsr"}
			rootCmd.AddCommand(cmd)
			rootCmd.SetArgs([]string{tt.cmdName})

			err := rootCmd.Execute()
			if err != nil {
				t.Errorf("Command should not error: %v", err)
			}

			// The commands run successfully and print warnings
			// We can't easily capture the cli.Warning output
			// so we just verify they execute without error
		})
	}
}
