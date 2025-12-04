package commands

import (
	"testing"

	"github.com/spf13/cobra"
)

func TestCompletionCommandStructure(t *testing.T) {
	tests := []struct {
		name           string
		expectedUse    string
		expectedShort  string
		validArgs      []string
		validArgsCount int
	}{
		{
			name:           "completion command has correct structure",
			expectedUse:    "completion [bash|zsh|fish|powershell]",
			expectedShort:  "Generate shell completion script",
			validArgs:      []string{"bash", "zsh", "fish", "powershell"},
			validArgsCount: 4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewCompletionCmd()

			if cmd.Use != tt.expectedUse {
				t.Errorf("expected Use %q, got %q", tt.expectedUse, cmd.Use)
			}

			if cmd.Short != tt.expectedShort {
				t.Errorf("expected Short %q, got %q", tt.expectedShort, cmd.Short)
			}

			if len(cmd.ValidArgs) != tt.validArgsCount {
				t.Errorf("expected %d valid args, got %d", tt.validArgsCount, len(cmd.ValidArgs))
			}

			// Verify valid args
			for _, expectedArg := range tt.validArgs {
				found := false
				for _, validArg := range cmd.ValidArgs {
					if validArg == expectedArg {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("expected valid arg %q not found", expectedArg)
				}
			}
		})
	}
}

func TestCompletionCommand(t *testing.T) {
	tests := []struct {
		name      string
		args      []string
		wantError bool
	}{
		{
			name:      "completion bash",
			args:      []string{"completion", "bash"},
			wantError: false,
		},
		{
			name:      "completion zsh",
			args:      []string{"completion", "zsh"},
			wantError: false,
		},
		{
			name:      "completion fish",
			args:      []string{"completion", "fish"},
			wantError: false,
		},
		{
			name:      "completion powershell",
			args:      []string{"completion", "powershell"},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rootCmd := &cobra.Command{Use: "vsr"}

			completionCmd := NewCompletionCmd()
			rootCmd.AddCommand(completionCmd)

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

func TestCompletionCommandRequiresShell(t *testing.T) {
	rootCmd := &cobra.Command{Use: "vsr"}
	completionCmd := NewCompletionCmd()
	rootCmd.AddCommand(completionCmd)

	rootCmd.SetArgs([]string{"completion"})
	_, err := rootCmd.ExecuteC()

	if err == nil {
		t.Error("expected error when no shell specified, got nil")
	}
}

func TestCompletionCommandInvalidShell(t *testing.T) {
	rootCmd := &cobra.Command{Use: "vsr"}
	completionCmd := NewCompletionCmd()
	rootCmd.AddCommand(completionCmd)

	rootCmd.SetArgs([]string{"completion", "invalid-shell"})
	_, err := rootCmd.ExecuteC()

	// Cobra will return an error for invalid arg, which is expected
	if err == nil {
		t.Error("expected error for invalid shell, got nil")
	}
}
