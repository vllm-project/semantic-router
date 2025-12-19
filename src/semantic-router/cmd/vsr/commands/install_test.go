package commands

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/spf13/cobra"
)

func TestInstallCommandStructure(t *testing.T) {
	tests := []struct {
		name          string
		expectedUse   string
		expectedShort string
	}{
		{
			name:          "install command has correct structure",
			expectedUse:   "install",
			expectedShort: "Install vLLM Semantic Router",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewInstallCmd()

			if cmd.Use != tt.expectedUse {
				t.Errorf("expected Use %q, got %q", tt.expectedUse, cmd.Use)
			}

			if cmd.Short != tt.expectedShort {
				t.Errorf("expected Short %q, got %q", tt.expectedShort, cmd.Short)
			}
		})
	}
}

func TestInstallCommand(t *testing.T) {
	tests := []struct {
		name      string
		args      []string
		wantError bool
	}{
		{
			name:      "install command runs",
			args:      []string{"install"},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rootCmd := &cobra.Command{Use: "vsr"}
			installCmd := NewInstallCmd()
			rootCmd.AddCommand(installCmd)

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

func TestInitCommandStructure(t *testing.T) {
	tests := []struct {
		name          string
		expectedUse   string
		expectedShort string
		hasFlags      []string
	}{
		{
			name:          "init command has correct structure",
			expectedUse:   "init",
			expectedShort: "Initialize a new configuration file",
			hasFlags:      []string{"output", "template"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewInitCmd()

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

func TestInitCommand(t *testing.T) {
	tmpDir := t.TempDir()

	tests := []struct {
		name      string
		args      []string
		wantError bool
		checkFile bool
	}{
		{
			name:      "init with default template",
			args:      []string{"init", "--output", filepath.Join(tmpDir, "test1.yaml")},
			wantError: false,
			checkFile: true,
		},
		{
			name:      "init with minimal template",
			args:      []string{"init", "--output", filepath.Join(tmpDir, "test2.yaml"), "--template", "minimal"},
			wantError: false,
			checkFile: true,
		},
		{
			name:      "init with full template",
			args:      []string{"init", "--output", filepath.Join(tmpDir, "test3.yaml"), "--template", "full"},
			wantError: false,
			checkFile: true,
		},
		{
			name:      "init with custom output path",
			args:      []string{"init", "--output", filepath.Join(tmpDir, "custom/config.yaml")},
			wantError: false,
			checkFile: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rootCmd := &cobra.Command{Use: "vsr"}
			initCmd := NewInitCmd()
			rootCmd.AddCommand(initCmd)

			rootCmd.SetArgs(tt.args)
			_, err := rootCmd.ExecuteC()

			if tt.wantError && err == nil {
				t.Error("expected error, got nil")
			}
			if !tt.wantError && err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			// Check if file was created
			if tt.checkFile && !tt.wantError {
				outputPath := ""
				for i, arg := range tt.args {
					if arg == "--output" && i+1 < len(tt.args) {
						outputPath = tt.args[i+1]
						break
					}
				}
				if outputPath != "" {
					if _, err := os.Stat(outputPath); os.IsNotExist(err) {
						t.Errorf("expected file to be created at %s", outputPath)
					}
				}
			}
		})
	}
}

func TestInitCommandFileExists(t *testing.T) {
	tmpDir := t.TempDir()
	existingFile := filepath.Join(tmpDir, "existing.yaml")

	// Create existing file
	if err := os.WriteFile(existingFile, []byte("existing content"), 0o644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	rootCmd := &cobra.Command{Use: "vsr"}
	initCmd := NewInitCmd()
	rootCmd.AddCommand(initCmd)

	rootCmd.SetArgs([]string{"init", "--output", existingFile})
	_, err := rootCmd.ExecuteC()

	if err == nil {
		t.Error("expected error when file exists, got nil")
	}
}

func TestInitCommandFlags(t *testing.T) {
	cmd := NewInitCmd()

	tests := []struct {
		flagName     string
		flagType     string
		defaultValue string
	}{
		{
			flagName:     "output",
			flagType:     "string",
			defaultValue: "config/config.yaml",
		},
		{
			flagName:     "template",
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

			if flag.DefValue != tt.defaultValue {
				t.Errorf("expected default value %q, got %q", tt.defaultValue, flag.DefValue)
			}
		})
	}
}

func TestGetTemplate(t *testing.T) {
	tests := []struct {
		name          string
		template      string
		shouldBeEmpty bool
	}{
		{
			name:          "default template",
			template:      "default",
			shouldBeEmpty: false,
		},
		{
			name:          "minimal template",
			template:      "minimal",
			shouldBeEmpty: false,
		},
		{
			name:          "full template",
			template:      "full",
			shouldBeEmpty: false,
		},
		{
			name:          "unknown template defaults to default",
			template:      "unknown",
			shouldBeEmpty: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := getTemplate(tt.template)

			if tt.shouldBeEmpty && result != "" {
				t.Error("expected empty template")
			}
			if !tt.shouldBeEmpty && result == "" {
				t.Error("expected non-empty template")
			}
		})
	}
}

func TestTemplateContent(t *testing.T) {
	tests := []struct {
		name             string
		template         string
		shouldContain    []string
		shouldNotContain []string
	}{
		{
			name:     "default template contains required fields",
			template: "default",
			shouldContain: []string{
				"bert_model:",
				"vllm_endpoints:",
				"model_config:",
				"categories:",
				"default_model:",
			},
		},
		{
			name:     "minimal template contains minimal fields",
			template: "minimal",
			shouldContain: []string{
				"bert_model:",
				"vllm_endpoints:",
				"default_model:",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			content := getTemplate(tt.template)

			for _, substring := range tt.shouldContain {
				if !containsString(content, substring) {
					t.Errorf("template should contain %q", substring)
				}
			}

			for _, substring := range tt.shouldNotContain {
				if containsString(content, substring) {
					t.Errorf("template should not contain %q", substring)
				}
			}
		})
	}
}

// Helper function to check if string contains substring
func containsString(s, substr string) bool {
	return len(s) >= len(substr) && findSubstring(s, substr) != -1
}

// Helper function to find substring index
func findSubstring(s, substr string) int {
	if len(substr) == 0 {
		return 0
	}
	if len(substr) > len(s) {
		return -1
	}
	for i := 0; i <= len(s)-len(substr); i++ {
		match := true
		for j := 0; j < len(substr); j++ {
			if s[i+j] != substr[j] {
				match = false
				break
			}
		}
		if match {
			return i
		}
	}
	return -1
}
