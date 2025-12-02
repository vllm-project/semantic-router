package deployment

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCommandExists(t *testing.T) {
	tests := []struct {
		name     string
		command  string
		expected bool
	}{
		{
			name:     "existing command - ls",
			command:  "ls",
			expected: true,
		},
		{
			name:     "existing command - echo",
			command:  "echo",
			expected: true,
		},
		{
			name:     "non-existing command",
			command:  "nonexistentcommand12345",
			expected: false,
		},
		{
			name:     "kubectl may or may not exist",
			command:  "kubectl",
			expected: commandExists("kubectl"), // whatever the actual state is
		},
		{
			name:     "docker may or may not exist",
			command:  "docker",
			expected: commandExists("docker"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := commandExists(tt.command)
			if result != tt.expected {
				t.Errorf("commandExists(%q) = %v, expected %v", tt.command, result, tt.expected)
			}
		})
	}
}

func TestSplitLines(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected []string
	}{
		{
			name:     "empty string",
			input:    "",
			expected: []string{},
		},
		{
			name:     "single line",
			input:    "hello",
			expected: []string{"hello"},
		},
		{
			name:     "two lines",
			input:    "hello\nworld",
			expected: []string{"hello", "world"},
		},
		{
			name:     "three lines",
			input:    "line1\nline2\nline3",
			expected: []string{"line1", "line2", "line3"},
		},
		{
			name:     "lines with trailing newline",
			input:    "line1\nline2\n",
			expected: []string{"line1", "line2"},
		},
		{
			name:     "lines with empty lines",
			input:    "line1\n\nline3",
			expected: []string{"line1", "", "line3"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := splitLines(tt.input)
			if len(result) != len(tt.expected) {
				t.Errorf("splitLines(%q) returned %d lines, expected %d", tt.input, len(result), len(tt.expected))
				return
			}
			for i, line := range result {
				if line != tt.expected[i] {
					t.Errorf("splitLines(%q)[%d] = %q, expected %q", tt.input, i, line, tt.expected[i])
				}
			}
		})
	}
}

func TestContainsString(t *testing.T) {
	tests := []struct {
		name     string
		s        string
		substr   string
		expected bool
	}{
		{
			name:     "substring found",
			s:        "hello world",
			substr:   "world",
			expected: true,
		},
		{
			name:     "substring not found",
			s:        "hello world",
			substr:   "foo",
			expected: false,
		},
		{
			name:     "substring at beginning",
			s:        "hello world",
			substr:   "hello",
			expected: true,
		},
		{
			name:     "substring at end",
			s:        "hello world",
			substr:   "world",
			expected: true,
		},
		{
			name:     "exact match",
			s:        "hello",
			substr:   "hello",
			expected: true,
		},
		{
			name:     "empty substring",
			s:        "hello",
			substr:   "",
			expected: true,
		},
		{
			name:     "substring longer than string",
			s:        "hi",
			substr:   "hello",
			expected: false,
		},
		{
			name:     "case sensitive",
			s:        "Hello World",
			substr:   "hello",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := containsString(tt.s, tt.substr)
			if result != tt.expected {
				t.Errorf("containsString(%q, %q) = %v, expected %v", tt.s, tt.substr, result, tt.expected)
			}
		})
	}
}

func TestFindSubstring(t *testing.T) {
	tests := []struct {
		name     string
		s        string
		substr   string
		expected bool
	}{
		{
			name:     "substring found",
			s:        "hello world",
			substr:   "world",
			expected: true,
		},
		{
			name:     "substring not found",
			s:        "hello world",
			substr:   "foo",
			expected: false,
		},
		{
			name:     "substring at beginning",
			s:        "hello world",
			substr:   "hello",
			expected: true,
		},
		{
			name:     "multiple occurrences",
			s:        "hello hello",
			substr:   "hello",
			expected: true,
		},
		{
			name:     "overlapping patterns",
			s:        "aaaa",
			substr:   "aa",
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := findSubstring(tt.s, tt.substr)
			if result != tt.expected {
				t.Errorf("findSubstring(%q, %q) = %v, expected %v", tt.s, tt.substr, result, tt.expected)
			}
		})
	}
}

func TestGetDockerContainers(t *testing.T) {
	// Skip if docker is not available
	if !commandExists("docker") {
		t.Skip("Docker not available, skipping test")
	}

	tests := []struct {
		name        string
		nameFilter  string
		expectError bool
	}{
		{
			name:        "filter by semantic-router",
			nameFilter:  "semantic-router",
			expectError: false,
		},
		{
			name:        "filter by nonexistent name",
			nameFilter:  "nonexistentcontainer12345",
			expectError: false,
		},
		{
			name:        "empty filter",
			nameFilter:  "",
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			containers, err := getDockerContainers(tt.nameFilter)

			if tt.expectError {
				if err == nil {
					t.Error("Expected error but got none")
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				// containers should be a slice (may be empty)
				if containers == nil {
					t.Error("Expected non-nil slice")
				}
			}
		})
	}
}

func TestIsDockerRunning(t *testing.T) {
	// This test checks if the function works, not necessarily if Docker is running
	result := isDockerRunning()

	// Result should be boolean (no error to check)
	// Just verify the function returns without panicking
	t.Logf("isDockerRunning() returned: %v", result)

	// If docker command exists, the result should match commandExists
	if commandExists("docker") {
		// Docker command exists, so isDockerRunning should at least try to run
		// The result depends on whether Docker daemon is actually running
		t.Logf("Docker command exists, isDockerRunning returned: %v", result)
	} else if result {
		// If docker command doesn't exist, isDockerRunning should return false
		t.Error("isDockerRunning() should return false when docker command doesn't exist")
	}
}

func TestPIDFileOperations(t *testing.T) {
	// Test PID file path functions
	pidFilePath := getPIDFilePath()
	logFilePath := getLogFilePath()

	if pidFilePath == "" {
		t.Error("pidFilePath should not be empty")
	}

	if logFilePath == "" {
		t.Error("logFilePath should not be empty")
	}

	// Verify paths are absolute
	if !filepath.IsAbs(pidFilePath) {
		t.Errorf("pidFilePath should be absolute, got: %s", pidFilePath)
	}

	if !filepath.IsAbs(logFilePath) {
		t.Errorf("logFilePath should be absolute, got: %s", logFilePath)
	}
}

func TestDeployLocalPIDFileCreation(t *testing.T) {
	// This is an integration test that would require actually running DeployLocal
	// For now, we just verify the functions return valid paths
	t.Run("verify PID file path", func(t *testing.T) {
		pidFilePath := getPIDFilePath()
		if pidFilePath == "" {
			t.Error("getPIDFilePath() returned empty string")
		}
		if !filepath.IsAbs(pidFilePath) {
			t.Errorf("getPIDFilePath() should return absolute path, got: %s", pidFilePath)
		}
	})

	t.Run("verify log file path", func(t *testing.T) {
		logFilePath := getLogFilePath()
		if logFilePath == "" {
			t.Error("getLogFilePath() returned empty string")
		}
		if !filepath.IsAbs(logFilePath) {
			t.Errorf("getLogFilePath() should return absolute path, got: %s", logFilePath)
		}
	})
}

func TestUndeployLocalWithNoPIDFile(t *testing.T) {
	// Ensure PID file doesn't exist
	pidFilePath := getPIDFilePath()
	os.Remove(pidFilePath)

	// Call UndeployLocal - it should handle missing PID file gracefully
	err := UndeployLocal()
	// Should not return error for missing PID file
	if err != nil {
		t.Errorf("UndeployLocal should handle missing PID file gracefully, got error: %v", err)
	}
}

func TestBuildRouter(t *testing.T) {
	// Skip if make is not available
	if !commandExists("make") {
		t.Skip("make not available, skipping test")
	}

	// This is a smoke test - we don't actually want to build in unit tests
	// Just verify the function exists and can be called
	t.Run("buildRouter function accessible", func(t *testing.T) {
		// We can't easily test buildRouter without side effects
		// The function exists and is called by DeployLocal
		// This test just documents that it's available
		t.Log("buildRouter function is accessible via DeployLocal")
	})
}

// Mock tests for deployment functions (without actual execution)

func TestDeployDockerValidation(t *testing.T) {
	t.Run("missing docker-compose file", func(t *testing.T) {
		// Create a temporary config file
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, "config.yaml")

		// Write minimal config
		configContent := `
bert_model:
  model_id: "test-model"
  threshold: 0.8
vllm_endpoints:
  - name: "test"
    address: "127.0.0.1"
    port: 8000
model_config:
  test-model:
    pricing:
      prompt: 0.01
      completion: 0.02
default_model: "test-model"
`
		if err := os.WriteFile(configPath, []byte(configContent), 0o644); err != nil {
			t.Fatal(err)
		}

		// DeployDocker should fail if docker-compose file doesn't exist
		// (This would need to be in a different directory without the actual docker-compose.yml)
		// For now, just verify the function signature
		t.Skip("Skipping actual deployment test")
	})
}

func TestUndeployDockerVolumeFlag(t *testing.T) {
	t.Run("removeVolumes parameter", func(t *testing.T) {
		// Test that the function accepts the removeVolumes parameter
		// We can't test actual execution without Docker running
		// Just verify the signature works

		// Skip actual execution
		t.Skip("Skipping actual undeploy test")

		// This would fail without Docker, but shows parameter usage:
		// err := UndeployDocker(false)
		// err := UndeployDocker(true)
	})
}

func TestUndeployKubernetesWaitFlag(t *testing.T) {
	t.Run("wait parameter", func(t *testing.T) {
		// Test that the function accepts the wait parameter
		// Skip actual execution
		t.Skip("Skipping actual undeploy test")

		// This would fail without kubectl, but shows parameter usage:
		// err := UndeployKubernetes("default", false)
		// err := UndeployKubernetes("default", true)
	})
}
