package debug

import (
	"os"
	"path/filepath"
	"testing"
)

func TestGetSystemInfo(t *testing.T) {
	info := GetSystemInfo()

	if info.OS == "" {
		t.Error("Expected OS to be set")
	}

	if info.Architecture == "" {
		t.Error("Expected Architecture to be set")
	}

	if info.GoVersion == "" {
		t.Error("Expected GoVersion to be set")
	}
}

func TestCheckPrerequisites(t *testing.T) {
	results := CheckPrerequisites()

	// Should have at least some results
	if len(results) == 0 {
		t.Error("Expected at least some prerequisite checks")
	}

	// Check that Go is always present (since we're running in Go)
	hasGo := false
	for _, result := range results {
		if result.Name == "Go" {
			hasGo = true
			if result.Status != "pass" {
				t.Error("Expected Go to pass (since we're running in Go)")
			}
		}
	}

	if !hasGo {
		t.Error("Expected Go to be in prerequisite checks")
	}
}

func TestCheckConfiguration(t *testing.T) {
	t.Run("nonexistent config file", func(t *testing.T) {
		results := CheckConfiguration("/nonexistent/config.yaml")

		// Should have at least one result
		if len(results) == 0 {
			t.Error("Expected at least one result")
		}

		// First result should be about missing file
		if results[0].Status != "fail" {
			t.Error("Expected fail status for nonexistent config")
		}
	})

	t.Run("invalid config file", func(t *testing.T) {
		// Create a temp invalid config
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, "config.yaml")
		_ = os.WriteFile(configPath, []byte("invalid: yaml: content: ["), 0o644)

		results := CheckConfiguration(configPath)

		// Should detect parse failure
		hasParseError := false
		for _, result := range results {
			if result.Name == "Config Parse" && result.Status == "fail" {
				hasParseError = true
			}
		}

		if !hasParseError {
			t.Error("Expected parse error for invalid config")
		}
	})
}

func TestCheckModelStatus(t *testing.T) {
	t.Run("nonexistent models directory", func(t *testing.T) {
		results := CheckModelStatus("/nonexistent/models")

		// Should have at least one result
		if len(results) == 0 {
			t.Error("Expected at least one result")
		}

		// Should fail
		if results[0].Status != "fail" {
			t.Error("Expected fail status for nonexistent models dir")
		}
	})

	t.Run("empty models directory", func(t *testing.T) {
		tmpDir := t.TempDir()

		results := CheckModelStatus(tmpDir)

		// Should have results
		if len(results) < 2 {
			t.Error("Expected at least 2 results (directory + model files)")
		}

		// Directory should pass
		if results[0].Status != "pass" {
			t.Error("Expected pass status for existing directory")
		}

		// Model files check should warn
		if results[1].Status != "warn" {
			t.Error("Expected warn status for no model files")
		}
	})
}

func TestCheckResources(t *testing.T) {
	results := CheckResources()

	// Should have at least disk space check
	if len(results) == 0 {
		t.Error("Expected at least one resource check")
	}

	hasDiskCheck := false
	for _, result := range results {
		if result.Name == "Disk Space" {
			hasDiskCheck = true
			// Disk check should have a message
			if result.Message == "" {
				t.Error("Expected disk space message")
			}
		}
	}

	if !hasDiskCheck {
		t.Error("Expected disk space check")
	}
}

func TestCheckConnectivity(t *testing.T) {
	t.Run("invalid endpoint", func(t *testing.T) {
		results := CheckConnectivity([]string{"http://invalid-endpoint-12345:9999"})

		// Should have one result
		if len(results) != 1 {
			t.Errorf("Expected 1 result, got %d", len(results))
		}

		// Should fail
		if results[0].Status != "fail" {
			t.Error("Expected fail status for invalid endpoint")
		}
	})

	t.Run("default endpoints", func(t *testing.T) {
		results := CheckConnectivity(nil)

		// Should check default endpoints
		if len(results) == 0 {
			t.Error("Expected at least one default endpoint check")
		}
	})
}

func TestIsPortAvailable(t *testing.T) {
	// Test with a very high port that's unlikely to be in use
	highPort := 54321

	// Should be available (or we can't test accurately)
	result := isPortAvailable(highPort)

	// Just verify the function runs without error
	_ = result
}

func TestGetStatusSymbol(t *testing.T) {
	tests := []struct {
		status   string
		expected string
	}{
		{"pass", "✓"},
		{"fail", "✗"},
		{"warn", "⚠"},
		{"unknown", "•"},
	}

	for _, tt := range tests {
		result := getStatusSymbol(tt.status)
		if result != tt.expected {
			t.Errorf("getStatusSymbol(%s) = %s, expected %s", tt.status, result, tt.expected)
		}
	}
}

func TestGenerateRecommendations(t *testing.T) {
	t.Run("all pass", func(t *testing.T) {
		report := &DiagnosticReport{
			Prerequisites: []CheckResult{
				{Name: "Go", Status: "pass"},
			},
			Configuration: []CheckResult{
				{Name: "Config File", Status: "pass"},
			},
			ModelStatus: []CheckResult{
				{Name: "Models", Status: "pass"},
			},
			Resources: []CheckResult{
				{Name: "Disk Space", Status: "pass"},
			},
		}

		recommendations := GenerateRecommendations(report)

		// Should have at least one recommendation (deploy)
		if len(recommendations) == 0 {
			t.Error("Expected at least one recommendation")
		}
	})

	t.Run("kubectl missing", func(t *testing.T) {
		report := &DiagnosticReport{
			Prerequisites: []CheckResult{
				{Name: "kubectl", Status: "fail"},
			},
		}

		recommendations := GenerateRecommendations(report)

		// Should recommend installing kubectl
		hasKubectlRec := false
		for _, rec := range recommendations {
			if containsIgnoreCase(rec, "kubectl") {
				hasKubectlRec = true
			}
		}

		if !hasKubectlRec {
			t.Error("Expected recommendation to install kubectl")
		}
	})

	t.Run("models missing", func(t *testing.T) {
		report := &DiagnosticReport{
			ModelStatus: []CheckResult{
				{Name: "Model Files", Status: "warn"},
			},
		}

		recommendations := GenerateRecommendations(report)

		// Should recommend downloading models
		hasModelRec := false
		for _, rec := range recommendations {
			if containsIgnoreCase(rec, "model") {
				hasModelRec = true
			}
		}

		if !hasModelRec {
			t.Error("Expected recommendation to download models")
		}
	})
}

func TestRunFullDiagnostics(t *testing.T) {
	t.Run("with nonexistent paths", func(t *testing.T) {
		report := RunFullDiagnostics("/nonexistent/config.yaml", "/nonexistent/models")

		// Should have a report
		if report == nil {
			t.Error("Expected non-nil report")
		}

		// Should have timestamp
		if report.Timestamp.IsZero() {
			t.Error("Expected timestamp to be set")
		}

		// Should have system info
		if report.SystemInfo.OS == "" {
			t.Error("Expected system info to be set")
		}

		// Should have some checks
		totalChecks := len(report.Prerequisites) + len(report.Configuration) +
			len(report.ModelStatus) + len(report.Resources) + len(report.Connectivity)

		if totalChecks == 0 {
			t.Error("Expected at least some checks to be run")
		}
	})
}

// Helper function
func containsIgnoreCase(s, substr string) bool {
	s = toLower(s)
	substr = toLower(substr)
	return contains(s, substr)
}

func toLower(s string) string {
	result := []rune{}
	for _, r := range s {
		if r >= 'A' && r <= 'Z' {
			result = append(result, r+32)
		} else {
			result = append(result, r)
		}
	}
	return string(result)
}

func contains(s, substr string) bool {
	return findIndex(s, substr) >= 0
}

func findIndex(s, substr string) int {
	if len(substr) == 0 {
		return 0
	}
	if len(substr) > len(s) {
		return -1
	}
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}
