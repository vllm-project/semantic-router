package handlers

import "testing"

// ============== Signal Normalization Tests ==============

func TestNormalizeSignalName(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "space to underscore",
			input:    "computer science",
			expected: "computer_science",
		},
		{
			name:     "multiple spaces",
			input:    "user  feedback  test",
			expected: "user__feedback__test",
		},
		{
			name:     "already normalized",
			input:    "code_keywords",
			expected: "code_keywords",
		},
		{
			name:     "empty string",
			input:    "",
			expected: "",
		},
		{
			name:     "no spaces",
			input:    "nospaces",
			expected: "nospaces",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := normalizeSignalName(tt.input)
			if result != tt.expected {
				t.Errorf("normalizeSignalName(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestNormalizeModelName(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "dots to dashes",
			input:    "qwen2.5-7b",
			expected: "qwen2-5-7b",
		},
		{
			name:     "colons to dashes",
			input:    "model:v1:latest",
			expected: "model-v1-latest",
		},
		{
			name:     "slashes to dashes",
			input:    "org/model/version",
			expected: "org-model-version",
		},
		{
			name:     "mixed special chars",
			input:    "gpt-4.0:turbo/v2",
			expected: "gpt-4-0-turbo-v2",
		},
		{
			name:     "already normalized",
			input:    "gpt-4",
			expected: "gpt-4",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := normalizeModelName(tt.input)
			if result != tt.expected {
				t.Errorf("normalizeModelName(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}
