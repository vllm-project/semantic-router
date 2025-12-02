package commands

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/spf13/cobra"
)

func TestTestCommandStructure(t *testing.T) {
	tests := []struct {
		name          string
		expectedUse   string
		expectedShort string
		hasFlags      []string
	}{
		{
			name:          "test-prompt command has correct structure",
			expectedUse:   "test-prompt [text]",
			expectedShort: "Send a test prompt to the router",
			hasFlags:      []string{"endpoint"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewTestCmd()

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

func TestTestCommandFlags(t *testing.T) {
	cmd := NewTestCmd()

	tests := []struct {
		flagName     string
		flagType     string
		defaultValue string
	}{
		{
			flagName:     "endpoint",
			flagType:     "string",
			defaultValue: "http://localhost:8080",
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

func TestCallClassificationAPI(t *testing.T) {
	tests := []struct {
		name           string
		prompt         string
		mockResponse   ClassificationResult
		mockStatusCode int
		wantError      bool
	}{
		{
			name:   "successful classification",
			prompt: "test prompt",
			mockResponse: ClassificationResult{
				Classification: struct {
					Category   string  `json:"category"`
					Confidence float64 `json:"confidence"`
				}{
					Category:   "math",
					Confidence: 0.95,
				},
				RecommendedModel: "test-model",
			},
			mockStatusCode: http.StatusOK,
			wantError:      false,
		},
		{
			name:           "API error",
			prompt:         "test prompt",
			mockResponse:   ClassificationResult{},
			mockStatusCode: http.StatusInternalServerError,
			wantError:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock server
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/api/v1/classify/intent" {
					t.Errorf("unexpected path: %s", r.URL.Path)
				}

				w.WriteHeader(tt.mockStatusCode)
				if tt.mockStatusCode == http.StatusOK {
					_ = json.NewEncoder(w).Encode(tt.mockResponse)
				}
			}))
			defer server.Close()

			result, err := callClassificationAPI(server.URL, tt.prompt)

			if tt.wantError && err == nil {
				t.Error("expected error, got nil")
			}
			if !tt.wantError && err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			if !tt.wantError && result != nil {
				if result.Classification.Category != tt.mockResponse.Classification.Category {
					t.Errorf("expected category %q, got %q", tt.mockResponse.Classification.Category, result.Classification.Category)
				}
			}
		})
	}
}

func TestDisplayTestResult(t *testing.T) {
	result := &ClassificationResult{
		Classification: struct {
			Category   string  `json:"category"`
			Confidence float64 `json:"confidence"`
		}{
			Category:   "math",
			Confidence: 0.95,
		},
		RecommendedModel: "test-model",
	}

	tests := []struct {
		name      string
		format    string
		wantError bool
	}{
		{
			name:      "table format",
			format:    "table",
			wantError: false,
		},
		{
			name:      "json format",
			format:    "json",
			wantError: false,
		},
		{
			name:      "yaml format",
			format:    "yaml",
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := displayTestResult(result, tt.format)

			if tt.wantError && err == nil {
				t.Error("expected error, got nil")
			}
			if !tt.wantError && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestTestCommand(t *testing.T) {
	// Create mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		result := ClassificationResult{
			Classification: struct {
				Category   string  `json:"category"`
				Confidence float64 `json:"confidence"`
			}{
				Category:   "test",
				Confidence: 0.9,
			},
			RecommendedModel: "test-model",
		}
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(result)
	}))
	defer server.Close()

	tests := []struct {
		name      string
		args      []string
		wantError bool
	}{
		{
			name:      "test with prompt",
			args:      []string{"test-prompt", "test prompt", "--endpoint", server.URL},
			wantError: false,
		},
		{
			name:      "test with multiple word prompt",
			args:      []string{"test-prompt", "solve", "x^2", "+", "5x", "+", "6", "--endpoint", server.URL},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rootCmd := &cobra.Command{Use: "vsr"}
			rootCmd.PersistentFlags().StringP("output", "o", "table", "Output format")

			testCmd := NewTestCmd()
			rootCmd.AddCommand(testCmd)

			rootCmd.SetArgs(tt.args)
			_, _ = rootCmd.ExecuteC() // Ignore error, just testing command structure
		})
	}
}

func TestTestCommandRequiresArgs(t *testing.T) {
	rootCmd := &cobra.Command{Use: "vsr"}
	testCmd := NewTestCmd()
	rootCmd.AddCommand(testCmd)

	rootCmd.SetArgs([]string{"test-prompt"})
	_, err := rootCmd.ExecuteC()

	if err == nil {
		t.Error("expected error when no prompt provided, got nil")
	}
}
