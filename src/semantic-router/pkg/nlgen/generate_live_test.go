//go:build live

package nlgen

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

var nlTestCases = []struct {
	name        string
	instruction string
	mustContain []string
}{
	{
		name:        "simple_domain_routing",
		instruction: "Route math questions to qwen2.5-math:7b with reasoning enabled. Route coding questions to deepseek-coder:6.7b. Everything else goes to qwen2.5:3b as the default.",
		mustContain: []string{"SIGNAL", "MODEL"},
	},
	{
		name:        "jailbreak_safety_gate",
		instruction: "Create a safety gate: detect jailbreak attempts (threshold 0.85) and block them with a fast response saying 'Request blocked'. All other queries go to gpt-4o.",
		mustContain: []string{"jailbreak", "MODEL"},
	},
	{
		name:        "decision_tree_multi_domain",
		instruction: "Use a decision tree for routing. First check for jailbreak (block with fast-reject). Then check if math (route to qwen-math). Then check if coding (route to deepseek-coder). Default to qwen2.5:7b.",
		mustContain: []string{"DECISION_TREE", "IF", "ELSE", "MODEL"},
	},
	{
		name:        "keyword_routing",
		instruction: "Route queries containing words like 'urgent', 'emergency', or 'critical' to gpt-4o. Route queries with 'billing' or 'invoice' keywords to a billing model. Default to qwen2.5:3b.",
		mustContain: []string{"keyword", "MODEL"},
	},
}

func getEnvOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// Environment variables for live tests:
//
//	VLLM_API_KEY           - API key for vLLM endpoints (required)
//	VLLM_SMALL_BASE_URL    - Base URL for the smaller model (default: http://localhost:8091)
//	VLLM_SMALL_MODEL       - Model name for the smaller model (default: Qwen2.5-7B)
//	VLLM_LARGE_BASE_URL    - Base URL for the larger model (default: http://localhost:8090)
//	VLLM_LARGE_MODEL       - Model name for the larger model (default: Qwen2.5-72B)
func TestNLLive(t *testing.T) {
	apiKey := os.Getenv("VLLM_API_KEY")
	if apiKey == "" {
		t.Fatal("VLLM_API_KEY environment variable is required for live tests")
	}

	models := []struct {
		name    string
		baseURL string
		model   string
	}{
		{
			getEnvOrDefault("VLLM_SMALL_MODEL", "Qwen2.5-7B"),
			getEnvOrDefault("VLLM_SMALL_BASE_URL", "http://localhost:8091"),
			getEnvOrDefault("VLLM_SMALL_MODEL", "Qwen2.5-7B"),
		},
		{
			getEnvOrDefault("VLLM_LARGE_MODEL", "Qwen2.5-72B"),
			getEnvOrDefault("VLLM_LARGE_BASE_URL", "http://localhost:8090"),
			getEnvOrDefault("VLLM_LARGE_MODEL", "Qwen2.5-72B"),
		},
	}

	for _, m := range models {
		t.Run(m.name, func(t *testing.T) {
			client := NewOpenAIClient(m.baseURL, m.model, apiKey)

			for _, tc := range nlTestCases {
				t.Run(tc.name, func(t *testing.T) {
					ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
					defer cancel()

					result, err := GenerateFromNL(ctx, client, tc.instruction,
						WithTemperature(0.1),
						WithMaxTokens(4096),
						WithMaxRetries(2),
					)
					if err != nil {
						t.Fatalf("GenerateFromNL error: %v", err)
					}

					t.Logf("Attempts: %d", result.Attempts)
					t.Logf("Parse error: %q", result.ParseError)
					if len(result.Warnings) > 0 {
						t.Logf("Warnings: %v", result.Warnings)
					}
					t.Logf("DSL output:\n%s", result.DSL)

					if result.ParseError != "" {
						t.Errorf("PARSE FAILED: %s\nRaw output:\n%s", result.ParseError, result.RawOutput)
					}

					for _, substr := range tc.mustContain {
						if !strings.Contains(result.DSL, substr) {
							t.Errorf("missing expected substring %q in output", substr)
						}
					}
				})
			}
		})
	}
}
