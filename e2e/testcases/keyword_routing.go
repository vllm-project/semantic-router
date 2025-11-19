package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("keyword-routing", pkgtestcases.TestCase{
		Description: "Test keyword routing accuracy and verify routing decisions",
		Tags:        []string{"ai-gateway", "keyword-routing", "classification"},
		Fn:          testKeywordRouting,
	})
}

// KeywordRoutingCase represents a test case for keyword routing
type KeywordRoutingCase struct {
	Name               string   `json:"name"`
	Description        string   `json:"description"`
	Query              string   `json:"query"`
	ExpectedCategory   string   `json:"expected_category"`
	ExpectedConfidence float64  `json:"expected_confidence"`
	MatchedKeywords    []string `json:"matched_keywords"`
}

// KeywordRoutingResult tracks the result of a single keyword routing test
type KeywordRoutingResult struct {
	Query            string
	ExpectedCategory string
	ActualCategory   string
	Correct          bool
	Error            string
}

func testKeywordRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing keyword routing accuracy")
	}

	// Setup service connection and get local port
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// Load test cases from JSON file
	testCases, err := loadKeywordRoutingCases("testdata/keyword_routing_cases.json")
	if err != nil {
		return fmt.Errorf("failed to load test cases: %w", err)
	}

	if opts.Verbose {
		fmt.Printf("[Test] Loaded %d keyword routing test cases\n", len(testCases))
	}

	// Test each case
	results := make([]KeywordRoutingResult, 0, len(testCases))
	successCount := 0

	for i, tc := range testCases {
		if opts.Verbose {
			fmt.Printf("[Test] %d/%d: Testing %s\n", i+1, len(testCases), tc.Name)
		}

		result := KeywordRoutingResult{
			Query:            tc.Query,
			ExpectedCategory: tc.ExpectedCategory,
		}

		// Make classification request
		category, err := classifyKeywordQuery(ctx, localPort, tc.Query)
		if err != nil {
			result.Error = err.Error()
			results = append(results, result)
			continue
		}

		result.ActualCategory = category
		result.Correct = (category == tc.ExpectedCategory)

		if result.Correct {
			successCount++
		}

		results = append(results, result)

		if opts.Verbose {
			if result.Correct {
				fmt.Printf("  ✓ PASS: Expected '%s', got '%s'\n", tc.ExpectedCategory, category)
			} else {
				fmt.Printf("  ✗ FAIL: Expected '%s', got '%s'\n", tc.ExpectedCategory, category)
			}
		}
	}

	// Print summary
	accuracy := float64(successCount) / float64(len(testCases)) * 100
	fmt.Printf("\n=== Keyword Routing Test Summary ===\n")
	fmt.Printf("Total tests: %d\n", len(testCases))
	fmt.Printf("Passed: %d\n", successCount)
	fmt.Printf("Failed: %d\n", len(testCases)-successCount)
	fmt.Printf("Accuracy: %.2f%%\n", accuracy)

	// Print failures if any
	if successCount < len(testCases) {
		fmt.Println("\n=== Failures ===")
		for _, result := range results {
			if !result.Correct {
				fmt.Printf("Query: %s\n", result.Query)
				fmt.Printf("  Expected: %s\n", result.ExpectedCategory)
				fmt.Printf("  Got: %s\n", result.ActualCategory)
				if result.Error != "" {
					fmt.Printf("  Error: %s\n", result.Error)
				}
			}
		}
	}

	// Require at least 80% accuracy
	if accuracy < 80.0 {
		return fmt.Errorf("keyword routing accuracy %.2f%% is below threshold of 80%%", accuracy)
	}

	return nil
}

// classifyKeywordQuery sends a classification request to the router
func classifyKeywordQuery(ctx context.Context, localPort string, query string) (string, error) {
	// Create HTTP request payload
	payload := map[string]interface{}{
		"model": "gpt-3.5-turbo",
		"messages": []map[string]string{
			{
				"role":    "user",
				"content": query,
			},
		},
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	// Make HTTP request
	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	// Extract classification from response headers
	category := resp.Header.Get("X-VSR-Category")
	if category == "" {
		// If no category header, check if it's an error response
		if resp.StatusCode != http.StatusOK {
			return "", fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
		}
		// No category means no match (empty category)
		return "", nil
	}

	return category, nil
}

// loadKeywordRoutingCases loads test cases from JSON file
func loadKeywordRoutingCases(filepath string) ([]KeywordRoutingCase, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read test data file %s: %w", filepath, err)
	}

	var cases []KeywordRoutingCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, fmt.Errorf("failed to parse keyword routing test cases: %w", err)
	}

	return cases, nil
}
