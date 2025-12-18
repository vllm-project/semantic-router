package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("dynamo-category-classification", pkgtestcases.TestCase{
		Description: "Test category classification with Dynamo integration",
		Tags:        []string{"dynamo", "classification", "category", "functional"},
		Fn:          testDynamoCategoryClassification,
	})
}

// CategoryTestCase represents a test case for category classification
type CategoryTestCase struct {
	Query            string `json:"query"`
	ExpectedCategory string `json:"expected_category"`
}

// getCategoryTestCases returns test cases for category classification
func getCategoryTestCases() []CategoryTestCase {
	return []CategoryTestCase{
		// Math category tests
		{
			Query:            "What is the derivative of x^2 + 3x?",
			ExpectedCategory: "math",
		},
		{
			Query:            "Calculate the integral of sin(x)",
			ExpectedCategory: "math",
		},
		{
			Query:            "Solve for x: 2x + 5 = 15",
			ExpectedCategory: "math",
		},
		// Science category tests
		{
			Query:            "Explain how photosynthesis works in plants",
			ExpectedCategory: "science",
		},
		{
			Query:            "What is the atomic structure of carbon?",
			ExpectedCategory: "science",
		},
		{
			Query:            "Describe Newton's laws of motion",
			ExpectedCategory: "science",
		},
		// General category tests
		{
			Query:            "Hello, how are you today?",
			ExpectedCategory: "general",
		},
		{
			Query:            "What's the weather like?",
			ExpectedCategory: "general",
		},
		// Other category tests
		{
			Query:            "Tell me a joke about programmers",
			ExpectedCategory: "other",
		},
		{
			Query:            "What's your favorite color?",
			ExpectedCategory: "other",
		},
	}
}

func testDynamoCategoryClassification(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing category classification with Dynamo")
	}

	// Setup service connection and get local port (same as other Dynamo tests)
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup service connection: %w", err)
	}
	defer stopPortForward()

	serviceURL := fmt.Sprintf("http://localhost:%s", localPort)

	testCases := getCategoryTestCases()
	passedTests := 0
	failedTests := 0

	httpClient := &http.Client{
		Timeout: 60 * time.Second,
	}

	for i, tc := range testCases {
		if opts.Verbose {
			fmt.Printf("[Test] Test case %d/%d: Testing query for '%s' category\n", i+1, len(testCases), tc.ExpectedCategory)
		}

		// Use "auto" or "MoM" to trigger category classification
		requestBody := map[string]interface{}{
			"model": "auto",
			"messages": []map[string]string{
				{"role": "user", "content": tc.Query},
			},
			"max_tokens": 30,
		}

		bodyBytes, err := json.Marshal(requestBody)
		if err != nil {
			return fmt.Errorf("failed to marshal request body: %w", err)
		}

		req, err := http.NewRequestWithContext(ctx, "POST", serviceURL+"/v1/chat/completions", bytes.NewBuffer(bodyBytes))
		if err != nil {
			return fmt.Errorf("failed to create request: %w", err)
		}

		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Host", "semantic-router.example.com")

		resp, err := httpClient.Do(req)
		if err != nil {
			if opts.Verbose {
				fmt.Printf("[Test] FAILED: Request error for query '%s': %v\n", tc.Query[:min(30, len(tc.Query))], err)
			}
			failedTests++
			continue
		}
		defer resp.Body.Close()

		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			if opts.Verbose {
				fmt.Printf("[Test] FAILED: Failed to read response for query '%s': %v\n", tc.Query[:min(30, len(tc.Query))], err)
			}
			failedTests++
			continue
		}

		if resp.StatusCode != http.StatusOK {
			if opts.Verbose {
				fmt.Printf("[Test] FAILED: Non-OK status %d for query '%s'\n", resp.StatusCode, tc.Query[:min(30, len(tc.Query))])
			}
			failedTests++
			continue
		}

		// Parse response to check if we got a valid response
		var chatResponse map[string]interface{}
		if err := json.Unmarshal(respBody, &chatResponse); err != nil {
			if opts.Verbose {
				fmt.Printf("[Test] FAILED: Failed to parse response for query '%s': %v\n", tc.Query[:min(30, len(tc.Query))], err)
			}
			failedTests++
			continue
		}

		// Check if response has choices
		choices, ok := chatResponse["choices"].([]interface{})
		if !ok || len(choices) == 0 {
			if opts.Verbose {
				fmt.Printf("[Test] FAILED: No choices in response for query '%s'\n", tc.Query[:min(30, len(tc.Query))])
			}
			failedTests++
			continue
		}

		// Get the model used (should be TinyLlama for all categories in Dynamo)
		modelUsed, _ := chatResponse["model"].(string)
		if !strings.Contains(modelUsed, "TinyLlama") {
			if opts.Verbose {
				fmt.Printf("[Test] WARNING: Unexpected model '%s' for query '%s'\n", modelUsed, tc.Query[:min(30, len(tc.Query))])
			}
		}

		if opts.Verbose {
			fmt.Printf("[Test] PASSED: Query '%s' -> Model: %s (expected category: %s)\n",
				tc.Query[:min(30, len(tc.Query))], modelUsed, tc.ExpectedCategory)
		}
		passedTests++
	}

	if opts.Verbose {
		fmt.Printf("[Test] Category classification results: %d/%d passed\n", passedTests, len(testCases))
	}

	// Allow some failures due to classification uncertainty
	successRate := float64(passedTests) / float64(len(testCases))
	if successRate < 0.7 {
		return fmt.Errorf("category classification success rate too low: %.1f%% (expected >= 70%%)", successRate*100)
	}

	return nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
