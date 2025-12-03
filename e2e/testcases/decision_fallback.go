package testcases

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

//go:embed testdata/decision_fallback_cases.json
var decisionFallbackCasesJSON []byte

func init() {
	pkgtestcases.Register("decision-fallback-behavior", pkgtestcases.TestCase{
		Description: "Test decision fallback behavior when no specific decision matches",
		Tags:        []string{"signal-decision", "fallback", "routing"},
		Fn:          testDecisionFallback,
	})
}

// DecisionFallbackCase represents a test case for decision fallback
type DecisionFallbackCase struct {
	Query            string `json:"query"`
	ExpectedDecision string `json:"expected_decision"`
	ShouldFallback   bool   `json:"should_fallback"`
	Description      string `json:"description"`
}

// DecisionFallbackTestData holds the test cases loaded from JSON
type DecisionFallbackTestData struct {
	Description string                 `json:"description"`
	TestCases   []DecisionFallbackCase `json:"test_cases"`
}

// DecisionFallbackResult tracks the result of a single fallback test
type DecisionFallbackResult struct {
	Query            string
	ExpectedDecision string
	ActualDecision   string
	ShouldFallback   bool
	DidFallback      bool
	Correct          bool
	Error            string
}

func testDecisionFallback(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing decision fallback behavior")
	}

	// Setup service connection and get local port
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// Load test cases from embedded JSON
	var testData DecisionFallbackTestData
	if err := json.Unmarshal(decisionFallbackCasesJSON, &testData); err != nil {
		return fmt.Errorf("failed to parse decision fallback test cases: %w", err)
	}
	testCases := testData.TestCases

	if opts.Verbose {
		fmt.Printf("[Test] Loaded %d test cases from testdata/decision_fallback_cases.json\n", len(testCases))
	}

	// Run fallback tests
	var results []DecisionFallbackResult
	totalTests := 0
	correctTests := 0

	for _, testCase := range testCases {
		totalTests++
		result := testSingleFallback(ctx, testCase, localPort, opts.Verbose)
		results = append(results, result)
		if result.Correct {
			correctTests++
		}

		// Small delay between tests
		time.Sleep(500 * time.Millisecond)
	}

	// Calculate accuracy
	accuracy := float64(correctTests) / float64(totalTests) * 100

	// Set details for reporting
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_tests":   totalTests,
			"correct_tests": correctTests,
			"accuracy_rate": fmt.Sprintf("%.2f%%", accuracy),
			"failed_tests":  totalTests - correctTests,
		})
	}

	// Print results
	printDecisionFallbackResults(results, totalTests, correctTests, accuracy)

	if opts.Verbose {
		fmt.Printf("[Test] Decision fallback test completed: %d/%d correct (%.2f%% accuracy)\n",
			correctTests, totalTests, accuracy)
	}

	// Return error if accuracy is below threshold
	if correctTests == 0 {
		return fmt.Errorf("decision fallback test failed: 0%% accuracy (0/%d correct)", totalTests)
	}

	return nil
}

func testSingleFallback(ctx context.Context, testCase DecisionFallbackCase, localPort string, verbose bool) DecisionFallbackResult {
	result := DecisionFallbackResult{
		Query:            testCase.Query,
		ExpectedDecision: testCase.ExpectedDecision,
		ShouldFallback:   testCase.ShouldFallback,
	}

	// Create chat completion request
	requestBody := map[string]interface{}{
		"model": "MoM", // Use Mixture of Models to trigger decision engine
		"messages": []map[string]string{
			{"role": "user", "content": testCase.Query},
		},
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		result.Error = fmt.Sprintf("failed to marshal request: %v", err)
		return result
	}

	// Send request
	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		result.Error = fmt.Sprintf("failed to create request: %v", err)
		return result
	}
	req.Header.Set("Content-Type", "application/json")

	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		result.Error = fmt.Sprintf("failed to send request: %v", err)
		return result
	}
	defer resp.Body.Close()

	// Check response status
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		result.Error = fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(bodyBytes))

		if verbose {
			fmt.Printf("[Test] ✗ HTTP %d Error for query: %s\n", resp.StatusCode, testCase.Query)
			fmt.Printf("  Expected decision: %s\n", testCase.ExpectedDecision)
			fmt.Printf("  Response: %s\n", string(bodyBytes))
		}

		return result
	}

	// Extract VSR decision headers
	result.ActualDecision = resp.Header.Get("x-vsr-selected-decision")

	// Determine if fallback occurred
	// "other_decision" or "general_decision" indicates fallback
	result.DidFallback = (result.ActualDecision == "other_decision" ||
		result.ActualDecision == "general_decision")

	// Check if the result matches expectations
	result.Correct = (result.ActualDecision == testCase.ExpectedDecision)

	if verbose {
		if result.Correct {
			fmt.Printf("[Test] ✓ Fallback behavior correct\n")
			fmt.Printf("  Query: %s\n", truncateString(testCase.Query, 60))
			fmt.Printf("  Decision: %s (fallback: %v)\n", result.ActualDecision, result.DidFallback)
		} else {
			fmt.Printf("[Test] ✗ Fallback behavior incorrect\n")
			fmt.Printf("  Query: %s\n", testCase.Query)
			fmt.Printf("  Expected: %s (should fallback: %v)\n", testCase.ExpectedDecision, testCase.ShouldFallback)
			fmt.Printf("  Actual:   %s (did fallback: %v)\n", result.ActualDecision, result.DidFallback)
			fmt.Printf("  Description: %s\n", testCase.Description)
		}
	}

	return result
}

func printDecisionFallbackResults(results []DecisionFallbackResult, totalTests, correctTests int, accuracy float64) {
	separator := "================================================================================"
	fmt.Println("\n" + separator)
	fmt.Println("DECISION FALLBACK BEHAVIOR TEST RESULTS")
	fmt.Println(separator)
	fmt.Printf("Total Tests: %d\n", totalTests)
	fmt.Printf("Correct Behaviors: %d\n", correctTests)
	fmt.Printf("Accuracy Rate: %.2f%%\n", accuracy)
	fmt.Println(separator)

	// Print summary by behavior type
	fallbackTests := 0
	fallbackCorrect := 0
	specificTests := 0
	specificCorrect := 0

	for _, result := range results {
		if result.ShouldFallback {
			fallbackTests++
			if result.Correct {
				fallbackCorrect++
			}
		} else {
			specificTests++
			if result.Correct {
				specificCorrect++
			}
		}
	}

	fmt.Println("\nTest Breakdown:")
	if fallbackTests > 0 {
		fallbackAccuracy := float64(fallbackCorrect) / float64(fallbackTests) * 100
		fmt.Printf("  - Fallback Tests:  %d/%d (%.2f%%)\n", fallbackCorrect, fallbackTests, fallbackAccuracy)
	}
	if specificTests > 0 {
		specificAccuracy := float64(specificCorrect) / float64(specificTests) * 100
		fmt.Printf("  - Specific Tests:  %d/%d (%.2f%%)\n", specificCorrect, specificTests, specificAccuracy)
	}

	// Print failed cases
	failedCount := 0
	for _, result := range results {
		if !result.Correct && result.Error == "" {
			failedCount++
		}
	}

	if failedCount > 0 {
		fmt.Println("\nFailed Fallback Behaviors:")
		for _, result := range results {
			if !result.Correct && result.Error == "" {
				fmt.Printf("  - Query: %s\n", truncateString(result.Query, 70))
				fmt.Printf("    Expected: %s (should fallback: %v)\n", result.ExpectedDecision, result.ShouldFallback)
				fmt.Printf("    Actual:   %s (did fallback: %v)\n", result.ActualDecision, result.DidFallback)
			}
		}
	}

	// Print errors
	errorCount := 0
	for _, result := range results {
		if result.Error != "" {
			errorCount++
		}
	}

	if errorCount > 0 {
		fmt.Println("\nErrors:")
		for _, result := range results {
			if result.Error != "" {
				fmt.Printf("  - Query: %s\n", truncateString(result.Query, 70))
				fmt.Printf("    Error: %s\n", result.Error)
			}
		}
	}

	fmt.Println(separator + "\n")
}
