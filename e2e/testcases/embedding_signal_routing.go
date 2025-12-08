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
	pkgtestcases.Register("embedding-signal-routing", pkgtestcases.TestCase{
		Description: "Test IntelligentRoute with EmbeddingSignal for semantic similarity routing",
		Tags:        []string{"signal-decision", "embedding", "routing", "semantic"},
		Fn:          testEmbeddingSignalRouting,
	})
}

// EmbeddingSignalTestCase represents a test case for embedding-based signal routing
type EmbeddingSignalTestCase struct {
	Description      string `json:"description"`
	Query            string `json:"query"`
	SignalName       string `json:"signal_name"`
	ExpectedMatch    bool   `json:"expected_match"`
	ExpectedDecision string `json:"expected_decision"`
	Category         string `json:"category"` // For grouping results
}

// EmbeddingSignalResult tracks the result of a single embedding signal test
type EmbeddingSignalResult struct {
	Description      string
	Query            string
	SignalName       string
	ExpectedMatch    bool
	ExpectedDecision string
	ActualDecision   string
	SignalTriggered  bool
	Correct          bool
	Error            string
	Category         string
}

// testEmbeddingSignalRouting tests IntelligentRoute with EmbeddingSignal configuration
func testEmbeddingSignalRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing IntelligentRoute with EmbeddingSignal routing")
	}

	// Setup service connection
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// Load test cases from JSON file
	testCases, err := loadEmbeddingSignalCases("e2e/testcases/testdata/embedding_signal_cases.json")
	if err != nil {
		return fmt.Errorf("failed to load test cases: %w", err)
	}

	// Run embedding signal routing tests
	results := runEmbeddingSignalTests(ctx, testCases, localPort, opts.Verbose)

	// Calculate metrics
	totalTests := len(results)
	correctTests := countCorrectTests(results)
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

	// Print detailed results
	printEmbeddingSignalResults(results, totalTests, correctTests, accuracy)

	if opts.Verbose {
		fmt.Printf("[Test] Embedding signal routing test completed: %d/%d correct (%.2f%% accuracy)\n",
			correctTests, totalTests, accuracy)
	}

	// Return error if accuracy is 0%
	if correctTests == 0 {
		return fmt.Errorf("embedding signal routing test failed: 0%% accuracy (0/%d correct)", totalTests)
	}

	return nil
}

// loadEmbeddingSignalCases loads test cases from JSON file
func loadEmbeddingSignalCases(filepath string) ([]EmbeddingSignalTestCase, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read test cases file: %w", err)
	}

	var cases []EmbeddingSignalTestCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, fmt.Errorf("failed to parse test cases: %w", err)
	}

	return cases, nil
}

// runEmbeddingSignalTests executes all test cases and collects results
func runEmbeddingSignalTests(ctx context.Context, testCases []EmbeddingSignalTestCase, localPort string, verbose bool) []EmbeddingSignalResult {
	results := make([]EmbeddingSignalResult, 0, len(testCases))

	for _, testCase := range testCases {
		result := testSingleEmbeddingSignal(ctx, testCase, localPort, verbose)
		results = append(results, result)
	}

	return results
}

// testSingleEmbeddingSignal tests a single embedding signal routing case
func testSingleEmbeddingSignal(ctx context.Context, testCase EmbeddingSignalTestCase, localPort string, verbose bool) EmbeddingSignalResult {
	result := EmbeddingSignalResult{
		Description:      testCase.Description,
		Query:            testCase.Query,
		SignalName:       testCase.SignalName,
		ExpectedMatch:    testCase.ExpectedMatch,
		ExpectedDecision: testCase.ExpectedDecision,
		Category:         testCase.Category,
	}

	// Create chat completion request
	requestBody := map[string]interface{}{
		"model": "auto", // Use "auto" to trigger intelligent routing with decision evaluation
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
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
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

	// Parse decision from response headers (check before status code)
	// The decision header is set even for blocked requests
	actualDecision := resp.Header.Get("x-vsr-selected-decision")
	result.ActualDecision = actualDecision

	// Check response status
	// Note: Blocked requests (e.g., PII policy violations) may return non-200 status
	// but still have the decision header set correctly
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		// Don't treat blocked requests as errors - they have valid decisions
		if actualDecision != "" {
			// Decision was made, but request was blocked (this is expected for block_pii, block_security, etc.)
			if verbose {
				fmt.Printf("[Test] Request blocked with status %d, decision=%s\n", resp.StatusCode, actualDecision)
			}
		} else {
			// No decision and non-200 status - this is an actual error
			result.Error = fmt.Sprintf("unexpected status code: %d, body: %s", resp.StatusCode, string(bodyBytes))
			return result
		}
	}

	// Semantic-router doesn't set individual signal headers (x-vsr-signal-*).
	// Instead, we evaluate correctness based solely on whether the decision matches:
	// - If the correct decision was made, the underlying signals worked properly
	// - For signal match detection (for display), infer from decision
	result.SignalTriggered = (actualDecision == testCase.ExpectedDecision)

	// Check if result matches expectation (simply: does actual decision == expected decision?)
	result.Correct = (actualDecision == testCase.ExpectedDecision)

	if verbose {
		printTestResult(result, testCase)
	}

	return result
}

// countCorrectTests counts number of correct test results
func countCorrectTests(results []EmbeddingSignalResult) int {
	correct := 0
	for _, result := range results {
		if result.Correct {
			correct++
		}
	}
	return correct
}

// printTestResult prints result of a single test
func printTestResult(result EmbeddingSignalResult, testCase EmbeddingSignalTestCase) {
	if result.Correct {
		fmt.Printf("[Test] ✓ Correct: %s\n", result.Description)
		fmt.Printf("       Signal: %s, Triggered: %v, Decision: %s\n",
			result.SignalName, result.SignalTriggered, result.ActualDecision)
	} else {
		fmt.Printf("[Test] ✗ Incorrect: %s\n", result.Description)
		fmt.Printf("       Expected: signal_match=%v, decision=%s\n",
			testCase.ExpectedMatch, testCase.ExpectedDecision)
		fmt.Printf("       Actual:   signal_match=%v, decision=%s\n",
			result.SignalTriggered, result.ActualDecision)
		if result.Error != "" {
			fmt.Printf("       Error: %s\n", result.Error)
		}
	}
}

// printEmbeddingSignalResults prints comprehensive test results
func printEmbeddingSignalResults(results []EmbeddingSignalResult, totalTests, correctTests int, accuracy float64) {
	separator := "================================================================================"
	fmt.Println("\n" + separator)
	fmt.Println("EMBEDDING SIGNAL ROUTING TEST RESULTS")
	fmt.Println(separator)
	fmt.Printf("Total Tests: %d\n", totalTests)
	fmt.Printf("Correctly Routed: %d\n", correctTests)
	fmt.Printf("Routing Accuracy: %.2f%%\n", accuracy)
	fmt.Println(separator)

	// Group results by category
	categoryStats := groupResultsByCategory(results)

	// Print per-category results
	fmt.Println("\nPer-Category Results:")
	for category, stats := range categoryStats {
		categoryAccuracy := float64(stats.correct) / float64(stats.total) * 100
		fmt.Printf("  - %-30s: %d/%d correct (%.2f%%)\n",
			category, stats.correct, stats.total, categoryAccuracy)
	}

	// Print failed cases
	printFailedCases(results)

	// Print errors
	printErrorCases(results)

	fmt.Println(separator + "\n")
}

// categoryStats tracks statistics per category
type categoryStats struct {
	total   int
	correct int
}

// groupResultsByCategory groups results by category for analysis
func groupResultsByCategory(results []EmbeddingSignalResult) map[string]categoryStats {
	stats := make(map[string]categoryStats)

	for _, result := range results {
		category := result.Category
		if category == "" {
			category = "Uncategorized"
		}

		s := stats[category]
		s.total++
		if result.Correct {
			s.correct++
		}
		stats[category] = s
	}

	return stats
}

// printFailedCases prints details of failed test cases
func printFailedCases(results []EmbeddingSignalResult) {
	failedCount := 0
	for _, result := range results {
		if !result.Correct && result.Error == "" {
			failedCount++
		}
	}

	if failedCount > 0 {
		fmt.Println("\nFailed Routing Cases:")
		for _, result := range results {
			if !result.Correct && result.Error == "" {
				fmt.Printf("  - %s\n", result.Description)
				fmt.Printf("    Query: %s\n", truncateString(result.Query, 80))
				fmt.Printf("    Expected: signal_match=%v, decision=%s\n",
					result.ExpectedMatch, result.ExpectedDecision)
				fmt.Printf("    Actual:   signal_match=%v, decision=%s\n",
					result.SignalTriggered, result.ActualDecision)
			}
		}
	}
}

// printErrorCases prints details of error cases
func printErrorCases(results []EmbeddingSignalResult) {
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
				fmt.Printf("  - %s\n", result.Description)
				fmt.Printf("    Error: %s\n", result.Error)
			}
		}
	}
}
