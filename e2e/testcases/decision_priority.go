package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("decision-priority-selection", pkgtestcases.TestCase{
		Description: "Test decision priority when multiple decisions match",
		Tags:        []string{"signal-decision", "priority", "routing"},
		Fn:          testDecisionPriority,
	})
}

// DecisionPriorityCase represents a test case for decision priority
type DecisionPriorityCase struct {
	Query             string   `json:"query"`
	ExpectedDecision  string   `json:"expected_decision"`
	ExpectedPriority  int      `json:"expected_priority"`
	MatchingDecisions []string `json:"matching_decisions"` // Decisions that should match
	Description       string   `json:"description"`
}

// DecisionPriorityResult tracks the result of a single priority test
type DecisionPriorityResult struct {
	Query            string
	ExpectedDecision string
	ActualDecision   string
	ExpectedPriority int
	ActualPriority   string
	Correct          bool
	Error            string
}

func testDecisionPriority(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing decision priority selection with multiple matches")
	}

	// Setup service connection and get local port
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// Define test cases inline
	testCases := []DecisionPriorityCase{
		{
			Query:             "Think carefully about this urgent business decision",
			ExpectedDecision:  "thinking_decision",
			ExpectedPriority:  15,
			MatchingDecisions: []string{"thinking_decision", "business_decision"},
			Description:       "Query matches both thinking (priority 15) and business (priority 10) - should select higher priority",
		},
		{
			Query:             "I need to think about complex math problems",
			ExpectedDecision:  "thinking_decision",
			ExpectedPriority:  15,
			MatchingDecisions: []string{"thinking_decision", "math_decision"},
			Description:       "Query matches thinking (priority 15) and math (priority 10) - should select higher priority",
		},
		{
			Query:             "What is 2 + 2?",
			ExpectedDecision:  "math_decision",
			ExpectedPriority:  10,
			MatchingDecisions: []string{"math_decision"},
			Description:       "Simple math query should match math decision (priority 10)",
		},
		{
			Query:             "Tell me about cellular biology",
			ExpectedDecision:  "biology_decision",
			ExpectedPriority:  10,
			MatchingDecisions: []string{"biology_decision"},
			Description:       "Biology query should match biology decision (priority 10)",
		},
	}

	// Run priority tests
	var results []DecisionPriorityResult
	totalTests := 0
	correctTests := 0

	for _, testCase := range testCases {
		totalTests++
		result := testSinglePrioritySelection(ctx, testCase, localPort, opts.Verbose)
		results = append(results, result)
		if result.Correct {
			correctTests++
		}
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
	printDecisionPriorityResults(results, totalTests, correctTests, accuracy)

	if opts.Verbose {
		fmt.Printf("[Test] Decision priority test completed: %d/%d correct (%.2f%% accuracy)\n",
			correctTests, totalTests, accuracy)
	}

	// Return error if accuracy is below threshold
	if correctTests == 0 {
		return fmt.Errorf("decision priority test failed: 0%% accuracy (0/%d correct)", totalTests)
	}

	return nil
}

func testSinglePrioritySelection(ctx context.Context, testCase DecisionPriorityCase, localPort string, verbose bool) DecisionPriorityResult {
	result := DecisionPriorityResult{
		Query:            testCase.Query,
		ExpectedDecision: testCase.ExpectedDecision,
		ExpectedPriority: testCase.ExpectedPriority,
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
			fmt.Printf("  Expected decision: %s (priority %d)\n", testCase.ExpectedDecision, testCase.ExpectedPriority)
			fmt.Printf("  Response: %s\n", string(bodyBytes))
		}

		return result
	}

	// Extract VSR decision headers
	result.ActualDecision = resp.Header.Get("x-vsr-selected-decision")

	// Check if the highest priority decision was selected
	// Note: We validate priority indirectly by checking which decision "wins"
	// when multiple decisions match. The x-vsr-decision-priority header is not
	// currently implemented in semantic-router backend.
	result.Correct = (result.ActualDecision == testCase.ExpectedDecision)

	if verbose {
		if result.Correct {
			fmt.Printf("[Test] ✓ Correct priority selection: %s\n", result.ActualDecision)
		} else {
			fmt.Printf("[Test] ✗ Wrong decision selected\n")
			fmt.Printf("  Query: %s\n", testCase.Query)
			fmt.Printf("  Expected: %s (priority %d)\n", testCase.ExpectedDecision, testCase.ExpectedPriority)
			fmt.Printf("  Actual:   %s\n", result.ActualDecision)
			fmt.Printf("  Description: %s\n", testCase.Description)
		}
	}

	return result
}

func printDecisionPriorityResults(results []DecisionPriorityResult, totalTests, correctTests int, accuracy float64) {
	separator := "================================================================================"
	fmt.Println("\n" + separator)
	fmt.Println("DECISION PRIORITY SELECTION TEST RESULTS")
	fmt.Println(separator)
	fmt.Printf("Total Tests: %d\n", totalTests)
	fmt.Printf("Correct Selections: %d\n", correctTests)
	fmt.Printf("Accuracy Rate: %.2f%%\n", accuracy)
	fmt.Println(separator)

	// Print failed cases
	failedCount := 0
	for _, result := range results {
		if !result.Correct && result.Error == "" {
			failedCount++
		}
	}

	if failedCount > 0 {
		fmt.Println("\nFailed Priority Selections:")
		for _, result := range results {
			if !result.Correct && result.Error == "" {
				fmt.Printf("  - Query: %s\n", result.Query)
				fmt.Printf("    Expected Decision: %s (priority %d)\n", result.ExpectedDecision, result.ExpectedPriority)
				fmt.Printf("    Actual Decision:   %s\n", result.ActualDecision)
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
				fmt.Printf("  - Query: %s\n", result.Query)
				fmt.Printf("    Error: %s\n", result.Error)
			}
		}
	}

	fmt.Println(separator + "\n")
}
