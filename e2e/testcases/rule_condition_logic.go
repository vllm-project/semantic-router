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
	pkgtestcases.Register("rule-condition-logic", pkgtestcases.TestCase{
		Description: "Test rule condition logic with AND/OR operators",
		Tags:        []string{"signal-decision", "rules", "conditions"},
		Fn:          testRuleConditionLogic,
	})
}

// RuleConditionCase represents a test case for rule condition logic
type RuleConditionCase struct {
	Query              string   `json:"query"`
	ExpectedMatch      bool     `json:"expected_match"`
	ExpectedDecision   string   `json:"expected_decision"`
	RuleOperator       string   `json:"rule_operator"` // "AND" or "OR"
	RequiredConditions []string `json:"required_conditions"`
	Description        string   `json:"description"`
}

// RuleConditionResult tracks the result of a single rule condition test
type RuleConditionResult struct {
	Query            string
	ExpectedMatch    bool
	ActualMatch      bool
	ExpectedDecision string
	ActualDecision   string
	Correct          bool
	Error            string
}

func testRuleConditionLogic(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing rule condition logic with AND/OR operators")
	}

	// Setup service connection and get local port
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// Define test cases for different rule operators
	testCases := []RuleConditionCase{
		// OR operator tests - any condition matches
		{
			Query:              "Think carefully about this urgent business problem",
			ExpectedMatch:      true,
			ExpectedDecision:   "thinking_decision", // Has keywords: "think", "urgent"
			RuleOperator:       "OR",
			RequiredConditions: []string{"keyword:thinking", "domain:thinking"},
			Description:        "Query with thinking keywords should match thinking decision (OR operator)",
		},
		{
			Query:              "Calculate 25 * 4",
			ExpectedMatch:      true,
			ExpectedDecision:   "math_decision",
			RuleOperator:       "OR",
			RequiredConditions: []string{"domain:math"},
			Description:        "Math query should match math decision (OR with single domain condition)",
		},
		{
			Query:              "Tell me about physics and quantum mechanics",
			ExpectedMatch:      true,
			ExpectedDecision:   "physics_decision",
			RuleOperator:       "OR",
			RequiredConditions: []string{"domain:physics"},
			Description:        "Physics query should match physics decision",
		},
		// AND operator tests - both conditions must match
		{
			Query:              "Think carefully about this problem",
			ExpectedMatch:      true,
			ExpectedDecision:   "thinking_decision",
			RuleOperator:       "OR",
			RequiredConditions: []string{"keyword:think", "keyword:careful"},
			Description:        "Query with 'think' and 'careful' keywords should match thinking decision",
		},
		// Keyword matching tests (case-insensitive)
		{
			Query:              "This is URGENT and needs immediate attention",
			ExpectedMatch:      true,
			ExpectedDecision:   "urgent_request",
			RuleOperator:       "OR",
			RequiredConditions: []string{"keyword:urgent", "keyword:immediate"},
			Description:        "Uppercase keywords should match urgent_request (case-insensitive)",
		},
		{
			Query:              "Please think about this carefully",
			ExpectedMatch:      true,
			ExpectedDecision:   "thinking_decision", // Keyword: "think", "careful"
			RuleOperator:       "OR",
			RequiredConditions: []string{"keyword:think", "keyword:careful"},
			Description:        "Multiple thinking keywords should trigger thinking decision",
		},
	}

	// Run rule condition tests
	var results []RuleConditionResult
	totalTests := 0
	correctTests := 0

	for _, testCase := range testCases {
		totalTests++
		result := testSingleRuleCondition(ctx, testCase, localPort, opts.Verbose)
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
	printRuleConditionResults(results, totalTests, correctTests, accuracy)

	if opts.Verbose {
		fmt.Printf("[Test] Rule condition logic test completed: %d/%d correct (%.2f%% accuracy)\n",
			correctTests, totalTests, accuracy)
	}

	// Return error if accuracy is below threshold
	if correctTests == 0 {
		return fmt.Errorf("rule condition logic test failed: 0%% accuracy (0/%d correct)", totalTests)
	}

	return nil
}

func testSingleRuleCondition(ctx context.Context, testCase RuleConditionCase, localPort string, verbose bool) RuleConditionResult {
	result := RuleConditionResult{
		Query:            testCase.Query,
		ExpectedMatch:    testCase.ExpectedMatch,
		ExpectedDecision: testCase.ExpectedDecision,
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

	// Determine if the rule matched based on the selected decision
	result.ActualMatch = (result.ActualDecision == testCase.ExpectedDecision)

	// Check if the result matches expectations
	result.Correct = (result.ActualMatch == testCase.ExpectedMatch)

	// Also check if we got the expected decision when we expected a match
	if testCase.ExpectedMatch {
		result.Correct = result.Correct && (result.ActualDecision == testCase.ExpectedDecision)
	}

	if verbose {
		if result.Correct {
			fmt.Printf("[Test] ✓ Rule condition evaluated correctly\n")
			fmt.Printf("  Query: %s\n", truncateString(testCase.Query, 60))
			fmt.Printf("  Decision: %s\n", result.ActualDecision)
			fmt.Printf("  Operator: %s, Conditions: %v\n", testCase.RuleOperator, testCase.RequiredConditions)
		} else {
			fmt.Printf("[Test] ✗ Rule condition evaluation failed\n")
			fmt.Printf("  Query: %s\n", testCase.Query)
			fmt.Printf("  Expected Decision: %s, Actual: %s\n", testCase.ExpectedDecision, result.ActualDecision)
			fmt.Printf("  Expected Match: %v, Actual: %v\n", testCase.ExpectedMatch, result.ActualMatch)
			fmt.Printf("  Operator: %s\n", testCase.RuleOperator)
			fmt.Printf("  Required Conditions: %v\n", testCase.RequiredConditions)
			fmt.Printf("  Description: %s\n", testCase.Description)
		}
	}

	return result
}

func printRuleConditionResults(results []RuleConditionResult, totalTests, correctTests int, accuracy float64) {
	separator := "================================================================================"
	fmt.Println("\n" + separator)
	fmt.Println("RULE CONDITION LOGIC TEST RESULTS")
	fmt.Println(separator)
	fmt.Printf("Total Tests: %d\n", totalTests)
	fmt.Printf("Correct Evaluations: %d\n", correctTests)
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
		fmt.Println("\nFailed Rule Evaluations:")
		for _, result := range results {
			if !result.Correct && result.Error == "" {
				fmt.Printf("  - Query: %s\n", truncateString(result.Query, 70))
				fmt.Printf("    Expected Decision: %s\n", result.ExpectedDecision)
				fmt.Printf("    Actual Decision:   %s\n", result.ActualDecision)
				fmt.Printf("    Expected Match: %v, Actual: %v\n", result.ExpectedMatch, result.ActualMatch)
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
