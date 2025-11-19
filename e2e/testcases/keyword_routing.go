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
	pkgtestcases.Register("keyword-routing", pkgtestcases.TestCase{
		Description: "Test keyword-based routing with case sensitivity and operators",
		Tags:        []string{"signal-decision", "keyword", "routing"},
		Fn:          testKeywordRouting,
	})
}

// KeywordRoutingCase represents a test case for keyword-based routing
type KeywordRoutingCase struct {
	Query            string   `json:"query"`
	ExpectedDecision string   `json:"expected_decision"`
	MatchingKeywords []string `json:"matching_keywords"`
	CaseSensitive    bool     `json:"case_sensitive"`
	Description      string   `json:"description"`
}

// KeywordRoutingResult tracks the result of a single keyword routing test
type KeywordRoutingResult struct {
	Query            string
	ExpectedDecision string
	ActualDecision   string
	MatchingKeywords []string
	Correct          bool
	Error            string
}

func testKeywordRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing keyword-based routing")
	}

	// Setup service connection and get local port
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// Define test cases for keyword routing
	// Based on the keyword_rules configuration: ["urgent", "immediate", "asap", "think", "careful"]
	testCases := []KeywordRoutingCase{
		{
			Query:            "This is URGENT and needs immediate attention",
			ExpectedDecision: "thinking_decision",
			MatchingKeywords: []string{"urgent", "immediate"},
			CaseSensitive:    false,
			Description:      "Uppercase keywords should match (case-insensitive)",
		},
		{
			Query:            "Please think carefully about this problem",
			ExpectedDecision: "thinking_decision",
			MatchingKeywords: []string{"think", "careful"},
			CaseSensitive:    false,
			Description:      "Multiple thinking keywords should trigger thinking decision",
		},
		{
			Query:            "We need this done ASAP",
			ExpectedDecision: "thinking_decision",
			MatchingKeywords: []string{"asap"},
			CaseSensitive:    false,
			Description:      "Single ASAP keyword should trigger thinking decision",
		},
		{
			Query:            "urgent: please think about this immediately",
			ExpectedDecision: "thinking_decision",
			MatchingKeywords: []string{"urgent", "think", "immediately"},
			CaseSensitive:    false,
			Description:      "Multiple keywords in one query should match",
		},
		{
			Query:            "What is 2 + 2?",
			ExpectedDecision: "math_decision",
			MatchingKeywords: []string{},
			CaseSensitive:    false,
			Description:      "Query without thinking keywords should not trigger thinking decision",
		},
		{
			Query:            "I need you to think through this step by step carefully",
			ExpectedDecision: "thinking_decision",
			MatchingKeywords: []string{"think", "careful"},
			CaseSensitive:    false,
			Description:      "Embedded keywords should be detected",
		},
	}

	// Run keyword routing tests
	var results []KeywordRoutingResult
	totalTests := 0
	correctTests := 0

	for _, testCase := range testCases {
		totalTests++
		result := testSingleKeywordRouting(ctx, testCase, localPort, opts.Verbose)
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
	printKeywordRoutingResults(results, totalTests, correctTests, accuracy)

	if opts.Verbose {
		fmt.Printf("[Test] Keyword routing test completed: %d/%d correct (%.2f%% accuracy)\n",
			correctTests, totalTests, accuracy)
	}

	// Return error if accuracy is below threshold
	if correctTests == 0 {
		return fmt.Errorf("keyword routing test failed: 0%% accuracy (0/%d correct)", totalTests)
	}

	return nil
}

func testSingleKeywordRouting(ctx context.Context, testCase KeywordRoutingCase, localPort string, verbose bool) KeywordRoutingResult {
	result := KeywordRoutingResult{
		Query:            testCase.Query,
		ExpectedDecision: testCase.ExpectedDecision,
		MatchingKeywords: testCase.MatchingKeywords,
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

	// Check if the correct decision was selected
	result.Correct = (result.ActualDecision == testCase.ExpectedDecision)

	if verbose {
		if result.Correct {
			fmt.Printf("[Test] ✓ Keyword routing correct\n")
			fmt.Printf("  Query: %s\n", truncateString(testCase.Query, 60))
			fmt.Printf("  Decision: %s\n", result.ActualDecision)
			if len(testCase.MatchingKeywords) > 0 {
				fmt.Printf("  Matching Keywords: %v\n", testCase.MatchingKeywords)
			}
		} else {
			fmt.Printf("[Test] ✗ Keyword routing incorrect\n")
			fmt.Printf("  Query: %s\n", testCase.Query)
			fmt.Printf("  Expected: %s, Actual: %s\n", testCase.ExpectedDecision, result.ActualDecision)
			if len(testCase.MatchingKeywords) > 0 {
				fmt.Printf("  Expected Keywords: %v\n", testCase.MatchingKeywords)
			}
			fmt.Printf("  Description: %s\n", testCase.Description)
		}
	}

	return result
}

func printKeywordRoutingResults(results []KeywordRoutingResult, totalTests, correctTests int, accuracy float64) {
	separator := "================================================================================"
	fmt.Println("\n" + separator)
	fmt.Println("KEYWORD ROUTING TEST RESULTS")
	fmt.Println(separator)
	fmt.Printf("Total Tests: %d\n", totalTests)
	fmt.Printf("Correct Routings: %d\n", correctTests)
	fmt.Printf("Accuracy Rate: %.2f%%\n", accuracy)
	fmt.Println(separator)

	// Print summary by keyword presence
	keywordTests := 0
	keywordCorrect := 0
	noKeywordTests := 0
	noKeywordCorrect := 0

	for _, result := range results {
		if len(result.MatchingKeywords) > 0 {
			keywordTests++
			if result.Correct {
				keywordCorrect++
			}
		} else {
			noKeywordTests++
			if result.Correct {
				noKeywordCorrect++
			}
		}
	}

	fmt.Println("\nTest Breakdown:")
	if keywordTests > 0 {
		keywordAccuracy := float64(keywordCorrect) / float64(keywordTests) * 100
		fmt.Printf("  - With Keywords:    %d/%d (%.2f%%)\n", keywordCorrect, keywordTests, keywordAccuracy)
	}
	if noKeywordTests > 0 {
		noKeywordAccuracy := float64(noKeywordCorrect) / float64(noKeywordTests) * 100
		fmt.Printf("  - Without Keywords: %d/%d (%.2f%%)\n", noKeywordCorrect, noKeywordTests, noKeywordAccuracy)
	}

	// Print failed cases
	failedCount := 0
	for _, result := range results {
		if !result.Correct && result.Error == "" {
			failedCount++
		}
	}

	if failedCount > 0 {
		fmt.Println("\nFailed Keyword Routings:")
		for _, result := range results {
			if !result.Correct && result.Error == "" {
				fmt.Printf("  - Query: %s\n", truncateString(result.Query, 70))
				fmt.Printf("    Expected: %s, Actual: %s\n", result.ExpectedDecision, result.ActualDecision)
				if len(result.MatchingKeywords) > 0 {
					fmt.Printf("    Expected Keywords: %v\n", result.MatchingKeywords)
				}
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
