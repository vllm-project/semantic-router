package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"reflect"
	"strconv"
	"strings"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("keyword-routing", pkgtestcases.TestCase{
		Description: "Test keyword routing accuracy and verify matched keywords",
		Tags:        []string{"ai-gateway", "routing", "keyword"},
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
	Name               string
	Query              string
	ExpectedCategory   string
	ActualCategory     string
	ExpectedKeywords   []string
	ActualKeywords     []string
	ExpectedConfidence float64
	ActualConfidence   float64
	Correct            bool
	KeywordsMatch      bool
	ConfidenceMatch    bool
	Error              string
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
	defer stopPortForward() // Ensure port forwarding is stopped when test completes

	// Load test cases from JSON file
	testCases, err := loadKeywordRoutingCases("e2e/testcases/testdata/keyword_routing_cases.json")
	if err != nil {
		return fmt.Errorf("failed to load test cases: %w", err)
	}

	// Run keyword routing tests
	var results []KeywordRoutingResult
	totalTests := 0
	correctTests := 0
	keywordsCorrect := 0

	for _, testCase := range testCases {
		totalTests++
		result := testSingleKeywordRouting(ctx, testCase, localPort, opts.Verbose)
		results = append(results, result)
		if result.Correct {
			correctTests++
		}
		if result.KeywordsMatch {
			keywordsCorrect++
		}
	}

	// Calculate accuracy
	accuracy := float64(correctTests) / float64(totalTests) * 100
	keywordAccuracy := float64(keywordsCorrect) / float64(totalTests) * 100

	// Set details for reporting
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_tests":      totalTests,
			"correct_tests":    correctTests,
			"accuracy_rate":    fmt.Sprintf("%.2f%%", accuracy),
			"keywords_correct": keywordsCorrect,
			"keyword_accuracy": fmt.Sprintf("%.2f%%", keywordAccuracy),
			"failed_tests":     totalTests - correctTests,
		})
	}

	// Print results
	printKeywordRoutingResults(results, totalTests, correctTests, keywordsCorrect, accuracy, keywordAccuracy)

	if opts.Verbose {
		fmt.Printf("[Test] Keyword routing test completed: %d/%d correct (%.2f%% accuracy), %d/%d keywords matched (%.2f%%)\n",
			correctTests, totalTests, accuracy, keywordsCorrect, totalTests, keywordAccuracy)
	}

	// Return error if accuracy is 0%
	if correctTests == 0 {
		return fmt.Errorf("keyword routing test failed: 0%% accuracy (0/%d correct)", totalTests)
	}

	return nil
}

func loadKeywordRoutingCases(filepath string) ([]KeywordRoutingCase, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read test cases file: %w", err)
	}

	var cases []KeywordRoutingCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, fmt.Errorf("failed to parse test cases: %w", err)
	}

	return cases, nil
}

func testSingleKeywordRouting(ctx context.Context, testCase KeywordRoutingCase, localPort string, verbose bool) KeywordRoutingResult {
	result := KeywordRoutingResult{
		Name:             testCase.Name,
		Query:            testCase.Query,
		ExpectedCategory: testCase.ExpectedCategory,
		ExpectedKeywords: testCase.MatchedKeywords,
	}

	// Create chat completion request
	requestBody := map[string]interface{}{
		"model": "MoM",
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

		// Log detailed error information including headers
		var errorMsg strings.Builder
		errorMsg.WriteString(fmt.Sprintf("Unexpected status code: %d\n", resp.StatusCode))
		errorMsg.WriteString(fmt.Sprintf("Response body: %s\n", string(bodyBytes)))
		errorMsg.WriteString("Response headers:\n")
		errorMsg.WriteString(formatResponseHeaders(resp.Header))

		result.Error = errorMsg.String()

		// Print detailed error to console for debugging
		if verbose {
			fmt.Printf("[Test] ✗ HTTP %d Error for test case: %s\n", resp.StatusCode, testCase.Name)
			fmt.Printf("  Query: %s\n", testCase.Query)
			fmt.Printf("  Expected category: %s\n", testCase.ExpectedCategory)
			fmt.Printf("  Response Headers:\n%s", formatResponseHeaders(resp.Header))
			fmt.Printf("  Response Body: %s\n", string(bodyBytes))
		}

		return result
	}

	// Extract routing headers
	// For keyword routing, use x-vsr-selected-decision since x-vsr-selected-category
	// is only set for domain-based routing. The decision name format is "{category}_decision"
	decision := resp.Header.Get("x-vsr-selected-decision")
	if decision != "" {
		// Extract category from decision name (e.g., "urgent_request_decision" -> "urgent_request")
		result.ActualCategory = strings.TrimSuffix(decision, "_decision")
	} else {
		// Fallback to x-vsr-selected-category for domain-based routing
		result.ActualCategory = resp.Header.Get("x-vsr-selected-category")
	}

	// Parse matched keywords from header (assuming comma-separated)
	keywordsHeader := resp.Header.Get("x-vsr-matched-keywords")
	if keywordsHeader != "" {
		result.ActualKeywords = strings.Split(keywordsHeader, ",")
		// Trim whitespace from each keyword
		for i, kw := range result.ActualKeywords {
			result.ActualKeywords[i] = strings.TrimSpace(kw)
		}
	} else {
		result.ActualKeywords = []string{}
	}

	// Parse confidence from header
	result.ExpectedConfidence = testCase.ExpectedConfidence
	confidenceHeader := resp.Header.Get("x-vsr-selected-confidence")
	if confidenceHeader != "" {
		if parsed, err := strconv.ParseFloat(confidenceHeader, 64); err == nil {
			result.ActualConfidence = parsed
		}
	}

	// Check if category is correct
	result.Correct = (result.ActualCategory == testCase.ExpectedCategory)

	// Check if matched keywords are correct
	// For empty expected keywords, also expect empty actual keywords
	if len(testCase.MatchedKeywords) == 0 && len(result.ActualKeywords) == 0 {
		result.KeywordsMatch = true
	} else {
		// Compare keyword lists (order-independent)
		result.KeywordsMatch = keywordListsMatch(testCase.MatchedKeywords, result.ActualKeywords)
	}

	// Check confidence (allow 0.1 tolerance for floating point comparison)
	if testCase.ExpectedConfidence == 0.0 {
		result.ConfidenceMatch = result.ActualConfidence == 0.0
	} else {
		// For positive confidence, we expect at least 0.5 (minimum match confidence)
		result.ConfidenceMatch = result.ActualConfidence >= 0.5
	}

	if verbose && (!result.Correct || !result.KeywordsMatch) {
		fmt.Printf("[Test] Test case failed: %s\n", testCase.Name)
		if !result.Correct {
			fmt.Printf("  Category mismatch: query='%s', expected=%s, actual=%s\n",
				testCase.Query, testCase.ExpectedCategory, result.ActualCategory)
		}
		if !result.KeywordsMatch {
			fmt.Printf("  Keywords mismatch: expected=%v, actual=%v\n",
				testCase.MatchedKeywords, result.ActualKeywords)
		}
		if !result.ConfidenceMatch {
			fmt.Printf("  Confidence mismatch: expected=%.2f, actual=%.2f\n",
				testCase.ExpectedConfidence, result.ActualConfidence)
		}
	}

	return result
}

// keywordListsMatch checks if two keyword lists match (order-independent)
func keywordListsMatch(expected, actual []string) bool {
	if len(expected) != len(actual) {
		return false
	}

	// Create maps for order-independent comparison
	expectedMap := make(map[string]bool)
	for _, kw := range expected {
		expectedMap[kw] = true
	}

	actualMap := make(map[string]bool)
	for _, kw := range actual {
		actualMap[kw] = true
	}

	return reflect.DeepEqual(expectedMap, actualMap)
}

func printKeywordRoutingResults(results []KeywordRoutingResult, totalTests, correctTests, keywordsCorrect int, accuracy, keywordAccuracy float64) {
	separator := "================================================================================"
	fmt.Println("\n" + separator)
	fmt.Println("KEYWORD ROUTING TEST RESULTS")
	fmt.Println(separator)
	fmt.Printf("Total Tests: %d\n", totalTests)
	fmt.Printf("Correct Categories: %d (%.2f%%)\n", correctTests, accuracy)
	fmt.Printf("Correct Keyword Matches: %d (%.2f%%)\n", keywordsCorrect, keywordAccuracy)
	fmt.Println(separator)

	// Print failed category matches
	categoryFailures := 0
	for _, result := range results {
		if !result.Correct && result.Error == "" {
			categoryFailures++
		}
	}

	if categoryFailures > 0 {
		fmt.Println("\nFailed Category Matches:")
		for _, result := range results {
			if !result.Correct && result.Error == "" {
				fmt.Printf("  - Test: %s\n", result.Name)
				fmt.Printf("    Query: %s\n", result.Query)
				fmt.Printf("    Expected Category: %s\n", result.ExpectedCategory)
				fmt.Printf("    Actual Category:   %s\n", result.ActualCategory)
			}
		}
	}

	// Print failed keyword matches
	keywordFailures := 0
	for _, result := range results {
		if !result.KeywordsMatch && result.Error == "" {
			keywordFailures++
		}
	}

	if keywordFailures > 0 {
		fmt.Println("\nFailed Keyword Matches:")
		for _, result := range results {
			if !result.KeywordsMatch && result.Error == "" {
				fmt.Printf("  - Test: %s\n", result.Name)
				fmt.Printf("    Query: %s\n", result.Query)
				fmt.Printf("    Expected Keywords: %v\n", result.ExpectedKeywords)
				fmt.Printf("    Actual Keywords:   %v\n", result.ActualKeywords)
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
				fmt.Printf("  - Test: %s\n", result.Name)
				fmt.Printf("    Query: %s\n", result.Query)
				fmt.Printf("    Error: %s\n", result.Error)
			}
		}
	}

	fmt.Println(separator + "\n")
}
