package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("entropy-routing", pkgtestcases.TestCase{
		Description: "Test entropy-based routing decisions and verify uncertainty-aware model selection",
		Tags:        []string{"ai-gateway", "routing", "entropy", "uncertainty"},
		Fn:          testEntropyRouting,
	})
}

// EntropyRoutingCase represents a test case for entropy-based routing
type EntropyRoutingCase struct {
	Name                     string  `json:"name"`
	Description              string  `json:"description"`
	Query                    string  `json:"query"`
	ExpectedUncertaintyLevel string  `json:"expected_uncertainty_level"`
	ExpectedUseReasoning     bool    `json:"expected_use_reasoning"`
	MinConfidence            float64 `json:"min_confidence"`
	MaxConfidence            float64 `json:"max_confidence"`
	ExpectedTopCategory      string  `json:"expected_top_category,omitempty"`
}

// EntropyRoutingResult tracks the result of a single entropy routing test
type EntropyRoutingResult struct {
	Name                string
	Query               string
	ExpectedUncertainty string
	ActualUncertainty   string
	ExpectedReasoning   bool
	ActualReasoning     bool
	ExpectedCategory    string
	ActualCategory      string
	ActualConfidence    float64
	MinConfidence       float64
	MaxConfidence       float64
	UncertaintyMatch    bool
	ReasoningMatch      bool
	ConfidenceInRange   bool
	CategoryMatch       bool
	Error               string
}

func testEntropyRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing entropy-based routing decisions")
	}

	// Setup service connection and get local port
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward() // Ensure port forwarding is stopped when test completes

	// Load test cases from JSON file
	testCases, err := loadEntropyRoutingCases("e2e/testcases/testdata/entropy_routing_cases.json")
	if err != nil {
		return fmt.Errorf("failed to load test cases: %w", err)
	}

	// Run entropy routing tests
	var results []EntropyRoutingResult
	totalTests := 0
	uncertaintyMatches := 0
	reasoningMatches := 0
	confidenceMatches := 0
	categoryMatches := 0

	for _, testCase := range testCases {
		totalTests++
		result := testSingleEntropyRouting(ctx, testCase, localPort, opts.Verbose)
		results = append(results, result)

		if result.UncertaintyMatch {
			uncertaintyMatches++
		}
		if result.ReasoningMatch {
			reasoningMatches++
		}
		if result.ConfidenceInRange {
			confidenceMatches++
		}
		if result.CategoryMatch || result.ExpectedCategory == "" {
			categoryMatches++
		}
	}

	// Calculate accuracy
	uncertaintyAccuracy := float64(uncertaintyMatches) / float64(totalTests) * 100
	reasoningAccuracy := float64(reasoningMatches) / float64(totalTests) * 100
	confidenceAccuracy := float64(confidenceMatches) / float64(totalTests) * 100
	categoryAccuracy := float64(categoryMatches) / float64(totalTests) * 100

	// Set details for reporting
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_tests":          totalTests,
			"uncertainty_matches":  uncertaintyMatches,
			"reasoning_matches":    reasoningMatches,
			"confidence_matches":   confidenceMatches,
			"category_matches":     categoryMatches,
			"uncertainty_accuracy": fmt.Sprintf("%.2f%%", uncertaintyAccuracy),
			"reasoning_accuracy":   fmt.Sprintf("%.2f%%", reasoningAccuracy),
			"confidence_accuracy":  fmt.Sprintf("%.2f%%", confidenceAccuracy),
			"category_accuracy":    fmt.Sprintf("%.2f%%", categoryAccuracy),
		})
	}

	// Print results
	printEntropyRoutingResults(results, totalTests, uncertaintyMatches, reasoningMatches,
		confidenceMatches, categoryMatches, uncertaintyAccuracy, reasoningAccuracy,
		confidenceAccuracy, categoryAccuracy)

	if opts.Verbose {
		fmt.Printf("[Test] Entropy routing test completed:\n")
		fmt.Printf("  Uncertainty: %d/%d (%.2f%%)\n", uncertaintyMatches, totalTests, uncertaintyAccuracy)
		fmt.Printf("  Reasoning:   %d/%d (%.2f%%)\n", reasoningMatches, totalTests, reasoningAccuracy)
		fmt.Printf("  Confidence:  %d/%d (%.2f%%)\n", confidenceMatches, totalTests, confidenceAccuracy)
		fmt.Printf("  Category:    %d/%d (%.2f%%)\n", categoryMatches, totalTests, categoryAccuracy)
	}

	// Return error if any critical metric is 0%
	if reasoningMatches == 0 {
		return fmt.Errorf("entropy routing test failed: 0%% reasoning accuracy (0/%d correct)", totalTests)
	}

	return nil
}

func loadEntropyRoutingCases(filepath string) ([]EntropyRoutingCase, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read test cases file: %w", err)
	}

	var cases []EntropyRoutingCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, fmt.Errorf("failed to parse test cases: %w", err)
	}

	return cases, nil
}

func testSingleEntropyRouting(ctx context.Context, testCase EntropyRoutingCase, localPort string, verbose bool) EntropyRoutingResult {
	result := EntropyRoutingResult{
		Name:                testCase.Name,
		Query:               testCase.Query,
		ExpectedUncertainty: testCase.ExpectedUncertaintyLevel,
		ExpectedReasoning:   testCase.ExpectedUseReasoning,
		ExpectedCategory:    testCase.ExpectedTopCategory,
		MinConfidence:       testCase.MinConfidence,
		MaxConfidence:       testCase.MaxConfidence,
	}

	// Create chat completion request with MoM model to trigger decision engine
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
			fmt.Printf("  Response Headers:\n%s", formatResponseHeaders(resp.Header))
			fmt.Printf("  Response Body: %s\n", string(bodyBytes))
		}

		return result
	}

	// Extract entropy-related headers
	result.ActualUncertainty = resp.Header.Get("x-vsr-uncertainty-level")
	result.ActualCategory = resp.Header.Get("x-vsr-selected-category")

	// Parse reasoning flag
	reasoningHeader := resp.Header.Get("x-vsr-selected-reasoning")
	result.ActualReasoning = (reasoningHeader == "true" || reasoningHeader == "True" || reasoningHeader == "1")

	// Parse confidence
	confidenceHeader := resp.Header.Get("x-vsr-confidence")
	if confidenceHeader != "" {
		if conf, err := strconv.ParseFloat(confidenceHeader, 64); err == nil {
			result.ActualConfidence = conf
		}
	}

	// Check if uncertainty level matches
	result.UncertaintyMatch = (result.ActualUncertainty == testCase.ExpectedUncertaintyLevel)

	// Check if reasoning decision matches
	result.ReasoningMatch = (result.ActualReasoning == testCase.ExpectedUseReasoning)

	// Check if confidence is in expected range
	result.ConfidenceInRange = (result.ActualConfidence >= testCase.MinConfidence &&
		result.ActualConfidence <= testCase.MaxConfidence)

	// Check if category matches (if expected category is specified)
	if testCase.ExpectedTopCategory != "" {
		result.CategoryMatch = (result.ActualCategory == testCase.ExpectedTopCategory)
	} else {
		result.CategoryMatch = true // Skip category check if not specified
	}

	if verbose && (!result.UncertaintyMatch || !result.ReasoningMatch || !result.ConfidenceInRange) {
		fmt.Printf("[Test] Test case failed: %s\n", testCase.Name)
		if !result.UncertaintyMatch {
			fmt.Printf("  Uncertainty mismatch: expected=%s, actual=%s\n",
				testCase.ExpectedUncertaintyLevel, result.ActualUncertainty)
		}
		if !result.ReasoningMatch {
			fmt.Printf("  Reasoning mismatch: expected=%v, actual=%v\n",
				testCase.ExpectedUseReasoning, result.ActualReasoning)
		}
		if !result.ConfidenceInRange {
			fmt.Printf("  Confidence out of range: expected [%.2f, %.2f], actual=%.3f\n",
				testCase.MinConfidence, testCase.MaxConfidence, result.ActualConfidence)
		}
		if !result.CategoryMatch && testCase.ExpectedTopCategory != "" {
			fmt.Printf("  Category mismatch: expected=%s, actual=%s\n",
				testCase.ExpectedTopCategory, result.ActualCategory)
		}
	}

	return result
}

func printEntropyRoutingResults(results []EntropyRoutingResult, totalTests, uncertaintyMatches,
	reasoningMatches, confidenceMatches, categoryMatches int,
	uncertaintyAccuracy, reasoningAccuracy, confidenceAccuracy, categoryAccuracy float64) {

	separator := "================================================================================"
	fmt.Println("\n" + separator)
	fmt.Println("ENTROPY-BASED ROUTING TEST RESULTS")
	fmt.Println(separator)
	fmt.Printf("Total Tests: %d\n", totalTests)
	fmt.Printf("Uncertainty Level Matches: %d (%.2f%%)\n", uncertaintyMatches, uncertaintyAccuracy)
	fmt.Printf("Reasoning Decision Matches: %d (%.2f%%)\n", reasoningMatches, reasoningAccuracy)
	fmt.Printf("Confidence Range Matches: %d (%.2f%%)\n", confidenceMatches, confidenceAccuracy)
	fmt.Printf("Category Matches: %d (%.2f%%)\n", categoryMatches, categoryAccuracy)
	fmt.Println(separator)

	// Print failed uncertainty matches
	uncertaintyFailures := 0
	for _, result := range results {
		if !result.UncertaintyMatch && result.Error == "" {
			uncertaintyFailures++
		}
	}

	if uncertaintyFailures > 0 {
		fmt.Println("\nFailed Uncertainty Matches:")
		for _, result := range results {
			if !result.UncertaintyMatch && result.Error == "" {
				fmt.Printf("  - Test: %s\n", result.Name)
				fmt.Printf("    Query: %s\n", result.Query)
				fmt.Printf("    Expected Uncertainty: %s\n", result.ExpectedUncertainty)
				fmt.Printf("    Actual Uncertainty:   %s\n", result.ActualUncertainty)
			}
		}
	}

	// Print failed reasoning matches
	reasoningFailures := 0
	for _, result := range results {
		if !result.ReasoningMatch && result.Error == "" {
			reasoningFailures++
		}
	}

	if reasoningFailures > 0 {
		fmt.Println("\nFailed Reasoning Matches:")
		for _, result := range results {
			if !result.ReasoningMatch && result.Error == "" {
				fmt.Printf("  - Test: %s\n", result.Name)
				fmt.Printf("    Query: %s\n", result.Query)
				fmt.Printf("    Expected Reasoning: %v\n", result.ExpectedReasoning)
				fmt.Printf("    Actual Reasoning:   %v\n", result.ActualReasoning)
				fmt.Printf("    Uncertainty Level:  %s\n", result.ActualUncertainty)
				fmt.Printf("    Confidence:         %.3f\n", result.ActualConfidence)
			}
		}
	}

	// Print failed confidence ranges
	confidenceFailures := 0
	for _, result := range results {
		if !result.ConfidenceInRange && result.Error == "" {
			confidenceFailures++
		}
	}

	if confidenceFailures > 0 {
		fmt.Println("\nFailed Confidence Ranges:")
		for _, result := range results {
			if !result.ConfidenceInRange && result.Error == "" {
				fmt.Printf("  - Test: %s\n", result.Name)
				fmt.Printf("    Query: %s\n", result.Query)
				fmt.Printf("    Expected Range: [%.2f, %.2f]\n", result.MinConfidence, result.MaxConfidence)
				fmt.Printf("    Actual Confidence: %.3f\n", result.ActualConfidence)
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
