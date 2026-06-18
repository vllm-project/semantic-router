package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// MCPTestCase represents a test case for MCP classification
type MCPTestCase struct {
	Description            string  `json:"description"`
	Query                  string  `json:"query"`
	ExpectedCategory       string  `json:"expected_category"`
	ExpectedModel          string  `json:"expected_model,omitempty"`
	ExpectedUseReasoning   *bool   `json:"expected_use_reasoning,omitempty"`
	ExpectedConfidenceMin  float64 `json:"expected_confidence_min,omitempty"`
	ValidateProbabilitySum bool    `json:"validate_probability_sum,omitempty"`
	ValidateNoNegatives    bool    `json:"validate_no_negatives,omitempty"`
	SimulateMCPFailure     bool    `json:"simulate_mcp_failure,omitempty"`
	SimulateMCPTimeout     bool    `json:"simulate_mcp_timeout,omitempty"`
	SimulateMCPError       bool    `json:"simulate_mcp_error,omitempty"`
	VerifyInTreeUsed       bool    `json:"verify_in_tree_used,omitempty"`
	TestRecovery           bool    `json:"test_recovery,omitempty"`
}

// MCPTestResult tracks the result of a single MCP test
type MCPTestResult struct {
	Description       string
	Query             string
	ExpectedCategory  string
	ActualCategory    string
	ExpectedModel     string
	ActualModel       string
	ExpectedReasoning *bool
	ActualReasoning   *bool
	Confidence        float64
	Probabilities     []float64
	Success           bool
	Error             string
}

// loadMCPTestCases loads test cases from a JSON file
func loadMCPTestCases(filepath string) ([]MCPTestCase, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read test cases file: %w", err)
	}

	var cases []MCPTestCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, fmt.Errorf("failed to parse test cases: %w", err)
	}

	return cases, nil
}

// executeMCPRequest sends a chat completion request and returns the response
func executeMCPRequest(ctx context.Context, localPort, query string, verbose bool) (*http.Response, error) {
	// Create chat completion request
	requestBody := map[string]interface{}{
		"model": "MoM",
		"messages": []map[string]string{
			{"role": "user", "content": query},
		},
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Send request
	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}

	return resp, nil
}

// validateMCPResponse validates an MCP classification response
func validateMCPResponse(resp *http.Response, testCase MCPTestCase, verbose bool) MCPTestResult {
	result := MCPTestResult{
		Description:       testCase.Description,
		Query:             testCase.Query,
		ExpectedCategory:  testCase.ExpectedCategory,
		ExpectedModel:     testCase.ExpectedModel,
		ExpectedReasoning: testCase.ExpectedUseReasoning,
		Success:           true,
	}

	// Check response status
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		result.Success = false
		result.Error = fmt.Sprintf("unexpected status code: %d, body: %s", resp.StatusCode, string(bodyBytes))

		if verbose {
			fmt.Printf("[Test] ✗ HTTP %d Error: %s\n", resp.StatusCode, testCase.Description)
			fmt.Printf("  Query: %s\n", testCase.Query)
			fmt.Printf("  Response body: %s\n", string(bodyBytes))
		}

		return result
	}

	// Extract routing headers
	result.ActualCategory = resp.Header.Get("x-vsr-selected-category")
	result.ActualModel = resp.Header.Get("x-vsr-selected-model")

	// Parse reasoning header
	reasoningHeader := resp.Header.Get("x-vsr-selected-reasoning")
	if reasoningHeader != "" {
		reasoningValue := (reasoningHeader == "on")
		result.ActualReasoning = &reasoningValue
	}

	// Validate category
	if result.ActualCategory != testCase.ExpectedCategory {
		result.Success = false
		result.Error = fmt.Sprintf("category mismatch: expected %s, got %s",
			testCase.ExpectedCategory, result.ActualCategory)

		if verbose {
			fmt.Printf("[Test] ✗ Category mismatch: %s\n", testCase.Description)
			fmt.Printf("  Expected: %s, Got: %s\n", testCase.ExpectedCategory, result.ActualCategory)
		}
	}

	// Validate model if specified
	if testCase.ExpectedModel != "" && result.ActualModel != testCase.ExpectedModel {
		result.Success = false
		result.Error = fmt.Sprintf("model mismatch: expected %s, got %s",
			testCase.ExpectedModel, result.ActualModel)

		if verbose {
			fmt.Printf("[Test] ✗ Model mismatch: %s\n", testCase.Description)
			fmt.Printf("  Expected: %s, Got: %s\n", testCase.ExpectedModel, result.ActualModel)
		}
	}

	// Validate reasoning if specified
	if testCase.ExpectedUseReasoning != nil {
		if result.ActualReasoning == nil {
			result.Success = false
			result.Error = "reasoning header not present"
		} else if *result.ActualReasoning != *testCase.ExpectedUseReasoning {
			result.Success = false
			result.Error = fmt.Sprintf("reasoning mismatch: expected %v, got %v",
				*testCase.ExpectedUseReasoning, *result.ActualReasoning)

			if verbose {
				fmt.Printf("[Test] ✗ Reasoning mismatch: %s\n", testCase.Description)
				fmt.Printf("  Expected: %v, Got: %v\n", *testCase.ExpectedUseReasoning, *result.ActualReasoning)
			}
		}
	}

	return result
}

// validateProbabilityDistribution validates probability arrays from MCP response
func validateProbabilityDistribution(probabilities []float64, testCase MCPTestCase) error {
	if testCase.ValidateNoNegatives {
		for i, prob := range probabilities {
			if prob < 0 {
				return fmt.Errorf("negative probability at index %d: %f", i, prob)
			}
		}
	}

	if testCase.ValidateProbabilitySum {
		sum := 0.0
		for _, prob := range probabilities {
			sum += prob
		}

		// Allow small tolerance for floating point arithmetic
		if sum < 0.99 || sum > 1.01 {
			return fmt.Errorf("probability sum out of range: %f (expected ~1.0)", sum)
		}
	}

	return nil
}

// printMCPTestResults prints a summary of MCP test results
func printMCPTestResults(testName string, results []MCPTestResult, totalTests, successfulTests int, accuracy float64) {
	separator := "================================================================================"
	fmt.Println("\n" + separator)
	fmt.Printf("%s TEST RESULTS\n", strings.ToUpper(testName))
	fmt.Println(separator)
	fmt.Printf("Total Tests: %d\n", totalTests)
	fmt.Printf("Successful Tests: %d (%.2f%%)\n", successfulTests, accuracy)
	fmt.Printf("Failed Tests: %d\n", totalTests-successfulTests)
	fmt.Println(separator)

	// Print failed tests
	failureCount := 0
	for _, result := range results {
		if !result.Success {
			failureCount++
		}
	}

	if failureCount > 0 {
		fmt.Println("\nFailed Tests:")
		for _, result := range results {
			if !result.Success {
				fmt.Printf("  - %s\n", result.Description)
				fmt.Printf("    Query: %s\n", result.Query)
				if result.ExpectedCategory != "" {
					fmt.Printf("    Expected Category: %s, Got: %s\n", result.ExpectedCategory, result.ActualCategory)
				}
				if result.ExpectedModel != "" {
					fmt.Printf("    Expected Model: %s, Got: %s\n", result.ExpectedModel, result.ActualModel)
				}
				if result.Error != "" {
					fmt.Printf("    Error: %s\n", result.Error)
				}
			}
		}
	}

	fmt.Println(separator + "\n")
}

// calculateAccuracy calculates the accuracy rate from test results
func calculateAccuracy(results []MCPTestResult) (int, float64) {
	successfulTests := 0
	for _, result := range results {
		if result.Success {
			successfulTests++
		}
	}

	totalTests := len(results)
	accuracy := float64(successfulTests) / float64(totalTests) * 100

	return successfulTests, accuracy
}
