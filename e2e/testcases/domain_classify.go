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

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("domain-classify", pkgtestcases.TestCase{
		Description: "Test domain classification accuracy and verify VSR decision headers",
		Tags:        []string{"ai-gateway", "classification", "routing"},
		Fn:          testDomainClassify,
	})
}

// DomainClassifyCase represents a test case for domain classification
type DomainClassifyCase struct {
	Category string `json:"category"`
	Question string `json:"question"`
}

// ClassificationResult tracks the result of a single classification test
type ClassificationResult struct {
	Question         string
	ExpectedCategory string
	ActualCategory   string
	ActualReasoning  string
	Correct          bool
	Error            string
}

func testDomainClassify(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing domain classification accuracy")
	}

	// Setup service connection and get local port
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward() // Ensure port forwarding is stopped when test completes

	// Load test cases from JSON file
	testCases, err := loadDomainClassifyCases("e2e/testcases/testdata/domain_classify_cases.json")
	if err != nil {
		return fmt.Errorf("failed to load test cases: %w", err)
	}

	// Run classification tests
	var results []ClassificationResult
	totalTests := 0
	correctTests := 0

	for _, testCase := range testCases {
		totalTests++
		result := testSingleClassification(ctx, testCase.Question, testCase.Category, localPort, opts.Verbose)
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
	printDomainClassifyResults(results, totalTests, correctTests, accuracy)

	if opts.Verbose {
		fmt.Printf("[Test] Domain classification test completed: %d/%d correct (%.2f%% accuracy)\n",
			correctTests, totalTests, accuracy)
	}

	return nil
}

func loadDomainClassifyCases(filepath string) ([]DomainClassifyCase, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read test cases file: %w", err)
	}

	var cases []DomainClassifyCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, fmt.Errorf("failed to parse test cases: %w", err)
	}

	return cases, nil
}

func testSingleClassification(ctx context.Context, question, expectedCategory, localPort string, verbose bool) ClassificationResult {
	result := ClassificationResult{
		Question:         question,
		ExpectedCategory: expectedCategory,
	}

	// Create chat completion request
	requestBody := map[string]interface{}{
		"model": "MoM",
		"messages": []map[string]string{
			{"role": "user", "content": question},
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
			fmt.Printf("[Test] ✗ HTTP %d Error for question: %s\n", resp.StatusCode, question)
			fmt.Printf("  Expected category: %s\n", expectedCategory)
			fmt.Printf("  Response Headers:\n%s", formatResponseHeaders(resp.Header))
			fmt.Printf("  Response Body: %s\n", string(bodyBytes))
		}

		return result
	}

	// Extract VSR headers
	result.ActualCategory = resp.Header.Get("x-vsr-selected-category")
	result.ActualReasoning = resp.Header.Get("x-vsr-selected-reasoning")

	// Check if classification is correct (only compare category, not reasoning)
	result.Correct = (result.ActualCategory == expectedCategory)

	if verbose && !result.Correct {
		fmt.Printf("[Test] Misclassification: question='%s', expected=%s, actual=%s (reasoning: %s)\n",
			question, expectedCategory, result.ActualCategory, result.ActualReasoning)
	}

	return result
}

func printDomainClassifyResults(results []ClassificationResult, totalTests, correctTests int, accuracy float64) {
	separator := "================================================================================"
	fmt.Println("\n" + separator)
	fmt.Println("DOMAIN CLASSIFICATION TEST RESULTS")
	fmt.Println(separator)
	fmt.Printf("Total Tests: %d\n", totalTests)
	fmt.Printf("Correct Classifications: %d\n", correctTests)
	fmt.Printf("Accuracy Rate: %.2f%%\n", accuracy)
	fmt.Println(separator)

	// Group results by category
	categoryStats := make(map[string]struct {
		total   int
		correct int
	})

	for _, result := range results {
		stats := categoryStats[result.ExpectedCategory]
		stats.total++
		if result.Correct {
			stats.correct++
		}
		categoryStats[result.ExpectedCategory] = stats
	}

	// Print per-category results
	fmt.Println("\nPer-Category Results:")
	for category, stats := range categoryStats {
		categoryAccuracy := float64(stats.correct) / float64(stats.total) * 100
		fmt.Printf("  - %-20s: %d/%d (%.2f%%)\n", category, stats.correct, stats.total, categoryAccuracy)
	}

	// Print failed cases
	failedCount := 0
	for _, result := range results {
		if !result.Correct && result.Error == "" {
			failedCount++
		}
	}

	if failedCount > 0 {
		fmt.Println("\nFailed Classifications:")
		for _, result := range results {
			if !result.Correct && result.Error == "" {
				fmt.Printf("  - Question: %s\n", result.Question)
				fmt.Printf("    Expected Category: %s\n", result.ExpectedCategory)
				fmt.Printf("    Actual Category:   %s\n", result.ActualCategory)
				fmt.Printf("    Actual Reasoning:  %s\n", result.ActualReasoning)
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
				fmt.Printf("  - Question: %s\n", result.Question)
				fmt.Printf("    Error: %s\n", result.Error)
			}
		}
	}

	fmt.Println(separator + "\n")
}
