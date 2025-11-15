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

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("pii-detection", pkgtestcases.TestCase{
		Description: "Test PII detection and blocking functionality",
		Tags:        []string{"ai-gateway", "security", "pii"},
		Fn:          testPIIDetection,
	})
}

// PIITestCase represents a test case for PII detection
type PIITestCase struct {
	Description     string `json:"description"`
	PIIType         string `json:"pii_type"`
	Question        string `json:"question"`
	ExpectedBlocked bool   `json:"expected_blocked"`
}

// PIIResult tracks the result of a PII detection test
type PIIResult struct {
	Description     string
	PIIType         string
	Question        string
	ExpectedBlocked bool
	ActuallyBlocked bool
	Correct         bool
	Error           string
}

func testPIIDetection(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing PII detection functionality")
	}

	// Setup service connection and get local port
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward() // Ensure port forwarding is stopped when test completes

	// Load test cases from JSON file
	testCases, err := loadPIICases("e2e/testcases/testdata/pii_detection_cases.json")
	if err != nil {
		return fmt.Errorf("failed to load test cases: %w", err)
	}

	// Run PII detection tests
	var results []PIIResult
	totalTests := 0
	correctTests := 0

	for _, testCase := range testCases {
		totalTests++
		result := testSinglePIIDetection(ctx, testCase, localPort, opts.Verbose)
		results = append(results, result)
		if result.Correct {
			correctTests++
		}
	}

	// Calculate detection rate and count blocked requests
	detectionRate := float64(correctTests) / float64(totalTests) * 100
	blockedCount := 0
	for _, result := range results {
		if result.ActuallyBlocked {
			blockedCount++
		}
	}

	// Set details for reporting
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_tests":    totalTests,
			"correct_tests":  correctTests,
			"detection_rate": fmt.Sprintf("%.2f%%", detectionRate),
			"blocked_count":  blockedCount,
			"failed_tests":   totalTests - correctTests,
		})
	}

	// Print results
	printPIIResults(results, totalTests, correctTests, detectionRate)

	if opts.Verbose {
		fmt.Printf("[Test] PII detection test completed: %d/%d correct (%.2f%% accuracy)\n",
			correctTests, totalTests, detectionRate)
	}

	return nil
}

func loadPIICases(filepath string) ([]PIITestCase, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read test cases file: %w", err)
	}

	var cases []PIITestCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, fmt.Errorf("failed to parse test cases: %w", err)
	}

	return cases, nil
}

func testSinglePIIDetection(ctx context.Context, testCase PIITestCase, localPort string, verbose bool) PIIResult {
	result := PIIResult{
		Description:     testCase.Description,
		PIIType:         testCase.PIIType,
		Question:        testCase.Question,
		ExpectedBlocked: testCase.ExpectedBlocked,
	}

	// Create chat completion request
	requestBody := map[string]interface{}{
		"model": "MoM",
		"messages": []map[string]string{
			{"role": "user", "content": testCase.Question},
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

	// Check response status (should be 200 OK even when blocked)
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
			fmt.Printf("[Test] ✗ HTTP %d Error for: %s\n", resp.StatusCode, testCase.Description)
			fmt.Printf("  Question: %s\n", testCase.Question)
			fmt.Printf("  Response Headers:\n%s", formatResponseHeaders(resp.Header))
			fmt.Printf("  Response Body: %s\n", string(bodyBytes))
		}

		return result
	}

	// Check for PII violation header
	piiViolationHeader := resp.Header.Get("x-vsr-pii-violation")
	result.ActuallyBlocked = (piiViolationHeader == "true")

	// Verify response body contains expected message
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		result.Error = fmt.Sprintf("failed to read response body: %v", err)
		return result
	}

	if result.ActuallyBlocked {
		// Verify the response contains PII violation message
		bodyStr := string(bodyBytes)
		if !strings.Contains(bodyStr, "personally identifiable information") {
			result.Error = "PII blocked but response message doesn't contain expected text"
		}
	}

	// Check if result matches expectation
	result.Correct = (result.ActuallyBlocked == result.ExpectedBlocked)

	if verbose {
		if result.Correct {
			fmt.Printf("[Test] ✓ Correct: %s (blocked=%v)\n", testCase.Description, result.ActuallyBlocked)
		} else {
			fmt.Printf("[Test] ✗ Incorrect: %s (expected blocked=%v, actual=%v)\n",
				testCase.Description, result.ExpectedBlocked, result.ActuallyBlocked)
		}
	}

	return result
}

func printPIIResults(results []PIIResult, totalTests, correctTests int, blockRate float64) {
	separator := "================================================================================"
	fmt.Println("\n" + separator)
	fmt.Println("PII DETECTION TEST RESULTS")
	fmt.Println(separator)
	fmt.Printf("Total Tests: %d\n", totalTests)
	fmt.Printf("Correctly Detected: %d\n", correctTests)
	fmt.Printf("Detection Accuracy: %.2f%%\n", blockRate)
	fmt.Println(separator)

	// Count blocked vs not blocked
	blockedCount := 0
	for _, result := range results {
		if result.ActuallyBlocked {
			blockedCount++
		}
	}
	fmt.Printf("\nBlocked Requests: %d/%d\n", blockedCount, totalTests)

	// Group results by PII type
	piiTypeStats := make(map[string]struct {
		total   int
		correct int
		blocked int
	})

	for _, result := range results {
		stats := piiTypeStats[result.PIIType]
		stats.total++
		if result.Correct {
			stats.correct++
		}
		if result.ActuallyBlocked {
			stats.blocked++
		}
		piiTypeStats[result.PIIType] = stats
	}

	// Print per-PII-type results
	fmt.Println("\nPer-PII-Type Results:")
	for piiType, stats := range piiTypeStats {
		accuracy := float64(stats.correct) / float64(stats.total) * 100
		fmt.Printf("  - %-20s: %d/%d correct (%.2f%%), %d blocked\n",
			piiType, stats.correct, stats.total, accuracy, stats.blocked)
	}

	// Print failed cases
	failedCount := 0
	for _, result := range results {
		if !result.Correct && result.Error == "" {
			failedCount++
		}
	}

	if failedCount > 0 {
		fmt.Println("\nFailed Detections:")
		for _, result := range results {
			if !result.Correct && result.Error == "" {
				fmt.Printf("  - %s\n", result.Description)
				fmt.Printf("    Question: %s\n", result.Question)
				fmt.Printf("    Expected blocked: %v, Actually blocked: %v\n",
					result.ExpectedBlocked, result.ActuallyBlocked)
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
				fmt.Printf("  - %s\n", result.Description)
				fmt.Printf("    Error: %s\n", result.Error)
			}
		}
	}

	fmt.Println(separator + "\n")
}
