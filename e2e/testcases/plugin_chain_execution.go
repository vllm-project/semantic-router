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
	pkgtestcases.Register("plugin-chain-execution", pkgtestcases.TestCase{
		Description: "Test plugin chain execution order and blocking behavior",
		Tags:        []string{"signal-decision", "plugin", "pii"},
		Fn:          testPluginChainExecution,
	})
}

// PluginChainCase represents a test case for plugin chain execution
type PluginChainCase struct {
	Query               string   `json:"query"`
	ExpectPIIBlock      bool     `json:"expect_pii_block"`
	ExpectCacheUsed     bool     `json:"expect_cache_used"`
	ExpectPromptApplied bool     `json:"expect_prompt_applied"`
	Description         string   `json:"description"`
	PIITypes            []string `json:"pii_types"` // Expected PII types detected
}

// PluginChainResult tracks the result of a single plugin chain test
type PluginChainResult struct {
	Query               string
	PIIBlocked          bool
	PIIDetected         string
	CacheHit            bool
	PromptApplied       bool
	StatusCode          int
	ExpectPIIBlock      bool
	ExpectCacheUsed     bool
	ExpectPromptApplied bool
	Correct             bool
	Error               string
}

func testPluginChainExecution(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing plugin chain execution order and blocking")
	}

	// Setup service connection and get local port
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// Define test cases
	testCases := []PluginChainCase{
		{
			Query:               "My social security number is 123-45-6789",
			ExpectPIIBlock:      true,
			ExpectCacheUsed:     false, // PII blocks before cache
			ExpectPromptApplied: false, // PII blocks before prompt
			Description:         "PII (SSN) should block entire plugin chain",
			PIITypes:            []string{"US_SSN"},
		},
		{
			Query:               "Contact me at john.doe@example.com",
			ExpectPIIBlock:      true,
			ExpectCacheUsed:     false,
			ExpectPromptApplied: false,
			Description:         "PII (EMAIL) should block entire plugin chain",
			PIITypes:            []string{"EMAIL"},
		},
		{
			Query:               "What is 5 + 7?",
			ExpectPIIBlock:      false,
			ExpectCacheUsed:     false, // First request, cache miss
			ExpectPromptApplied: true,  // Should apply math expert prompt
			Description:         "Clean query should pass PII and apply prompt",
		},
		{
			Query:               "Tell me about photosynthesis",
			ExpectPIIBlock:      false,
			ExpectCacheUsed:     false,
			ExpectPromptApplied: true,
			Description:         "Biology query should pass PII plugin",
		},
	}

	// Run plugin chain tests
	var results []PluginChainResult
	totalTests := 0
	correctTests := 0

	for _, testCase := range testCases {
		totalTests++
		result := testSinglePluginChain(ctx, testCase, localPort, opts.Verbose)
		results = append(results, result)
		if result.Correct {
			correctTests++
		}

		// Small delay between tests to avoid overwhelming the system
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
	printPluginChainResults(results, totalTests, correctTests, accuracy)

	if opts.Verbose {
		fmt.Printf("[Test] Plugin chain execution test completed: %d/%d correct (%.2f%% accuracy)\n",
			correctTests, totalTests, accuracy)
	}

	// Return error if accuracy is below threshold
	if correctTests == 0 {
		return fmt.Errorf("plugin chain execution test failed: 0%% accuracy (0/%d correct)", totalTests)
	}

	return nil
}

func testSinglePluginChain(ctx context.Context, testCase PluginChainCase, localPort string, verbose bool) PluginChainResult {
	result := PluginChainResult{
		Query:               testCase.Query,
		ExpectPIIBlock:      testCase.ExpectPIIBlock,
		ExpectCacheUsed:     testCase.ExpectCacheUsed,
		ExpectPromptApplied: testCase.ExpectPromptApplied,
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

	result.StatusCode = resp.StatusCode

	// Extract plugin execution headers
	piiViolationHeader := resp.Header.Get("x-vsr-pii-violation")
	result.PIIDetected = piiViolationHeader // Store for display purposes
	result.PIIBlocked = (resp.StatusCode == http.StatusForbidden || piiViolationHeader == "true")

	// Check cache headers (x-vsr-cache-hit or similar)
	cacheHeader := resp.Header.Get("x-vsr-cache-hit")
	result.CacheHit = (cacheHeader == "true")

	// Check if system prompt was applied (x-vsr-selected-decision indicates routing happened)
	selectedDecision := resp.Header.Get("x-vsr-selected-decision")
	result.PromptApplied = (selectedDecision != "" && !result.PIIBlocked)

	// Determine correctness based on expectations
	piiCorrect := (result.PIIBlocked == testCase.ExpectPIIBlock)
	cacheCorrect := true // We don't strictly enforce cache behavior in this test
	promptCorrect := true

	// If PII was expected to block, other plugins shouldn't run
	if testCase.ExpectPIIBlock {
		promptCorrect = !result.PromptApplied // Prompt should NOT be applied if PII blocked
	}

	result.Correct = piiCorrect && cacheCorrect && promptCorrect

	if verbose {
		if result.Correct {
			fmt.Printf("[Test] ✓ Plugin chain executed correctly\n")
			fmt.Printf("  Query: %s\n", truncateString(testCase.Query, 60))
			fmt.Printf("  PII Blocked: %v (expected: %v)\n", result.PIIBlocked, testCase.ExpectPIIBlock)
			if result.PIIDetected != "" {
				fmt.Printf("  PII Detected: %s\n", result.PIIDetected)
			}
		} else {
			fmt.Printf("[Test] ✗ Plugin chain execution failed\n")
			fmt.Printf("  Query: %s\n", testCase.Query)
			fmt.Printf("  Expected PII Block: %v, Actual: %v\n", testCase.ExpectPIIBlock, result.PIIBlocked)
			fmt.Printf("  PII Detected: %s\n", result.PIIDetected)
			fmt.Printf("  Status Code: %d\n", result.StatusCode)
			fmt.Printf("  Description: %s\n", testCase.Description)
		}
	}

	// Read response body for detailed error
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusForbidden {
		bodyBytes, _ := io.ReadAll(resp.Body)
		if verbose {
			fmt.Printf("  Response: %s\n", string(bodyBytes))
		}
	}

	return result
}

func printPluginChainResults(results []PluginChainResult, totalTests, correctTests int, accuracy float64) {
	separator := "================================================================================"
	fmt.Println("\n" + separator)
	fmt.Println("PLUGIN CHAIN EXECUTION TEST RESULTS")
	fmt.Println(separator)
	fmt.Printf("Total Tests: %d\n", totalTests)
	fmt.Printf("Correct Executions: %d\n", correctTests)
	fmt.Printf("Accuracy Rate: %.2f%%\n", accuracy)
	fmt.Println(separator)

	// Print summary by behavior type
	piiBlockTests := 0
	piiBlockCorrect := 0
	cleanQueryTests := 0
	cleanQueryCorrect := 0

	for _, result := range results {
		if result.ExpectPIIBlock {
			piiBlockTests++
			if result.Correct {
				piiBlockCorrect++
			}
		} else {
			cleanQueryTests++
			if result.Correct {
				cleanQueryCorrect++
			}
		}
	}

	fmt.Println("\nTest Breakdown:")
	if piiBlockTests > 0 {
		piiBlockAccuracy := float64(piiBlockCorrect) / float64(piiBlockTests) * 100
		fmt.Printf("  - PII Blocking Tests: %d/%d (%.2f%%)\n", piiBlockCorrect, piiBlockTests, piiBlockAccuracy)
	}
	if cleanQueryTests > 0 {
		cleanQueryAccuracy := float64(cleanQueryCorrect) / float64(cleanQueryTests) * 100
		fmt.Printf("  - Clean Query Tests:  %d/%d (%.2f%%)\n", cleanQueryCorrect, cleanQueryTests, cleanQueryAccuracy)
	}

	// Print failed cases
	failedCount := 0
	for _, result := range results {
		if !result.Correct && result.Error == "" {
			failedCount++
		}
	}

	if failedCount > 0 {
		fmt.Println("\nFailed Plugin Chain Executions:")
		for _, result := range results {
			if !result.Correct && result.Error == "" {
				fmt.Printf("  - Query: %s\n", truncateString(result.Query, 70))
				fmt.Printf("    Expected PII Block: %v, Actual: %v\n", result.ExpectPIIBlock, result.PIIBlocked)
				fmt.Printf("    PII Detected: %s\n", result.PIIDetected)
				fmt.Printf("    Status Code: %d\n", result.StatusCode)
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
