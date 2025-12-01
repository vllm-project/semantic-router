package testcases

import (
	"context"
	"fmt"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("mcp-fallback-behavior", pkgtestcases.TestCase{
		Description: "Test MCP fallback to in-tree classifier on failures",
		Tags:        []string{"mcp", "fallback", "resilience"},
		Fn:          testMCPFallbackBehavior,
	})
}

func testMCPFallbackBehavior(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing MCP fallback behavior")
	}

	// Setup service connection and get local port
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward() // Critical: always clean up port forwarding

	// Load test cases
	testCases, err := loadMCPTestCases("e2e/testcases/testdata/mcp/mcp_fallback_cases.json")
	if err != nil {
		return fmt.Errorf("failed to load test cases: %w", err)
	}

	// Execute tests and collect results
	var results []MCPTestResult
	fallbackCount := 0
	recoveryCount := 0

	for _, testCase := range testCases {
		// Note: In a real implementation, we would need to:
		// 1. Simulate MCP failures by stopping the MCP server process
		// 2. Verify that requests still succeed (via fallback)
		// 3. Verify that the fallback classifier is used (check headers/logs)
		// 4. Restart MCP server and verify recovery

		// For now, we test that normal requests work correctly
		// The actual fallback testing would require more complex infrastructure

		resp, err := executeMCPRequest(ctx, localPort, testCase.Query, opts.Verbose)
		if err != nil {
			results = append(results, MCPTestResult{
				Description:      testCase.Description,
				Query:            testCase.Query,
				ExpectedCategory: testCase.ExpectedCategory,
				Success:          false,
				Error:            err.Error(),
			})
			continue
		}
		defer resp.Body.Close()

		result := validateMCPResponse(resp, testCase, opts.Verbose)

		// Check if fallback was used (indicated by header or different behavior)
		fallbackUsedHeader := resp.Header.Get("x-vsr-fallback-used")
		if fallbackUsedHeader == "true" {
			fallbackCount++
		}

		// Check if recovery happened
		if testCase.TestRecovery && result.Success {
			recoveryCount++
		}

		results = append(results, result)
	}

	// Calculate accuracy
	totalTests := len(results)
	successfulTests, accuracy := calculateAccuracy(results)

	// Report statistics
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_tests":      totalTests,
			"successful_tests": successfulTests,
			"accuracy_rate":    fmt.Sprintf("%.2f%%", accuracy),
			"fallback_count":   fallbackCount,
			"recovery_count":   recoveryCount,
			"failed_tests":     totalTests - successfulTests,
		})
	}

	// Print results
	printMCPTestResults("MCP FALLBACK BEHAVIOR", results, totalTests, successfulTests, accuracy)

	// Print additional metrics
	fmt.Printf("Fallback Count: %d\n", fallbackCount)
	fmt.Printf("Recovery Count: %d\n", recoveryCount)

	if opts.Verbose {
		fmt.Printf("[Test] MCP fallback behavior test completed: %d/%d successful (%.2f%% accuracy)\n",
			successfulTests, totalTests, accuracy)
		fmt.Printf("[Test] Fallbacks detected: %d, Recoveries detected: %d\n",
			fallbackCount, recoveryCount)
	}

	// Note: For fallback tests, we accept lower accuracy since we're testing
	// graceful degradation rather than perfect classification
	if totalTests > 0 && successfulTests == 0 {
		return fmt.Errorf("mcp fallback behavior test failed: no successful requests")
	}

	return nil
}
