package testcases

import (
	"context"
	"fmt"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("mcp-model-reasoning", pkgtestcases.TestCase{
		Description: "Test MCP model recommendation and reasoning decisions",
		Tags:        []string{"mcp", "model", "reasoning"},
		Fn:          testMCPModelReasoning,
	})
}

func testMCPModelReasoning(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing MCP model recommendation and reasoning decisions")
	}

	// Setup service connection and get local port
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward() // Critical: always clean up port forwarding

	// Load test cases
	testCases, err := loadMCPTestCases("e2e/testcases/testdata/mcp/mcp_model_reasoning_cases.json")
	if err != nil {
		return fmt.Errorf("failed to load test cases: %w", err)
	}

	// Execute tests and collect results
	var results []MCPTestResult
	modelRecommendationsFollowed := 0
	reasoningDecisionsCorrect := 0

	for _, testCase := range testCases {
		resp, err := executeMCPRequest(ctx, localPort, testCase.Query, opts.Verbose)
		if err != nil {
			results = append(results, MCPTestResult{
				Description:      testCase.Description,
				Query:            testCase.Query,
				ExpectedCategory: testCase.ExpectedCategory,
				ExpectedModel:    testCase.ExpectedModel,
				Success:          false,
				Error:            err.Error(),
			})
			continue
		}
		defer resp.Body.Close()

		result := validateMCPResponse(resp, testCase, opts.Verbose)
		results = append(results, result)

		// Track model recommendations
		if result.Success && testCase.ExpectedModel != "" && result.ActualModel == testCase.ExpectedModel {
			modelRecommendationsFollowed++
		}

		// Track reasoning decisions
		if result.Success && testCase.ExpectedUseReasoning != nil && result.ActualReasoning != nil {
			if *result.ActualReasoning == *testCase.ExpectedUseReasoning {
				reasoningDecisionsCorrect++
			}
		}
	}

	// Calculate accuracy
	totalTests := len(results)
	successfulTests, accuracy := calculateAccuracy(results)

	// Report statistics
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_tests":                    totalTests,
			"successful_tests":               successfulTests,
			"accuracy_rate":                  fmt.Sprintf("%.2f%%", accuracy),
			"model_recommendations_followed": modelRecommendationsFollowed,
			"reasoning_decisions_correct":    reasoningDecisionsCorrect,
			"failed_tests":                   totalTests - successfulTests,
		})
	}

	// Print results
	printMCPTestResults("MCP MODEL REASONING", results, totalTests, successfulTests, accuracy)

	// Print additional metrics
	fmt.Printf("Model Recommendations Followed: %d\n", modelRecommendationsFollowed)
	fmt.Printf("Reasoning Decisions Correct: %d\n", reasoningDecisionsCorrect)

	if opts.Verbose {
		fmt.Printf("[Test] MCP model reasoning test completed: %d/%d successful (%.2f%% accuracy)\n",
			successfulTests, totalTests, accuracy)
		fmt.Printf("[Test] Model recommendations followed: %d, Reasoning decisions correct: %d\n",
			modelRecommendationsFollowed, reasoningDecisionsCorrect)
	}

	// Return error if accuracy is too low
	if successfulTests == 0 {
		return fmt.Errorf("mcp model reasoning test failed: 0%% accuracy (0/%d successful)", totalTests)
	}

	return nil
}
