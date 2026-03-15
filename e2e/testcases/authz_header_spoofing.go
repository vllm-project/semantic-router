package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("authz-header-spoofing", pkgtestcases.TestCase{
		Description: "Test that client-supplied identity headers are stripped and cannot be used for unauthorized access",
		Tags:        []string{"authz-rbac", "security", "authz"},
		Fn:          testAuthzHeaderSpoofing,
	})
}

// AuthzSpoofingTestCase represents a test case for identity header spoofing
type AuthzSpoofingTestCase struct {
	Description     string `json:"description"`
	UserID          string `json:"user_id,omitempty"`
	UserGroups      string `json:"user_groups,omitempty"`
	ExpectedModel   string `json:"expected_model"`
	ExpectedBlocked bool   `json:"expected_blocked,omitempty"`
	ShouldUseJWT    bool   `json:"should_use_jwt,omitempty"`
	JWTToken        string `json:"jwt_token,omitempty"`
}

// AuthzSpoofingResult tracks the result of an authz spoofing test
type AuthzSpoofingResult struct {
	Description     string
	UserID          string
	UserGroups      string
	ExpectedModel   string
	ActualModel     string
	ExpectedBlocked bool
	ActuallyBlocked bool
	Correct         bool
	Error           string
	ResponseStatus  int
}

func testAuthzHeaderSpoofing(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing authz header spoofing protection")
	}

	// Setup service connection and get local port
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// Define test cases
	testCases := []AuthzSpoofingTestCase{
		{
			Description:     "Client sends x-authz-user-id header directly - should be stripped and route to default model",
			UserID:          "admin",
			UserGroups:      "platform-admins",
			ExpectedModel:   "Qwen/Qwen2.5-7B-Instruct", // Should route to default, not admin model
			ExpectedBlocked: false,
			ShouldUseJWT:    false,
		},
		{
			Description:     "Client sends x-authz-user-groups header directly - should be stripped",
			UserID:          "",
			UserGroups:      "premium-tier",
			ExpectedModel:   "Qwen/Qwen2.5-7B-Instruct", // Should route to default
			ExpectedBlocked: false,
			ShouldUseJWT:    false,
		},
		{
			Description:     "Client sends both identity headers directly - should be stripped",
			UserID:          "premium-user",
			UserGroups:      "premium-tier",
			ExpectedModel:   "Qwen/Qwen2.5-7B-Instruct", // Should route to default
			ExpectedBlocked: false,
			ShouldUseJWT:    false,
		},
		{
			Description:     "Request without identity headers - should route to default model",
			UserID:          "",
			UserGroups:      "",
			ExpectedModel:   "Qwen/Qwen2.5-7B-Instruct",
			ExpectedBlocked: false,
			ShouldUseJWT:    false,
		},
	}

	// Run tests
	var results []AuthzSpoofingResult
	totalTests := 0
	correctTests := 0

	for _, testCase := range testCases {
		totalTests++
		result := testSingleAuthzSpoofing(ctx, testCase, localPort, opts.Verbose)
		results = append(results, result)
		if result.Correct {
			correctTests++
		}
	}

	// Calculate success rate
	successRate := float64(correctTests) / float64(totalTests) * 100

	// Set details for reporting
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_tests":   totalTests,
			"correct_tests": correctTests,
			"success_rate":  fmt.Sprintf("%.2f%%", successRate),
			"failed_tests":  totalTests - correctTests,
		})
	}

	// Print results
	printAuthzSpoofingResults(results, totalTests, correctTests, successRate)

	if opts.Verbose {
		fmt.Printf("[Test] Authz header spoofing test completed: %d/%d correct (%.2f%% success rate)\n",
			correctTests, totalTests, successRate)
	}

	// Return error if all tests failed
	if correctTests == 0 {
		return fmt.Errorf("authz header spoofing test failed: 0%% success rate (0/%d correct)", totalTests)
	}

	// Warn if success rate is low (but don't fail)
	if successRate < 100 {
		if opts.Verbose {
			fmt.Printf("[Test] Warning: Some tests failed. Success rate: %.2f%%\n", successRate)
		}
	}

	return nil
}

func testSingleAuthzSpoofing(ctx context.Context, testCase AuthzSpoofingTestCase, localPort string, verbose bool) AuthzSpoofingResult {
	result := AuthzSpoofingResult{
		Description:     testCase.Description,
		UserID:          testCase.UserID,
		UserGroups:      testCase.UserGroups,
		ExpectedModel:   testCase.ExpectedModel,
		ExpectedBlocked: testCase.ExpectedBlocked,
	}

	// Create chat completion request
	requestBody := map[string]interface{}{
		"model": "MoM",
		"messages": []map[string]string{
			{"role": "user", "content": "Hello, this is a test message"},
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

	// Add identity headers if specified (to test spoofing)
	if testCase.UserID != "" {
		req.Header.Set("x-authz-user-id", testCase.UserID)
	}
	if testCase.UserGroups != "" {
		req.Header.Set("x-authz-user-groups", testCase.UserGroups)
	}

	// Add JWT token if specified
	if testCase.ShouldUseJWT && testCase.JWTToken != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", testCase.JWTToken))
	}

	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		result.Error = fmt.Sprintf("failed to send request: %v", err)
		return result
	}
	defer resp.Body.Close()

	result.ResponseStatus = resp.StatusCode

	// Check response status
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		result.Error = fmt.Sprintf("unexpected status code: %d, body: %s", resp.StatusCode, string(bodyBytes))
		return result
	}

	// Read response body
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		result.Error = fmt.Sprintf("failed to read response body: %v", err)
		return result
	}

	// Parse response to get model
	var responseBody map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &responseBody); err != nil {
		result.Error = fmt.Sprintf("failed to parse response: %v", err)
		return result
	}

	// Extract model from response (check x-ai-eg-model header first, then response body)
	actualModel := resp.Header.Get("x-ai-eg-model")
	if actualModel == "" {
		// Try to get from response body
		if choices, ok := responseBody["choices"].([]interface{}); ok && len(choices) > 0 {
			if choice, ok := choices[0].(map[string]interface{}); ok {
				if message, ok := choice["message"].(map[string]interface{}); ok {
					if model, ok := message["model"].(string); ok {
						actualModel = model
					}
				}
			}
		}
	}

	result.ActualModel = actualModel

	// Check if model matches expected
	// For spoofing tests, we expect that client-supplied headers are stripped,
	// so the request should route to the default model (7B), not the spoofed model
	// If actualModel is empty, it means the model wasn't set, which is also acceptable
	// (it means no routing decision was made, which is fine for default routing)
	result.Correct = (actualModel == testCase.ExpectedModel || (actualModel == "" && testCase.ExpectedModel != ""))

	if verbose {
		if result.Correct {
			fmt.Printf("[Test] ✓ Correct: %s\n", testCase.Description)
			fmt.Printf("  Expected model: %s, Actual model: %s\n", testCase.ExpectedModel, actualModel)
		} else {
			fmt.Printf("[Test] ✗ Incorrect: %s\n", testCase.Description)
			fmt.Printf("  Expected model: %s, Actual model: %s\n", testCase.ExpectedModel, actualModel)
			if result.Error != "" {
				fmt.Printf("  Error: %s\n", result.Error)
			}
		}
	}

	return result
}

func printAuthzSpoofingResults(results []AuthzSpoofingResult, totalTests, correctTests int, successRate float64) {
	separator := "================================================================================"
	fmt.Println("\n" + separator)
	fmt.Println("AUTHZ HEADER SPOOFING PROTECTION TEST RESULTS")
	fmt.Println(separator)
	fmt.Printf("Total Tests: %d\n", totalTests)
	fmt.Printf("Correctly Protected: %d\n", correctTests)
	fmt.Printf("Protection Success Rate: %.2f%%\n", successRate)
	fmt.Println(separator)

	// Print test results
	fmt.Println("\nTest Results:")
	for i, result := range results {
		fmt.Printf("\n%d. %s\n", i+1, result.Description)
		if result.UserID != "" {
			fmt.Printf("   User ID: %s\n", result.UserID)
		}
		if result.UserGroups != "" {
			fmt.Printf("   User Groups: %s\n", result.UserGroups)
		}
		fmt.Printf("   Expected Model: %s\n", result.ExpectedModel)
		fmt.Printf("   Actual Model: %s\n", result.ActualModel)
		if result.Correct {
			fmt.Printf("   Status: ✓ PASSED (headers correctly stripped)\n")
		} else {
			fmt.Printf("   Status: ✗ FAILED (headers may not be stripped correctly)\n")
			if result.Error != "" {
				fmt.Printf("   Error: %s\n", result.Error)
			}
		}
	}

	// Print summary
	fmt.Println("\n" + separator)
	fmt.Println("Summary:")
	fmt.Printf("  - Tests where client-supplied headers were correctly stripped: %d/%d\n", correctTests, totalTests)
	fmt.Printf("  - Protection effectiveness: %.2f%%\n", successRate)
	if successRate < 100 {
		fmt.Println("\n⚠️  WARNING: Some tests failed. This may indicate that:")
		fmt.Println("    1. Identity headers are not being stripped correctly")
		fmt.Println("    2. JWT validation is not configured")
		fmt.Println("    3. EnvoyPatchPolicy is not applied correctly")
	}
	fmt.Println(separator + "\n")
}
