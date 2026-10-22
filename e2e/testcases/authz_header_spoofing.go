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

	return validateAuthzSpoofingResults(len(testCases), results)
}

func validateAuthzSpoofingResults(expected int, results []AuthzSpoofingResult) error {
	if expected == 0 || len(results) != expected {
		return fmt.Errorf("authz header spoofing completed %d cases, expected %d", len(results), expected)
	}
	for _, result := range results {
		if !result.Correct || result.Error != "" {
			return fmt.Errorf("authz header spoofing %q: model=%q, want %q: %s", result.Description, result.ActualModel, result.ExpectedModel, result.Error)
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

	req, err := createAuthzSpoofingRequest(ctx, testCase, localPort)
	if err != nil {
		result.Error = err.Error()
		return result
	}

	resp, err := sendAuthzRequest(req)
	if err != nil {
		result.Error = err.Error()
		return result
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil && result.Error == "" {
			result.Error = fmt.Sprintf("failed to close response body: %v", closeErr)
		}
	}()

	result.ResponseStatus = resp.StatusCode

	if err := validateAuthzResponse(resp, &result, testCase); err != nil {
		result.Error = err.Error()
		return result
	}

	actualModel := result.ActualModel
	result.Correct = true // validateAuthzResponse requires the exact non-empty model.

	if verbose {
		printAuthzSpoofingTestResult(testCase.Description, result, testCase.ExpectedModel, actualModel)
	}

	return result
}

func createAuthzSpoofingRequest(ctx context.Context, testCase AuthzSpoofingTestCase, localPort string) (*http.Request, error) {
	requestBody := map[string]interface{}{
		"model": "vllm-sr/auto",
		"messages": []map[string]string{
			// The profile routes premium code requests to 14B. A greeting also
			// routes premium users to 7B, hiding a failed header-stripping rule.
			{"role": "user", "content": "Write code to implement a sorting function"},
		},
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-vsr-debug", "true")

	if testCase.UserID != "" {
		req.Header.Set("x-authz-user-id", testCase.UserID)
	}
	if testCase.UserGroups != "" {
		req.Header.Set("x-authz-user-groups", testCase.UserGroups)
	}
	if testCase.ShouldUseJWT && testCase.JWTToken != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", testCase.JWTToken))
	}
	return req, nil
}

func sendAuthzRequest(req *http.Request) (*http.Response, error) {
	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	return resp, nil
}

func validateAuthzResponse(resp *http.Response, result *AuthzSpoofingResult, testCase AuthzSpoofingTestCase) error {
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("unexpected status code: %d, body: %s", resp.StatusCode, string(bodyBytes))
	}

	actualModel, err := extractModelFromResponse(resp)
	if err != nil {
		return fmt.Errorf("failed to extract model: %w", err)
	}
	result.ActualModel = actualModel
	if testCase.ExpectedModel == "" || actualModel != testCase.ExpectedModel {
		return fmt.Errorf("selected model %q, expected %q after stripping client identity", actualModel, testCase.ExpectedModel)
	}
	return nil
}

func extractModelFromResponse(resp *http.Response) (string, error) {
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}

	var responseBody struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(bodyBytes, &responseBody); err != nil {
		return "", fmt.Errorf("failed to parse response: %w", err)
	}

	selected := resp.Header.Get("x-vsr-selected-model")
	if selected != "" && responseBody.Model != "" && selected != responseBody.Model {
		return "", fmt.Errorf("selected model %q disagrees with response model %q", selected, responseBody.Model)
	}
	if responseBody.Model != "" {
		return responseBody.Model, nil
	}
	if selected != "" {
		return selected, nil
	}
	return "", fmt.Errorf("response provides neither a top-level model nor x-vsr-selected-model")
}

func printAuthzSpoofingTestResult(description string, result AuthzSpoofingResult, expectedModel, actualModel string) {
	if result.Correct {
		fmt.Printf("[Test] ✓ Correct: %s\n", description)
		fmt.Printf("  Expected model: %s, Actual model: %s\n", expectedModel, actualModel)
	} else {
		fmt.Printf("[Test] ✗ Incorrect: %s\n", description)
		fmt.Printf("  Expected model: %s, Actual model: %s\n", expectedModel, actualModel)
		if result.Error != "" {
			fmt.Printf("  Error: %s\n", result.Error)
		}
	}
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
