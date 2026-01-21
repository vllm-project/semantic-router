package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("response-api-error-missing-input", pkgtestcases.TestCase{
		Description: "Error handling - Invalid request format (missing input field)",
		Tags:        []string{"response-api", "error-handling"},
		Fn:          testResponseAPIErrorMissingInput,
	})
	pkgtestcases.Register("response-api-error-nonexistent-previous-response-id", pkgtestcases.TestCase{
		Description: "Error handling - Non-existent previous_response_id",
		Tags:        []string{"response-api", "error-handling"},
		Fn:          testResponseAPIErrorNonexistentPreviousResponseID,
	})
	pkgtestcases.Register("response-api-error-nonexistent-response-id-get", pkgtestcases.TestCase{
		Description: "Error handling - Non-existent response ID for GET",
		Tags:        []string{"response-api", "error-handling"},
		Fn:          testResponseAPIErrorNonexistentResponseIDGet,
	})
	pkgtestcases.Register("response-api-error-nonexistent-response-id-delete", pkgtestcases.TestCase{
		Description: "Error handling - Non-existent response ID for DELETE",
		Tags:        []string{"response-api", "error-handling"},
		Fn:          testResponseAPIErrorNonexistentResponseIDDelete,
	})
	pkgtestcases.Register("response-api-error-backend-passthrough", pkgtestcases.TestCase{
		Description: "Error handling - Backend error passthrough",
		Tags:        []string{"response-api", "error-handling"},
		Fn:          testResponseAPIErrorBackendPassthrough,
	})
}

// APIErrorResponse represents an error response from the API
type APIErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail contains the error details
type ErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    int    `json:"code,omitempty"`
}

// testResponseAPIErrorMissingInput tests that POST /v1/responses without 'input' field returns 400 error
func testResponseAPIErrorMissingInput(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API error handling: missing input field")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// Create a request without 'input' field (using 'messages' instead like Chat Completions)
	invalidRequest := map[string]interface{}{
		"model": "openai/gpt-oss-20b",
		"messages": []map[string]string{
			{"role": "user", "content": "Hello"},
		},
	}

	jsonData, err := json.Marshal(invalidRequest)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("http://localhost:%s/v1/responses", localPort)
	if opts.Verbose {
		fmt.Printf("[Test] Sending POST request with missing 'input' field to %s\n", url)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response: %w", err)
	}

	if opts.Verbose {
		fmt.Printf("[Test] Response status: %d\n", resp.StatusCode)
		fmt.Printf("[Test] Response body: %s\n", truncateString(string(body), 500))
	}

	// Verify we get a 400 Bad Request
	if resp.StatusCode != http.StatusBadRequest {
		return fmt.Errorf("expected status 400, got %d: %s", resp.StatusCode, string(body))
	}

	// Parse and validate error response
	var apiError APIErrorResponse
	if err := json.Unmarshal(body, &apiError); err != nil {
		return fmt.Errorf("failed to parse error response: %w", err)
	}

	// Verify error message mentions 'input' field
	if apiError.Error.Message == "" {
		return fmt.Errorf("error response missing message")
	}
	if !strings.Contains(strings.ToLower(apiError.Error.Message), "input") {
		return fmt.Errorf("error message should mention 'input' field: %s", apiError.Error.Message)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status_code":   resp.StatusCode,
			"error_message": apiError.Error.Message,
			"error_type":    apiError.Error.Type,
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✅ Missing input field error handled correctly (status=%d, message=%s)\n", resp.StatusCode, apiError.Error.Message)
	}

	return nil
}

// testResponseAPIErrorNonexistentPreviousResponseID tests that using a non-existent previous_response_id
// still allows the request to proceed (with empty history) or returns appropriate error
func testResponseAPIErrorNonexistentPreviousResponseID(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API error handling: non-existent previous_response_id")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// Create a request with a non-existent previous_response_id
	storeTrue := true
	reqBody := ResponseAPIRequest{
		Model:              "openai/gpt-oss-20b",
		Input:              "Hello, continuing a non-existent conversation",
		PreviousResponseID: "resp_nonexistent_12345",
		Store:              &storeTrue,
		Metadata:           map[string]string{"test": "response-api-error-handling"},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("http://localhost:%s/v1/responses", localPort)
	if opts.Verbose {
		fmt.Printf("[Test] Sending POST request with non-existent previous_response_id to %s\n", url)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response: %w", err)
	}

	if opts.Verbose {
		fmt.Printf("[Test] Response status: %d\n", resp.StatusCode)
		fmt.Printf("[Test] Response body: %s\n", truncateString(string(body), 500))
	}

	// The current implementation:
	// - If the response is 200 OK: request succeeded with empty history (graceful degradation)
	// - If the response is 404 Not Found: strict validation of previous_response_id
	// Both behaviors are acceptable depending on the implementation choice

	if resp.StatusCode == http.StatusOK {
		// Graceful degradation: request succeeded without history
		var apiResp ResponseAPIResponse
		if err := json.Unmarshal(body, &apiResp); err != nil {
			return fmt.Errorf("failed to parse response: %w", err)
		}

		if opts.SetDetails != nil {
			opts.SetDetails(map[string]interface{}{
				"behavior":    "graceful_degradation",
				"status_code": resp.StatusCode,
				"response_id": apiResp.ID,
			})
		}

		if opts.Verbose {
			fmt.Printf("[Test] ✅ Non-existent previous_response_id handled gracefully (new response created: %s)\n", apiResp.ID)
		}
	} else if resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusBadRequest {
		// Strict validation: error returned
		var apiError APIErrorResponse
		if err := json.Unmarshal(body, &apiError); err != nil {
			return fmt.Errorf("failed to parse error response: %w", err)
		}

		if opts.SetDetails != nil {
			opts.SetDetails(map[string]interface{}{
				"behavior":      "strict_validation",
				"status_code":   resp.StatusCode,
				"error_message": apiError.Error.Message,
			})
		}

		if opts.Verbose {
			fmt.Printf("[Test] ✅ Non-existent previous_response_id error handled correctly (status=%d, message=%s)\n", resp.StatusCode, apiError.Error.Message)
		}
	} else {
		return fmt.Errorf("unexpected status code %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

// testResponseAPIErrorNonexistentResponseIDGet tests that GET /v1/responses/{id} with non-existent ID returns 404
func testResponseAPIErrorNonexistentResponseIDGet(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API error handling: non-existent response ID for GET")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// Try to get a non-existent response
	nonexistentID := "resp_nonexistent_67890"
	url := fmt.Sprintf("http://localhost:%s/v1/responses/%s", localPort, nonexistentID)
	if opts.Verbose {
		fmt.Printf("[Test] Sending GET request for non-existent response to %s\n", url)
	}

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response: %w", err)
	}

	if opts.Verbose {
		fmt.Printf("[Test] Response status: %d\n", resp.StatusCode)
		fmt.Printf("[Test] Response body: %s\n", string(body))
	}

	// Verify we get a 404 Not Found
	if resp.StatusCode != http.StatusNotFound {
		return fmt.Errorf("expected status 404, got %d: %s", resp.StatusCode, string(body))
	}

	// Parse and validate error response
	var apiError APIErrorResponse
	if err := json.Unmarshal(body, &apiError); err != nil {
		return fmt.Errorf("failed to parse error response: %w", err)
	}

	if apiError.Error.Message == "" {
		return fmt.Errorf("error response missing message")
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status_code":   resp.StatusCode,
			"error_message": apiError.Error.Message,
			"error_type":    apiError.Error.Type,
			"requested_id":  nonexistentID,
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✅ Non-existent response ID GET error handled correctly (status=%d)\n", resp.StatusCode)
	}

	return nil
}

// testResponseAPIErrorNonexistentResponseIDDelete tests that DELETE /v1/responses/{id} with non-existent ID returns 404
func testResponseAPIErrorNonexistentResponseIDDelete(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API error handling: non-existent response ID for DELETE")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// Try to delete a non-existent response
	nonexistentID := "resp_nonexistent_abcde"
	url := fmt.Sprintf("http://localhost:%s/v1/responses/%s", localPort, nonexistentID)
	if opts.Verbose {
		fmt.Printf("[Test] Sending DELETE request for non-existent response to %s\n", url)
	}

	req, err := http.NewRequestWithContext(ctx, "DELETE", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response: %w", err)
	}

	if opts.Verbose {
		fmt.Printf("[Test] Response status: %d\n", resp.StatusCode)
		fmt.Printf("[Test] Response body: %s\n", string(body))
	}

	// Verify we get a 404 Not Found
	if resp.StatusCode != http.StatusNotFound {
		return fmt.Errorf("expected status 404, got %d: %s", resp.StatusCode, string(body))
	}

	// Parse and validate error response
	var apiError APIErrorResponse
	if err := json.Unmarshal(body, &apiError); err != nil {
		return fmt.Errorf("failed to parse error response: %w", err)
	}

	if apiError.Error.Message == "" {
		return fmt.Errorf("error response missing message")
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status_code":   resp.StatusCode,
			"error_message": apiError.Error.Message,
			"error_type":    apiError.Error.Type,
			"requested_id":  nonexistentID,
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✅ Non-existent response ID DELETE error handled correctly (status=%d)\n", resp.StatusCode)
	}

	return nil
}

// testResponseAPIErrorBackendPassthrough tests that backend errors are passed through correctly
func testResponseAPIErrorBackendPassthrough(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API error handling: backend error passthrough")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// Create a request that should trigger a backend error
	// Use an invalid/non-existent model to trigger backend error
	storeTrue := true
	reqBody := ResponseAPIRequest{
		Model:    "invalid-model-that-does-not-exist",
		Input:    "This should trigger a backend error",
		Store:    &storeTrue,
		Metadata: map[string]string{"test": "response-api-error-backend-passthrough"},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("http://localhost:%s/v1/responses", localPort)
	if opts.Verbose {
		fmt.Printf("[Test] Sending POST request with invalid model to %s\n", url)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response: %w", err)
	}

	if opts.Verbose {
		fmt.Printf("[Test] Response status: %d\n", resp.StatusCode)
		fmt.Printf("[Test] Response body: %s\n", truncateString(string(body), 500))
	}

	// We expect either:
	// - A 4xx/5xx error from the backend passed through
	// - Or a successful response if the model gets auto-routed to a valid model

	// Check if this is an error response
	var apiError APIErrorResponse
	if err := json.Unmarshal(body, &apiError); err == nil && apiError.Error.Message != "" {
		// Error response - verify it's properly formatted
		if resp.StatusCode < 400 {
			return fmt.Errorf("error response should have 4xx/5xx status code, got %d", resp.StatusCode)
		}

		if opts.SetDetails != nil {
			opts.SetDetails(map[string]interface{}{
				"behavior":      "error_passthrough",
				"status_code":   resp.StatusCode,
				"error_message": apiError.Error.Message,
				"error_type":    apiError.Error.Type,
			})
		}

		if opts.Verbose {
			fmt.Printf("[Test] ✅ Backend error passed through correctly (status=%d, message=%s)\n", resp.StatusCode, apiError.Error.Message)
		}
	} else {
		// Successful response - the model might have been auto-routed
		var apiResp ResponseAPIResponse
		if err := json.Unmarshal(body, &apiResp); err != nil {
			return fmt.Errorf("failed to parse response: %w", err)
		}

		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("expected status 200 for successful response, got %d", resp.StatusCode)
		}

		if opts.SetDetails != nil {
			opts.SetDetails(map[string]interface{}{
				"behavior":    "model_auto_routed",
				"status_code": resp.StatusCode,
				"response_id": apiResp.ID,
				"model":       apiResp.Model,
			})
		}

		if opts.Verbose {
			fmt.Printf("[Test] ✅ Request succeeded (model may have been auto-routed, id=%s, model=%s)\n", apiResp.ID, apiResp.Model)
		}
	}

	return nil
}
