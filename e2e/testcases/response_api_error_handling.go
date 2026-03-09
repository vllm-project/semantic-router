package testcases

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
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

func testResponseAPIErrorMissingInput(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	apiClient := fixtures.NewResponseAPIClient(session, 30*time.Second)
	raw, err := apiClient.CreateRaw(ctx, map[string]interface{}{
		"model": "openai/gpt-oss-20b",
		"messages": []map[string]string{
			{"role": "user", "content": "Hello"},
		},
	})
	if err != nil {
		return err
	}
	if raw.StatusCode != 400 {
		return fmt.Errorf("expected status 400, got %d: %s", raw.StatusCode, string(raw.Body))
	}

	var apiError fixtures.APIErrorResponse
	if err := raw.DecodeJSON(&apiError); err != nil {
		return err
	}
	if apiError.Error.Message == "" {
		return fmt.Errorf("error response missing message")
	}
	if !strings.Contains(strings.ToLower(apiError.Error.Message), "input") {
		return fmt.Errorf("error message should mention 'input' field: %s", apiError.Error.Message)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status_code":   raw.StatusCode,
			"error_message": apiError.Error.Message,
			"error_type":    apiError.Error.Type,
		})
	}
	return nil
}

func testResponseAPIErrorNonexistentPreviousResponseID(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	storeTrue := true
	apiClient := fixtures.NewResponseAPIClient(session, 30*time.Second)
	raw, err := apiClient.CreateRaw(ctx, fixtures.ResponseAPIRequest{
		Model:              "openai/gpt-oss-20b",
		Input:              "Hello, continuing a non-existent conversation",
		PreviousResponseID: "resp_nonexistent_12345",
		Store:              &storeTrue,
		Metadata:           map[string]string{"test": "response-api-error-handling"},
	})
	if err != nil {
		return err
	}

	switch raw.StatusCode {
	case 200:
		var apiResp fixtures.ResponseAPIResponse
		if err := raw.DecodeJSON(&apiResp); err != nil {
			return err
		}
		if opts.SetDetails != nil {
			opts.SetDetails(map[string]interface{}{
				"behavior":    "graceful_degradation",
				"status_code": raw.StatusCode,
				"response_id": apiResp.ID,
			})
		}
		return nil
	case 400, 404:
		var apiError fixtures.APIErrorResponse
		if err := raw.DecodeJSON(&apiError); err != nil {
			return err
		}
		if opts.SetDetails != nil {
			opts.SetDetails(map[string]interface{}{
				"behavior":      "strict_validation",
				"status_code":   raw.StatusCode,
				"error_message": apiError.Error.Message,
			})
		}
		return nil
	default:
		return fmt.Errorf("unexpected status code %d: %s", raw.StatusCode, string(raw.Body))
	}
}

func testResponseAPIErrorNonexistentResponseIDGet(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	apiClient := fixtures.NewResponseAPIClient(session, 30*time.Second)
	nonexistentID := "resp_nonexistent_67890"
	raw, err := apiClient.GetRaw(ctx, nonexistentID)
	if err != nil {
		return err
	}
	if raw.StatusCode != 404 {
		return fmt.Errorf("expected status 404, got %d: %s", raw.StatusCode, string(raw.Body))
	}

	var apiError fixtures.APIErrorResponse
	if err := raw.DecodeJSON(&apiError); err != nil {
		return err
	}
	if apiError.Error.Message == "" {
		return fmt.Errorf("error response missing message")
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status_code":   raw.StatusCode,
			"error_message": apiError.Error.Message,
			"error_type":    apiError.Error.Type,
			"requested_id":  nonexistentID,
		})
	}
	return nil
}

func testResponseAPIErrorNonexistentResponseIDDelete(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	apiClient := fixtures.NewResponseAPIClient(session, 30*time.Second)
	nonexistentID := "resp_nonexistent_abcde"
	raw, err := apiClient.DeleteRaw(ctx, nonexistentID)
	if err != nil {
		return err
	}
	if raw.StatusCode != 404 {
		return fmt.Errorf("expected status 404, got %d: %s", raw.StatusCode, string(raw.Body))
	}

	var apiError fixtures.APIErrorResponse
	if err := raw.DecodeJSON(&apiError); err != nil {
		return err
	}
	if apiError.Error.Message == "" {
		return fmt.Errorf("error response missing message")
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status_code":   raw.StatusCode,
			"error_message": apiError.Error.Message,
			"error_type":    apiError.Error.Type,
			"requested_id":  nonexistentID,
		})
	}
	return nil
}

func testResponseAPIErrorBackendPassthrough(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	storeTrue := true
	apiClient := fixtures.NewResponseAPIClient(session, 30*time.Second)
	raw, err := apiClient.CreateRaw(ctx, fixtures.ResponseAPIRequest{
		Model:    "invalid-model-that-does-not-exist",
		Input:    "This should trigger a backend error",
		Store:    &storeTrue,
		Metadata: map[string]string{"test": "response-api-error-backend-passthrough"},
	})
	if err != nil {
		return err
	}

	var apiError fixtures.APIErrorResponse
	if err := raw.DecodeJSON(&apiError); err == nil && apiError.Error.Message != "" {
		if raw.StatusCode < 400 {
			return fmt.Errorf("error response should have 4xx/5xx status code, got %d", raw.StatusCode)
		}
		if opts.SetDetails != nil {
			opts.SetDetails(map[string]interface{}{
				"behavior":      "error_passthrough",
				"status_code":   raw.StatusCode,
				"error_message": apiError.Error.Message,
				"error_type":    apiError.Error.Type,
			})
		}
		return nil
	}

	var apiResp fixtures.ResponseAPIResponse
	if err := raw.DecodeJSON(&apiResp); err != nil {
		return err
	}
	if raw.StatusCode != 200 {
		return fmt.Errorf("expected status 200 for successful response, got %d", raw.StatusCode)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"behavior":    "model_auto_routed",
			"status_code": raw.StatusCode,
			"response_id": apiResp.ID,
			"model":       apiResp.Model,
		})
	}
	return nil
}
