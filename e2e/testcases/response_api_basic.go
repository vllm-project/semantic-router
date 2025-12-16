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
	pkgtestcases.Register("response-api-create", pkgtestcases.TestCase{
		Description: "POST /v1/responses - Create a new response",
		Tags:        []string{"response-api", "functional"},
		Fn:          testResponseAPICreate,
	})
	pkgtestcases.Register("response-api-get", pkgtestcases.TestCase{
		Description: "GET /v1/responses/{id} - Retrieve a response",
		Tags:        []string{"response-api", "functional"},
		Fn:          testResponseAPIGet,
	})
	pkgtestcases.Register("response-api-delete", pkgtestcases.TestCase{
		Description: "DELETE /v1/responses/{id} - Delete a response",
		Tags:        []string{"response-api", "functional"},
		Fn:          testResponseAPIDelete,
	})
	pkgtestcases.Register("response-api-input-items", pkgtestcases.TestCase{
		Description: "GET /v1/responses/{id}/input_items - List input items",
		Tags:        []string{"response-api", "functional"},
		Fn:          testResponseAPIInputItems,
	})
}

// ResponseAPIRequest represents a Response API request
type ResponseAPIRequest struct {
	Model        string            `json:"model"`
	Input        interface{}       `json:"input"`
	Instructions string            `json:"instructions,omitempty"`
	Store        *bool             `json:"store,omitempty"`
	Metadata     map[string]string `json:"metadata,omitempty"`
}

// ResponseAPIResponse represents a Response API response
type ResponseAPIResponse struct {
	ID                 string                   `json:"id"`
	Object             string                   `json:"object"`
	CreatedAt          int64                    `json:"created_at"`
	Model              string                   `json:"model"`
	Status             string                   `json:"status"`
	Output             []map[string]interface{} `json:"output"`
	OutputText         string                   `json:"output_text,omitempty"`
	PreviousResponseID string                   `json:"previous_response_id,omitempty"`
	Usage              map[string]interface{}   `json:"usage,omitempty"`
	Instructions       string                   `json:"instructions,omitempty"`
	Metadata           map[string]string        `json:"metadata,omitempty"`
}

// DeleteResponseResult represents the result of deleting a response
type DeleteResponseResult struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Deleted bool   `json:"deleted"`
}

// InputItemsListResponse represents the response for GET /v1/responses/{id}/input_items
type InputItemsListResponse struct {
	Object  string                   `json:"object"`
	Data    []map[string]interface{} `json:"data"`
	FirstID string                   `json:"first_id"`
	LastID  string                   `json:"last_id"`
	HasMore bool                     `json:"has_more"`
}

// testResponseAPICreate tests POST /v1/responses
func testResponseAPICreate(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API: POST /v1/responses")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// Create a Response API request
	storeTrue := true
	reqBody := ResponseAPIRequest{
		Model:        "MoM",
		Input:        "What is 2 + 2?",
		Instructions: "You are a helpful math assistant.",
		Store:        &storeTrue,
		Metadata:     map[string]string{"test": "response-api-create"},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("http://localhost:%s/v1/responses", localPort)
	if opts.Verbose {
		fmt.Printf("[Test] Sending POST request to %s\n", url)
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

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected status 200, got %d: %s", resp.StatusCode, string(body))
	}

	// Parse and validate response
	var apiResp ResponseAPIResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return fmt.Errorf("failed to parse response: %w", err)
	}

	// Validate response fields
	if apiResp.ID == "" || !strings.HasPrefix(apiResp.ID, "resp_") {
		return fmt.Errorf("invalid response ID: %s (expected resp_xxx format)", apiResp.ID)
	}
	if apiResp.Object != "response" {
		return fmt.Errorf("invalid object type: %s (expected 'response')", apiResp.Object)
	}
	if apiResp.Status != "completed" && apiResp.Status != "in_progress" {
		return fmt.Errorf("unexpected status: %s", apiResp.Status)
	}
	if apiResp.CreatedAt == 0 {
		return fmt.Errorf("created_at should not be zero")
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"response_id": apiResp.ID,
			"status":      apiResp.Status,
			"model":       apiResp.Model,
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✅ Response API create successful (id=%s)\n", apiResp.ID)
	}

	return nil
}

// testResponseAPIGet tests GET /v1/responses/{id}
func testResponseAPIGet(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API: GET /v1/responses/{id}")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// First, create a response to retrieve
	responseID, err := createTestResponse(ctx, localPort, opts.Verbose)
	if err != nil {
		return fmt.Errorf("failed to create test response: %w", err)
	}

	// Now retrieve it
	url := fmt.Sprintf("http://localhost:%s/v1/responses/%s", localPort, responseID)
	if opts.Verbose {
		fmt.Printf("[Test] Sending GET request to %s\n", url)
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
		fmt.Printf("[Test] Response body: %s\n", truncateString(string(body), 500))
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected status 200, got %d: %s", resp.StatusCode, string(body))
	}

	// Parse and validate response
	var apiResp ResponseAPIResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return fmt.Errorf("failed to parse response: %w", err)
	}

	if apiResp.ID != responseID {
		return fmt.Errorf("response ID mismatch: got %s, expected %s", apiResp.ID, responseID)
	}
	if apiResp.Object != "response" {
		return fmt.Errorf("invalid object type: %s", apiResp.Object)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"response_id": apiResp.ID,
			"status":      apiResp.Status,
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✅ Response API get successful (id=%s)\n", apiResp.ID)
	}

	return nil
}

// testResponseAPIDelete tests DELETE /v1/responses/{id}
func testResponseAPIDelete(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API: DELETE /v1/responses/{id}")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// First, create a response to delete
	responseID, err := createTestResponse(ctx, localPort, opts.Verbose)
	if err != nil {
		return fmt.Errorf("failed to create test response: %w", err)
	}

	// Delete it
	url := fmt.Sprintf("http://localhost:%s/v1/responses/%s", localPort, responseID)
	if opts.Verbose {
		fmt.Printf("[Test] Sending DELETE request to %s\n", url)
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

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected status 200, got %d: %s", resp.StatusCode, string(body))
	}

	// Parse and validate response
	var deleteResp DeleteResponseResult
	if err := json.Unmarshal(body, &deleteResp); err != nil {
		return fmt.Errorf("failed to parse response: %w", err)
	}

	if deleteResp.ID != responseID {
		return fmt.Errorf("response ID mismatch: got %s, expected %s", deleteResp.ID, responseID)
	}
	if deleteResp.Object != "response.deleted" {
		return fmt.Errorf("invalid object type: %s (expected 'response.deleted')", deleteResp.Object)
	}
	if !deleteResp.Deleted {
		return fmt.Errorf("deleted should be true")
	}

	// Verify it's actually deleted by trying to get it
	getReq, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
	getResp, err := httpClient.Do(getReq)
	if err != nil {
		return fmt.Errorf("failed to verify deletion: %w", err)
	}
	defer getResp.Body.Close()

	if getResp.StatusCode != http.StatusNotFound {
		return fmt.Errorf("expected 404 after deletion, got %d", getResp.StatusCode)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"deleted_id": responseID,
			"verified":   true,
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✅ Response API delete successful (id=%s)\n", responseID)
	}

	return nil
}

// testResponseAPIInputItems tests GET /v1/responses/{id}/input_items
func testResponseAPIInputItems(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API: GET /v1/responses/{id}/input_items")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// First, create a response with instructions
	responseID, err := createTestResponseWithInstructions(ctx, localPort, opts.Verbose)
	if err != nil {
		return fmt.Errorf("failed to create test response: %w", err)
	}

	// Get input items
	url := fmt.Sprintf("http://localhost:%s/v1/responses/%s/input_items", localPort, responseID)
	if opts.Verbose {
		fmt.Printf("[Test] Sending GET request to %s\n", url)
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
		fmt.Printf("[Test] Response body: %s\n", truncateString(string(body), 500))
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected status 200, got %d: %s", resp.StatusCode, string(body))
	}

	// Parse and validate response
	var listResp InputItemsListResponse
	if err := json.Unmarshal(body, &listResp); err != nil {
		return fmt.Errorf("failed to parse response: %w", err)
	}

	if listResp.Object != "list" {
		return fmt.Errorf("invalid object type: %s (expected 'list')", listResp.Object)
	}
	if len(listResp.Data) == 0 {
		return fmt.Errorf("expected at least one input item")
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"response_id": responseID,
			"item_count":  len(listResp.Data),
			"has_more":    listResp.HasMore,
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✅ Response API input_items successful (id=%s, items=%d)\n", responseID, len(listResp.Data))
	}

	return nil
}

// createTestResponse creates a test response and returns its ID
func createTestResponse(ctx context.Context, localPort string, verbose bool) (string, error) {
	storeTrue := true
	reqBody := ResponseAPIRequest{
		Model: "MoM",
		Input: "Hello, how are you?",
		Store: &storeTrue,
	}

	jsonData, _ := json.Marshal(reqBody)
	url := fmt.Sprintf("http://localhost:%s/v1/responses", localPort)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("failed to create response: %d - %s", resp.StatusCode, string(body))
	}

	var apiResp ResponseAPIResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return "", err
	}

	if verbose {
		fmt.Printf("[Test] Created test response: %s\n", apiResp.ID)
	}

	return apiResp.ID, nil
}

// createTestResponseWithInstructions creates a test response with instructions
func createTestResponseWithInstructions(ctx context.Context, localPort string, verbose bool) (string, error) {
	storeTrue := true
	reqBody := ResponseAPIRequest{
		Model:        "MoM",
		Input:        "What is the capital of France?",
		Instructions: "You are a geography expert. Answer concisely.",
		Store:        &storeTrue,
	}

	jsonData, _ := json.Marshal(reqBody)
	url := fmt.Sprintf("http://localhost:%s/v1/responses", localPort)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("failed to create response: %d - %s", resp.StatusCode, string(body))
	}

	var apiResp ResponseAPIResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return "", err
	}

	if verbose {
		fmt.Printf("[Test] Created test response with instructions: %s\n", apiResp.ID)
	}

	return apiResp.ID, nil
}
