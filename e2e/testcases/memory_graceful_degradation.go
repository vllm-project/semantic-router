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
	pkgtestcases.Register("memory-graceful-degradation", pkgtestcases.TestCase{
		Description: "Verify requests succeed when memory backend (Milvus) is unavailable",
		Tags:        []string{"memory", "graceful-degradation", "resilience", "functional"},
		Fn:          testMemoryGracefulDegradation,
	})
}

// degradationRequest represents a Response API request used in graceful degradation tests.
type degradationRequest struct {
	Model        string                `json:"model"`
	Input        interface{}           `json:"input"`
	Instructions string                `json:"instructions,omitempty"`
	MemoryConfig *degradationMemoryCfg `json:"memory_config,omitempty"`
	MemoryCtx    *degradationMemoryCtx `json:"memory_context,omitempty"`
}

type degradationMemoryCfg struct {
	Enabled   bool `json:"enabled"`
	AutoStore bool `json:"auto_store,omitempty"`
}

type degradationMemoryCtx struct {
	UserID string `json:"user_id"`
}

type degradationResponse struct {
	ID         string                   `json:"id"`
	Status     string                   `json:"status"`
	Output     []map[string]interface{} `json:"output"`
	OutputText string                   `json:"output_text,omitempty"`
}

// testMemoryGracefulDegradation verifies that the router continues to serve
// requests when the memory backend is unavailable or not configured.
// Success criteria: every request returns status "completed" even with
// memory_config.enabled=true and auto_store=true.
func testMemoryGracefulDegradation(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Memory Graceful Degradation")
		fmt.Println("[Test] Success criteria: requests succeed even when memory backend is unavailable")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	userID := "user_degradation_test"

	// Test 1: Request with memory store enabled (auto_store=true)
	// The memory backend may be down or not configured; the request must still succeed.
	if opts.Verbose {
		fmt.Println("[Test] Step 1: Sending request with memory auto_store=true (memory backend may be unavailable)")
	}

	storeReq := degradationRequest{
		Model:        "MoM",
		Input:        "Remember this: my favorite color is blue",
		Instructions: "You are a helpful assistant.",
		MemoryConfig: &degradationMemoryCfg{
			Enabled:   true,
			AutoStore: true,
		},
		MemoryCtx: &degradationMemoryCtx{
			UserID: userID,
		},
	}

	storeResp, err := sendDegradationRequest(ctx, localPort, storeReq, opts.Verbose)
	if err != nil {
		return fmt.Errorf("request with auto_store=true failed (should succeed even if memory backend is down): %w", err)
	}

	if storeResp.Status != "completed" {
		return fmt.Errorf("request with auto_store=true returned status %q, expected \"completed\"", storeResp.Status)
	}

	if opts.Verbose {
		fmt.Println("[Test] ✓ Step 1 passed: request with auto_store=true completed successfully")
	}

	// Brief pause between requests
	time.Sleep(1 * time.Second)

	// Test 2: Request with memory retrieval enabled (auto_store=false)
	// Memory retrieval will fail silently; the LLM response must still be returned.
	if opts.Verbose {
		fmt.Println("[Test] Step 2: Sending request with memory retrieval enabled (memory backend may be unavailable)")
	}

	retrieveReq := degradationRequest{
		Model:        "MoM",
		Input:        "What is my favorite color?",
		Instructions: "You are a helpful assistant. Use your memory to answer questions.",
		MemoryConfig: &degradationMemoryCfg{
			Enabled:   true,
			AutoStore: false,
		},
		MemoryCtx: &degradationMemoryCtx{
			UserID: userID,
		},
	}

	retrieveResp, err := sendDegradationRequest(ctx, localPort, retrieveReq, opts.Verbose)
	if err != nil {
		return fmt.Errorf("request with memory retrieval failed (should succeed even if memory backend is down): %w", err)
	}

	if retrieveResp.Status != "completed" {
		return fmt.Errorf("request with memory retrieval returned status %q, expected \"completed\"", retrieveResp.Status)
	}

	if opts.Verbose {
		fmt.Println("[Test] ✓ Step 2 passed: request with memory retrieval completed successfully")
	}

	time.Sleep(1 * time.Second)

	// Test 3: Request without memory (baseline sanity check)
	if opts.Verbose {
		fmt.Println("[Test] Step 3: Sending request without memory config (baseline)")
	}

	baselineReq := degradationRequest{
		Model:        "MoM",
		Input:        "What is 2 + 2?",
		Instructions: "You are a helpful assistant.",
	}

	baselineResp, err := sendDegradationRequest(ctx, localPort, baselineReq, opts.Verbose)
	if err != nil {
		return fmt.Errorf("baseline request without memory config failed: %w", err)
	}

	if baselineResp.Status != "completed" {
		return fmt.Errorf("baseline request returned status %q, expected \"completed\"", baselineResp.Status)
	}

	if opts.Verbose {
		fmt.Println("[Test] ✓ Step 3 passed: baseline request without memory completed successfully")
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"store_request_status":    storeResp.Status,
			"retrieve_request_status": retrieveResp.Status,
			"baseline_request_status": baselineResp.Status,
			"graceful_degradation":    true,
		})
	}

	if opts.Verbose {
		fmt.Println("[Test] ✅ Memory graceful degradation test passed!")
		fmt.Println("[Test]    All requests completed successfully regardless of memory backend availability")
	}

	return nil
}

func sendDegradationRequest(ctx context.Context, localPort string, req degradationRequest, verbose bool) (*degradationResponse, error) {
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("http://localhost:%s/v1/responses", localPort)
	if verbose {
		fmt.Printf("[Test] Sending request to %s\n", url)
		fmt.Printf("[Test] Request body: %s\n", truncateString(string(jsonData), 500))
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpClient := &http.Client{Timeout: 60 * time.Second}
	resp, err := httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if verbose {
		fmt.Printf("[Test] Response status: %d\n", resp.StatusCode)
		fmt.Printf("[Test] Response body: %s\n", truncateString(string(body), 500))
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("expected status 200, got %d: %s", resp.StatusCode, string(body))
	}

	var apiResp degradationResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &apiResp, nil
}
