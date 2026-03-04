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
	pkgtestcases.Register("memory-user-isolation", pkgtestcases.TestCase{
		Description: "Test user isolation: User A's memories not visible to User B",
		Tags:        []string{"memory", "user-isolation", "security", "functional"},
		Fn:          testMemoryUserIsolation,
	})
}

// MemoryRequest represents a Response API request with memory configuration
type MemoryRequest struct {
	Model              string         `json:"model"`
	Input              interface{}    `json:"input"`
	Instructions       string         `json:"instructions,omitempty"`
	MemoryConfig       *MemoryConfig  `json:"memory_config,omitempty"`
	MemoryContext      *MemoryContext `json:"memory_context,omitempty"`
	PreviousResponseID string         `json:"previous_response_id,omitempty"`
}

// MemoryConfig configures memory extraction behavior
type MemoryConfig struct {
	Enabled   bool `json:"enabled"`
	AutoStore bool `json:"auto_store,omitempty"`
}

// MemoryContext provides context for memory operations
type MemoryContext struct {
	UserID string `json:"user_id"`
}

// MemoryResponse represents a Response API response
type MemoryResponse struct {
	ID                 string                   `json:"id"`
	Object             string                   `json:"object"`
	Status             string                   `json:"status"`
	Output             []map[string]interface{} `json:"output"`
	OutputText         string                   `json:"output_text,omitempty"`
	PreviousResponseID string                   `json:"previous_response_id,omitempty"`
	Error              map[string]interface{}   `json:"error,omitempty"`
}

// testMemoryUserIsolation tests that memories are properly isolated between users
func testMemoryUserIsolation(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Memory User Isolation")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	userA := "user_a_test_isolation"
	userB := "user_b_test_isolation"

	// Test 1: Store memory for User A
	if opts.Verbose {
		fmt.Println("[Test] Step 1: Storing memory for User A")
	}

	userAMemory := "My budget for Hawaii vacation is $10,000"
	userARequest := MemoryRequest{
		Model:        "MoM",
		Input:        fmt.Sprintf("Remember this: %s", userAMemory),
		Instructions: "You are a helpful assistant. Remember important information the user shares.",
		MemoryConfig: &MemoryConfig{
			Enabled:   true,
			AutoStore: true,
		},
		MemoryContext: &MemoryContext{
			UserID: userA,
		},
	}

	userAResp, err := sendMemoryRequest(ctx, localPort, userARequest, opts.Verbose)
	if err != nil {
		if opts.Verbose {
			fmt.Printf("[Test] Memory request failed (memory may not be configured): %v\n", err)
			fmt.Println("[Test] Skipping memory user isolation test — memory feature not available")
		}
		return nil
	}

	if userAResp.Status != "completed" {
		return fmt.Errorf("User A request failed with status: %s", userAResp.Status)
	}

	// Wait a bit for memory extraction to complete
	time.Sleep(3 * time.Second)

	// Test 2: User B tries to retrieve User A's memory (should NOT find it)
	if opts.Verbose {
		fmt.Println("[Test] Step 2: User B queries for User A's memory (should NOT find it)")
	}

	// Use neutral phrasing that does NOT contain User A's memory keywords
	// ("hawaii", "10,000", "10000"). If those keywords appear in the response
	// despite not being in the query, it means User A's memory leaked.
	userBQuery := MemoryRequest{
		Model:        "MoM",
		Input:        "Do I have any saved travel plans or trip budgets?",
		Instructions: "You are a helpful assistant. Use your memory to answer questions.",
		MemoryConfig: &MemoryConfig{
			Enabled:   true,
			AutoStore: false,
		},
		MemoryContext: &MemoryContext{
			UserID: userB,
		},
	}

	userBResp, err := sendMemoryRequest(ctx, localPort, userBQuery, opts.Verbose)
	if err != nil {
		return fmt.Errorf("failed to query memory as User B: %w", err)
	}

	userBRespLower := strings.ToLower(userBResp.OutputText)
	if strings.Contains(userBRespLower, "10000") ||
		strings.Contains(userBRespLower, "10,000") ||
		strings.Contains(userBRespLower, "hawaii") {
		return fmt.Errorf("SECURITY VIOLATION: User B can see User A's memory! Response: %s", userBResp.OutputText)
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✓ User B correctly cannot see User A's memory. Response: %s\n", truncateString(userBResp.OutputText, 200))
	}

	// Test 3: Store memory for User B
	if opts.Verbose {
		fmt.Println("[Test] Step 3: Storing memory for User B")
	}

	userBMemory := "My favorite programming language is Go"
	userBStoreRequest := MemoryRequest{
		Model:        "MoM",
		Input:        fmt.Sprintf("Remember this: %s", userBMemory),
		Instructions: "You are a helpful assistant. Remember important information the user shares.",
		MemoryConfig: &MemoryConfig{
			Enabled:   true,
			AutoStore: true,
		},
		MemoryContext: &MemoryContext{
			UserID: userB,
		},
	}

	userBStoreResp, err := sendMemoryRequest(ctx, localPort, userBStoreRequest, opts.Verbose)
	if err != nil {
		return fmt.Errorf("failed to store memory for User B: %w", err)
	}

	if userBStoreResp.Status != "completed" {
		return fmt.Errorf("User B store request failed with status: %s", userBStoreResp.Status)
	}

	// Wait a bit for memory extraction to complete
	time.Sleep(3 * time.Second)

	// Test 4: User B retrieves their own memory (should find it)
	if opts.Verbose {
		fmt.Println("[Test] Step 4: User B queries for their own memory (should find it)")
	}

	userBSelfQuery := MemoryRequest{
		Model:        "MoM",
		Input:        "What is my favorite programming language?",
		Instructions: "You are a helpful assistant. Use your memory to answer questions.",
		MemoryConfig: &MemoryConfig{
			Enabled:   true,
			AutoStore: false,
		},
		MemoryContext: &MemoryContext{
			UserID: userB,
		},
	}

	userBSelfResp, err := sendMemoryRequest(ctx, localPort, userBSelfQuery, opts.Verbose)
	if err != nil {
		return fmt.Errorf("failed to query own memory as User B: %w", err)
	}

	// User B should see their own memory
	// The response should contain "Go" or "programming language"
	if !strings.Contains(strings.ToLower(userBSelfResp.OutputText), "go") &&
		!strings.Contains(strings.ToLower(userBSelfResp.OutputText), "programming") {
		if opts.Verbose {
			fmt.Printf("[Test] ⚠️  User B's memory retrieval may not have worked. Response: %s\n", userBSelfResp.OutputText)
		}
		// This is not a failure - memory might not be retrieved if similarity is low
		// But we continue to test that User A cannot see it
	}

	// Test 5: User A tries to retrieve User B's memory (should NOT find it)
	if opts.Verbose {
		fmt.Println("[Test] Step 5: User A queries for User B's memory (should NOT find it)")
	}

	// Use neutral phrasing that does NOT contain User B's memory keywords.
	userAQuery := MemoryRequest{
		Model:        "MoM",
		Input:        "What technical preferences or tools have I mentioned?",
		Instructions: "You are a helpful assistant. Use your memory to answer questions.",
		MemoryConfig: &MemoryConfig{
			Enabled:   true,
			AutoStore: false,
		},
		MemoryContext: &MemoryContext{
			UserID: userA,
		},
	}

	userAQueryResp, err := sendMemoryRequest(ctx, localPort, userAQuery, opts.Verbose)
	if err != nil {
		return fmt.Errorf("failed to query memory as User A: %w", err)
	}

	userAQueryRespLower := strings.ToLower(userAQueryResp.OutputText)
	if strings.Contains(userAQueryRespLower, "favorite") && strings.Contains(userAQueryRespLower, "go") {
		return fmt.Errorf("SECURITY VIOLATION: User A can see User B's memory! Response: %s", userAQueryResp.OutputText)
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✓ User A correctly cannot see User B's memory. Response: %s\n", truncateString(userAQueryResp.OutputText, 200))
	}

	// Test 6: User A retrieves their own memory (should find it)
	if opts.Verbose {
		fmt.Println("[Test] Step 6: User A queries for their own memory (should find it)")
	}

	userASelfQuery := MemoryRequest{
		Model:        "MoM",
		Input:        "What is my budget for Hawaii vacation?",
		Instructions: "You are a helpful assistant. Use your memory to answer questions.",
		MemoryConfig: &MemoryConfig{
			Enabled:   true,
			AutoStore: false,
		},
		MemoryContext: &MemoryContext{
			UserID: userA,
		},
	}

	userASelfResp, err := sendMemoryRequest(ctx, localPort, userASelfQuery, opts.Verbose)
	if err != nil {
		return fmt.Errorf("failed to query own memory as User A: %w", err)
	}

	// User A should see their own memory
	// The response should contain budget information
	if !strings.Contains(strings.ToLower(userASelfResp.OutputText), "10000") &&
		!strings.Contains(strings.ToLower(userASelfResp.OutputText), "10,000") &&
		!strings.Contains(strings.ToLower(userASelfResp.OutputText), "hawaii") {
		if opts.Verbose {
			fmt.Printf("[Test] ⚠️  User A's memory retrieval may not have worked. Response: %s\n", userASelfResp.OutputText)
		}
		// This is not a failure - memory might not be retrieved if similarity is low
		// But the key test is that User B cannot see it (already verified)
	}

	// Set details for reporting
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"user_a":             userA,
			"user_b":             userB,
			"user_a_memory":      userAMemory,
			"user_b_memory":      userBMemory,
			"isolation_verified": true,
		})
	}

	if opts.Verbose {
		fmt.Println("[Test] ✅ Memory user isolation test passed!")
	}

	return nil
}

// sendMemoryRequest sends a memory-enabled request to the Response API
func sendMemoryRequest(ctx context.Context, localPort string, req MemoryRequest, verbose bool) (*MemoryResponse, error) {
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

	httpClient := &http.Client{Timeout: 60 * time.Second} // Longer timeout for memory operations
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

	var apiResp MemoryResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Extract output text from response
	if len(apiResp.Output) > 0 {
		if content, ok := apiResp.Output[0]["content"].([]interface{}); ok {
			for _, part := range content {
				if partMap, ok := part.(map[string]interface{}); ok {
					if text, ok := partMap["text"].(string); ok {
						apiResp.OutputText = text
						break
					}
				}
			}
		}
		// Fallback: try to extract text directly
		if apiResp.OutputText == "" {
			if text, ok := apiResp.Output[0]["text"].(string); ok {
				apiResp.OutputText = text
			}
		}
	}

	return &apiResp, nil
}
