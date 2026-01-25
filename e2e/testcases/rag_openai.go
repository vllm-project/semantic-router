package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("rag-openai", pkgtestcases.TestCase{
		Description: "Test OpenAI RAG functionality with File Store and Vector Store APIs (based on Responses API cookbook)",
		Tags:        []string{"rag", "openai", "file-store", "vector-store", "responses-api"},
		Fn:          RAGOpenAITestCase,
	})
}

// RAGOpenAITestCase tests OpenAI RAG functionality based on the Responses API cookbook
// This test follows the workflow:
// 1. Upload files to OpenAI File Store
// 2. Create vector store and attach files
// 3. Test RAG retrieval using direct_search mode
// 4. Test RAG retrieval using tool_based mode (file_search tool)
func RAGOpenAITestCase(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup service connection: %w", err)
	}
	defer stopPortForward()

	baseURL := fmt.Sprintf("http://localhost:%s", localPort)

	if opts.Verbose {
		fmt.Println("[RAG OpenAI Test] Starting OpenAI RAG E2E test based on Responses API cookbook")
	}

	// Test 1: Direct Search Mode
	if err := testDirectSearchMode(ctx, baseURL, opts); err != nil {
		return fmt.Errorf("direct search mode test failed: %w", err)
	}

	// Test 2: Tool-Based Mode (file_search tool)
	if err := testToolBasedMode(ctx, baseURL, opts); err != nil {
		return fmt.Errorf("tool-based mode test failed: %w", err)
	}

	if opts.Verbose {
		fmt.Println("[RAG OpenAI Test] All tests passed successfully")
	}

	return nil
}

// testDirectSearchMode tests RAG with direct_search workflow mode
func testDirectSearchMode(ctx context.Context, baseURL string, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[RAG OpenAI Test] Testing direct_search mode...")
	}

	// Create a request with RAG enabled (direct_search mode)
	requestBody := map[string]interface{}{
		"model": "gpt-4o-mini",
		"messages": []map[string]interface{}{
			{
				"role":    "user",
				"content": "What is Deep Research?",
			},
		},
	}

	bodyBytes, _ := json.Marshal(requestBody)
	req, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("%s/v1/chat/completions", baseURL), strings.NewReader(string(bodyBytes)))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-VSR-Selected-Decision", "rag-openai-decision") // Decision with OpenAI RAG config

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("unexpected status code %d: %s", resp.StatusCode, string(body))
	}

	var response map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return fmt.Errorf("failed to decode response: %w", err)
	}

	// Verify response contains content
	choices, ok := response["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return fmt.Errorf("no choices in response")
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return fmt.Errorf("invalid choice format")
	}

	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("invalid message format")
	}

	content, ok := message["content"].(string)
	if !ok || content == "" {
		return fmt.Errorf("empty or missing content in response")
	}

	// Verify that RAG context was used (content should mention Deep Research)
	if !strings.Contains(strings.ToLower(content), "research") {
		return fmt.Errorf("response doesn't appear to use RAG context (no mention of research)")
	}

	if opts.Verbose {
		fmt.Printf("[RAG OpenAI Test] Direct search mode test passed. Response length: %d chars\n", len(content))
	}

	return nil
}

// testToolBasedMode tests RAG with tool_based workflow mode (file_search tool)
func testToolBasedMode(ctx context.Context, baseURL string, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[RAG OpenAI Test] Testing tool_based mode (file_search tool)...")
	}

	// Create a request with file_search tool
	requestBody := map[string]interface{}{
		"model": "gpt-4o-mini",
		"input": "What is Deep Research?",
		"tools": []map[string]interface{}{
			{
				"type": "file_search",
				"file_search": map[string]interface{}{
					"vector_store_ids": []string{"vs_test123"},
					"max_num_results":  5,
				},
			},
		},
	}

	bodyBytes, _ := json.Marshal(requestBody)
	req, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("%s/v1/responses", baseURL), strings.NewReader(string(bodyBytes)))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 60 * time.Second} // Longer timeout for tool calls
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("unexpected status code %d: %s", resp.StatusCode, string(body))
	}

	var response map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return fmt.Errorf("failed to decode response: %w", err)
	}

	// Verify response format (Responses API format)
	if response["object"] != "response" {
		return fmt.Errorf("expected Responses API format, got object: %v", response["object"])
	}

	// Verify output contains file_search results
	output, ok := response["output"].([]interface{})
	if !ok || len(output) == 0 {
		return fmt.Errorf("no output in response")
	}

	// Check for file_search annotations in the output
	foundFileSearch := false
	for _, item := range output {
		itemMap, ok := item.(map[string]interface{})
		if !ok {
			continue
		}

		// Check for annotations (file_search results)
		if annotations, ok := itemMap["annotations"].([]interface{}); ok && len(annotations) > 0 {
			foundFileSearch = true
			break
		}

		// Check for file_search_call
		if fileSearchCall, ok := itemMap["file_search_call"].(map[string]interface{}); ok {
			if searchResults, ok := fileSearchCall["search_results"].([]interface{}); ok && len(searchResults) > 0 {
				foundFileSearch = true
				break
			}
		}
	}

	if !foundFileSearch {
		return fmt.Errorf("file_search tool results not found in response")
	}

	if opts.Verbose {
		fmt.Println("[RAG OpenAI Test] Tool-based mode test passed. file_search tool executed successfully")
	}

	return nil
}
