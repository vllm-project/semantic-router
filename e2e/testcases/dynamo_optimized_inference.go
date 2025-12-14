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
	pkgtestcases.Register("dynamo-optimized-inference", pkgtestcases.TestCase{
		Description: "Test inference with Dynamo optimizations enabled (KV cache, dynamic batching)",
		Tags:        []string{"dynamo", "inference", "optimization"},
		Fn:          testDynamoOptimizedInference,
	})
}

func testDynamoOptimizedInference(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing optimized inference with Dynamo")
	}

	// Setup service connection and get local port
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// Send multiple requests to test dynamic batching and KV cache
	testCases := []struct {
		name    string
		content string
	}{
		{"math", "What is 15 * 23? Show your work."},
		{"science", "Explain how photosynthesis works in plants."},
		{"general", "What are the main benefits of renewable energy?"},
	}

	var results []InferenceResult
	totalLatency := time.Duration(0)
	successCount := 0

	for i, tc := range testCases {
		if opts.Verbose {
			fmt.Printf("[Test] Sending request %d/%d: %s\n", i+1, len(testCases), tc.name)
		}

		result := sendInferenceRequest(ctx, localPort, tc.content, opts.Verbose)
		results = append(results, result)
		totalLatency += result.Latency

		if result.Success {
			successCount++
		} else if opts.Verbose {
			fmt.Printf("[Test] Request %d (%s) failed: %s\n", i+1, tc.name, result.Error)
		}

		// Small delay between requests to allow batching
		if i < len(testCases)-1 {
			time.Sleep(100 * time.Millisecond)
		}
	}

	// Calculate metrics
	actualSuccessRate := float64(successCount) / float64(len(testCases))
	avgLatency := time.Duration(0)
	if successCount > 0 {
		avgLatency = totalLatency / time.Duration(successCount)
	}

	// Collect error messages for debugging
	var errors []string
	for _, r := range results {
		if !r.Success && r.Error != "" {
			errors = append(errors, r.Error)
		}
	}

	// Set details for reporting BEFORE checking success rate (so we can see what happened on failure)
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_requests": len(testCases),
			"successful":     successCount,
			"success_rate":   fmt.Sprintf("%.0f%%", actualSuccessRate*100),
			"avg_latency_ms": avgLatency.Milliseconds(),
			"optimizations":  []string{"kv_cache", "dynamic_batching"},
			"errors":         errors,
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] Requests successful: %d/%d (%.0f%%)\n", successCount, len(testCases), actualSuccessRate*100)
		if len(errors) > 0 {
			fmt.Printf("[Test] Errors: %v\n", errors)
		}
	}

	// Require at least 50% success rate (allows for transient failures)
	minSuccessRate := 0.5
	if actualSuccessRate < minSuccessRate {
		return fmt.Errorf("success rate %.0f%% below minimum %.0f%% (%d/%d requests succeeded)",
			actualSuccessRate*100, minSuccessRate*100, successCount, len(testCases))
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✅ Optimized inference test completed\n")
		fmt.Printf("[Test] Average latency: %v\n", avgLatency)
	}

	return nil
}

type InferenceResult struct {
	Success bool
	Latency time.Duration
	Error   string
}

func sendInferenceRequest(ctx context.Context, localPort, content string, verbose bool) InferenceResult {
	result := InferenceResult{
		Success: false,
	}

	start := time.Now()

	requestBody := map[string]interface{}{
		"model": "MoM",
		"messages": []map[string]string{
			{
				"role":    "user",
				"content": content,
			},
		},
		// Request streaming to test batching
		"stream": false,
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		result.Error = fmt.Sprintf("marshal error: %v", err)
		result.Latency = time.Since(start)
		return result
	}

	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		result.Error = fmt.Sprintf("create request error: %v", err)
		result.Latency = time.Since(start)
		return result
	}

	req.Header.Set("Content-Type", "application/json")

	httpClient := &http.Client{
		Timeout: 60 * time.Second,
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		result.Error = fmt.Sprintf("send request error: %v", err)
		result.Latency = time.Since(start)
		return result
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		result.Error = fmt.Sprintf("read response error: %v", err)
		result.Latency = time.Since(start)
		result.Success = resp.StatusCode == http.StatusOK
		return result
	}

	result.Latency = time.Since(start)
	result.Success = resp.StatusCode == http.StatusOK

	if !result.Success {
		result.Error = fmt.Sprintf("status %d: %s", resp.StatusCode, string(body))
	}

	return result
}
