package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("dynamo-dynamic-batching", pkgtestcases.TestCase{
		Description: "Test Dynamo's dynamic batching functionality with concurrent requests",
		Tags:        []string{"dynamo", "batching", "concurrency"},
		Fn:          testDynamoDynamicBatching,
	})
}

func testDynamoDynamicBatching(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Dynamo dynamic batching")
	}

	// Setup service connection and get local port
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// Send concurrent requests to test batching
	const concurrentRequests = 5
	testContent := "What is 2+2? Provide a brief answer."

	if opts.Verbose {
		fmt.Printf("[Test] Sending %d concurrent requests to test batching\n", concurrentRequests)
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	var results []BatchResult

	startTime := time.Now()

	for i := 0; i < concurrentRequests; i++ {
		wg.Add(1)
		go func(requestID int) {
			defer wg.Done()

			result := sendBatchedRequest(ctx, localPort, testContent, requestID, opts.Verbose)

			mu.Lock()
			results = append(results, result)
			mu.Unlock()
		}(i)
	}

	wg.Wait()
	totalTime := time.Since(startTime)

	// Analyze results
	successCount := 0
	var totalLatency time.Duration
	for _, result := range results {
		if result.Success {
			successCount++
			totalLatency += result.Latency
		}
	}

	avgLatency := time.Duration(0)
	if successCount > 0 {
		avgLatency = totalLatency / time.Duration(successCount)
	}

	// Calculate batching efficiency
	// If requests are batched, the total time should be less than sum of individual latencies
	sequentialTime := totalLatency
	batchingEfficiency := float64(sequentialTime) / float64(totalTime)
	if batchingEfficiency > 1.0 {
		batchingEfficiency = 1.0 / batchingEfficiency
	}

	// Set details for reporting
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"concurrent_requests": concurrentRequests,
			"successful":          successCount,
			"success_rate":        fmt.Sprintf("%.2f%%", float64(successCount)/float64(concurrentRequests)*100),
			"avg_latency_ms":      avgLatency.Milliseconds(),
			"total_time_ms":       totalTime.Milliseconds(),
			"batching_efficiency": fmt.Sprintf("%.2f", batchingEfficiency),
			"sequential_time_ms":  sequentialTime.Milliseconds(),
		})
	}

	if opts.Verbose {
		separator := strings.Repeat("=", 80)
		fmt.Println("\n" + separator)
		fmt.Println("Dynamic Batching Test Results")
		fmt.Println(separator)
		fmt.Printf("Concurrent Requests: %d\n", concurrentRequests)
		fmt.Printf("Successful:          %d (%.2f%%)\n", successCount, float64(successCount)/float64(concurrentRequests)*100)
		fmt.Printf("Average Latency:    %v (%.2f ms)\n", avgLatency, float64(avgLatency.Milliseconds()))
		fmt.Printf("Total Time:         %v (%.2f ms)\n", totalTime, float64(totalTime.Milliseconds()))
		fmt.Printf("Sequential Time:    %v (%.2f ms)\n", sequentialTime, float64(sequentialTime.Milliseconds()))
		fmt.Printf("Batching Efficiency: %.2f%%\n", batchingEfficiency*100)
		fmt.Println(separator)

		if batchingEfficiency > 0.5 {
			fmt.Println("[Test] ✅ Dynamic batching appears to be working (efficiency > 50%)")
		} else {
			fmt.Println("[Test] ⚠️  Batching efficiency is low - may need tuning")
		}
	}

	return nil
}

type BatchResult struct {
	RequestID int
	Success   bool
	Latency   time.Duration
	Error     string
}

func sendBatchedRequest(ctx context.Context, localPort, content string, requestID int, verbose bool) BatchResult {
	result := BatchResult{
		RequestID: requestID,
		Success:   false,
	}

	start := time.Now()

	requestBody := map[string]interface{}{
		"model": "MoM",
		"messages": []map[string]string{
			{
				"role":    "user",
				"content": fmt.Sprintf("[Request %d] %s", requestID, content),
			},
		},
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
		Timeout: 30 * time.Second,
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

	if verbose {
		fmt.Printf("[Test] Request %d: %s (latency: %v)\n", requestID,
			map[bool]string{true: "success", false: "failed"}[result.Success], result.Latency)
	}

	return result
}
