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
	pkgtestcases.Register("dynamo-performance-comparison", pkgtestcases.TestCase{
		Description: "Compare performance metrics with Dynamo optimizations (latency, throughput)",
		Tags:        []string{"dynamo", "performance", "benchmark"},
		Fn:          testDynamoPerformanceComparison,
	})
}

func testDynamoPerformanceComparison(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Running performance comparison test")
	}

	// Setup service connection and get local port
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// Test with a batch of requests to measure throughput
	const batchSize = 10
	testContent := "What is the capital of France? Provide a brief explanation."

	if opts.Verbose {
		fmt.Printf("[Test] Sending %d requests to measure performance\n", batchSize)
	}

	var latencies []time.Duration
	successCount := 0
	startTime := time.Now()

	for i := 0; i < batchSize; i++ {
		reqStart := time.Now()

		requestBody := map[string]interface{}{
			"model": "MoM",
			"messages": []map[string]string{
				{
					"role":    "user",
					"content": testContent,
				},
			},
		}

		jsonData, err := json.Marshal(requestBody)
		if err != nil {
			if opts.Verbose {
				fmt.Printf("[Test] Request %d: marshal error: %v\n", i+1, err)
			}
			continue
		}

		url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
		req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
		if err != nil {
			if opts.Verbose {
				fmt.Printf("[Test] Request %d: create request error: %v\n", i+1, err)
			}
			continue
		}

		req.Header.Set("Content-Type", "application/json")

		httpClient := &http.Client{
			Timeout: 30 * time.Second,
		}

		resp, err := httpClient.Do(req)
		if err != nil {
			if opts.Verbose {
				fmt.Printf("[Test] Request %d: send error: %v\n", i+1, err)
			}
			continue
		}

		_, err = io.ReadAll(resp.Body)
		resp.Body.Close()

		latency := time.Since(reqStart)
		latencies = append(latencies, latency)

		if resp.StatusCode == http.StatusOK && err == nil {
			successCount++
		}

		// Small delay between requests
		if i < batchSize-1 {
			time.Sleep(50 * time.Millisecond)
		}
	}

	totalTime := time.Since(startTime)
	throughput := float64(successCount) / totalTime.Seconds() // requests per second

	// Calculate latency statistics
	var totalLatency time.Duration
	minLatency := time.Hour
	maxLatency := time.Duration(0)

	for _, lat := range latencies {
		totalLatency += lat
		if lat < minLatency {
			minLatency = lat
		}
		if lat > maxLatency {
			maxLatency = lat
		}
	}

	avgLatency := totalLatency / time.Duration(len(latencies))
	p50Latency := calculatePercentile(latencies, 50)
	p95Latency := calculatePercentile(latencies, 95)
	p99Latency := calculatePercentile(latencies, 99)

	// Set details for reporting
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_requests":     batchSize,
			"successful":         successCount,
			"success_rate":       fmt.Sprintf("%.2f%%", float64(successCount)/float64(batchSize)*100),
			"throughput_rps":     fmt.Sprintf("%.2f", throughput),
			"avg_latency_ms":     avgLatency.Milliseconds(),
			"min_latency_ms":     minLatency.Milliseconds(),
			"max_latency_ms":     maxLatency.Milliseconds(),
			"p50_latency_ms":     p50Latency.Milliseconds(),
			"p95_latency_ms":     p95Latency.Milliseconds(),
			"p99_latency_ms":     p99Latency.Milliseconds(),
			"total_time_seconds": totalTime.Seconds(),
		})
	}

	// Print summary
	if opts.Verbose {
		separator := strings.Repeat("=", 80)
		fmt.Println("\n" + separator)
		fmt.Println("Performance Comparison Results")
		fmt.Println(separator)
		fmt.Printf("Total Requests:     %d\n", batchSize)
		fmt.Printf("Successful:         %d (%.2f%%)\n", successCount, float64(successCount)/float64(batchSize)*100)
		fmt.Printf("Throughput:         %.2f requests/second\n", throughput)
		fmt.Printf("Average Latency:    %v (%.2f ms)\n", avgLatency, float64(avgLatency.Milliseconds()))
		fmt.Printf("Min Latency:        %v (%.2f ms)\n", minLatency, float64(minLatency.Milliseconds()))
		fmt.Printf("Max Latency:        %v (%.2f ms)\n", maxLatency, float64(maxLatency.Milliseconds()))
		fmt.Printf("P50 Latency:        %v (%.2f ms)\n", p50Latency, float64(p50Latency.Milliseconds()))
		fmt.Printf("P95 Latency:        %v (%.2f ms)\n", p95Latency, float64(p95Latency.Milliseconds()))
		fmt.Printf("P99 Latency:        %v (%.2f ms)\n", p99Latency, float64(p99Latency.Milliseconds()))
		fmt.Printf("Total Time:         %v\n", totalTime)
		fmt.Println(separator)
		fmt.Println("[Test] ✅ Performance comparison test completed")
	}

	return nil
}

func calculatePercentile(latencies []time.Duration, percentile int) time.Duration {
	if len(latencies) == 0 {
		return 0
	}

	// Simple percentile calculation (not exact but good enough for testing)
	index := (len(latencies) * percentile) / 100
	if index >= len(latencies) {
		index = len(latencies) - 1
	}
	return latencies[index]
}
