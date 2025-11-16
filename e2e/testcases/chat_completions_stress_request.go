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
	pkgtestcases.Register("chat-completions-stress-request", pkgtestcases.TestCase{
		Description: "Send 200 sequential requests and measure success rate",
		Tags:        []string{"llm", "stress", "reliability"},
		Fn:          testStressTest,
	})
}

// StressTestResult tracks the result of a single request
type StressTestResult struct {
	RequestID    int
	Success      bool
	StatusCode   int
	Duration     time.Duration
	ErrorMessage string
}

func testStressTest(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Starting stress test: 1000 sequential requests")
	}

	// Setup service connection and get local port
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward() // Ensure port forwarding is stopped when test completes

	const totalRequests = 200
	var results []StressTestResult
	successCount := 0
	totalDuration := time.Duration(0)

	// Send 100 requests sequentially
	for i := 1; i <= totalRequests; i++ {
		result := sendSingleRequest(ctx, i, localPort, opts.Verbose)
		results = append(results, result)

		if result.Success {
			successCount++
		}
		totalDuration += result.Duration

		// Print progress every 100 requests
		if opts.Verbose && i%100 == 0 {
			currentSuccessRate := float64(successCount) / float64(i) * 100
			fmt.Printf("[Test] Progress: %d/%d requests completed (%.2f%% success rate)\n",
				i, totalRequests, currentSuccessRate)
		}
	}

	// Calculate statistics
	successRate := float64(successCount) / float64(totalRequests) * 100
	failureCount := totalRequests - successCount
	avgDuration := totalDuration / time.Duration(totalRequests)

	// Set details for reporting
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_requests":  totalRequests,
			"successful":      successCount,
			"failed":          failureCount,
			"success_rate":    fmt.Sprintf("%.2f%%", successRate),
			"avg_duration_ms": avgDuration.Milliseconds(),
		})
	}

	// Print summary
	printStressTestResults(results, totalRequests, successCount, successRate, avgDuration)

	if opts.Verbose {
		fmt.Printf("[Test] Stress test completed: %d/%d successful (%.2f%% success rate)\n",
			successCount, totalRequests, successRate)
	}

	return nil
}

func sendSingleRequest(ctx context.Context, requestID int, localPort string, verbose bool) StressTestResult {
	result := StressTestResult{
		RequestID: requestID,
		Success:   false,
	}

	start := time.Now()

	// Prepare request body with random content
	requestBody := map[string]interface{}{
		"model": "MoM",
		"messages": []map[string]string{
			{
				"role":    "user",
				"content": generateRandomContent(requestID),
			},
		},
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		result.ErrorMessage = fmt.Sprintf("marshal error: %v", err)
		result.Duration = time.Since(start)
		return result
	}

	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		result.ErrorMessage = fmt.Sprintf("create request error: %v", err)
		result.Duration = time.Since(start)
		return result
	}

	req.Header.Set("Content-Type", "application/json")

	httpClient := &http.Client{
		Timeout: 30 * time.Second,
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		result.ErrorMessage = fmt.Sprintf("send request error: %v", err)
		result.Duration = time.Since(start)
		return result
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		result.ErrorMessage = fmt.Sprintf("read response error: %v", err)
		result.Duration = time.Since(start)
		result.StatusCode = resp.StatusCode
		return result
	}

	result.Duration = time.Since(start)
	result.StatusCode = resp.StatusCode

	if resp.StatusCode == http.StatusOK {
		result.Success = true
	} else {
		result.ErrorMessage = fmt.Sprintf("status %d: %s", resp.StatusCode, string(body))
	}

	return result
}

func printStressTestResults(results []StressTestResult, totalRequests, successCount int, successRate float64, avgDuration time.Duration) {
	separator := strings.Repeat("=", 80)
	fmt.Println("\n" + separator)
	fmt.Println("Stress Test Results")
	fmt.Println(separator)
	fmt.Printf("Total Requests:    %d\n", totalRequests)
	fmt.Printf("Successful:        %d\n", successCount)
	fmt.Printf("Failed:            %d\n", totalRequests-successCount)
	fmt.Printf("Success Rate:      %.2f%%\n", successRate)
	fmt.Printf("Average Duration:  %v\n", avgDuration)
	fmt.Println(separator)

	// Show first 10 failures if any
	failureCount := 0
	fmt.Println("\nFirst 10 Failures (if any):")
	for _, result := range results {
		if !result.Success && failureCount < 10 {
			failureCount++
			fmt.Printf("  Request #%d: %s (duration: %v)\n",
				result.RequestID, result.ErrorMessage, result.Duration)
		}
		if failureCount >= 10 {
			break
		}
	}

	if failureCount == 0 {
		fmt.Println("  No failures! 🎉")
	}
	fmt.Println()
}
