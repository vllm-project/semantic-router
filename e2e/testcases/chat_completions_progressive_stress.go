package testcases

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("chat-completions-progressive-stress", pkgtestcases.TestCase{
		Description: "Progressive stress test with 10/20 concurrent requests and success rate tracking",
		Tags:        []string{"llm", "stress", "progressive", "concurrency"},
		Fn:          testProgressiveStress,
	})
}

// ProgressiveStageResult tracks results for a single QPS stage
type ProgressiveStageResult struct {
	QPS          int
	TotalReqs    int
	SuccessCount int
	FailureCount int
	SuccessRate  float64
	AvgDuration  time.Duration
	MinDuration  time.Duration
	MaxDuration  time.Duration
	Results      []StressTestResult
}

func testProgressiveStress(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Starting progressive stress test: 10/20/50 concurrent requests")
	}

	// Setup service connection and get local port
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// Define concurrent request counts for each stage
	concurrencyStages := []int{10, 20, 50}

	var stageResults []ProgressiveStageResult

	// Run each concurrency stage
	for _, concurrency := range concurrencyStages {
		if opts.Verbose {
			fmt.Printf("\n[Test] Starting stage: %d concurrent requests\n", concurrency)
		}

		stageResult := runConcurrencyStage(ctx, concurrency, localPort, opts.Verbose)
		stageResults = append(stageResults, stageResult)

		if opts.Verbose {
			fmt.Printf("[Test] Stage %d concurrent requests completed: %d/%d successful (%.2f%% success rate)\n",
				concurrency, stageResult.SuccessCount, stageResult.TotalReqs, stageResult.SuccessRate)
		}

		// Brief pause between stages
		time.Sleep(5 * time.Second)
	}

	// Print comprehensive summary
	printProgressiveResults(stageResults)

	// Set details for reporting
	if opts.SetDetails != nil {
		details := make(map[string]interface{})
		for _, stage := range stageResults {
			stageKey := fmt.Sprintf("qps_%d", stage.QPS)
			details[stageKey] = map[string]interface{}{
				"total_requests": stage.TotalReqs,
				"successful":     stage.SuccessCount,
				"failed":         stage.FailureCount,
				"success_rate":   fmt.Sprintf("%.2f%%", stage.SuccessRate),
				"avg_duration":   stage.AvgDuration.Milliseconds(),
				"min_duration":   stage.MinDuration.Milliseconds(),
				"max_duration":   stage.MaxDuration.Milliseconds(),
			}
		}
		opts.SetDetails(details)
	}

	return nil
}

func runConcurrencyStage(ctx context.Context, concurrency int, localPort string, verbose bool) ProgressiveStageResult {
	result := ProgressiveStageResult{
		QPS:         concurrency, // Store as QPS field for compatibility
		MinDuration: time.Hour,   // Initialize with large value
	}

	var mu sync.Mutex
	var wg sync.WaitGroup

	if verbose {
		fmt.Printf("[Test] Sending %d concurrent requests...\n", concurrency)
	}

	// Send all requests concurrently
	for i := 1; i <= concurrency; i++ {
		wg.Add(1)

		go func(reqID int) {
			defer wg.Done()

			// Send request
			reqResult := sendSingleRequest(ctx, reqID, localPort, false)

			// Update results
			mu.Lock()
			result.Results = append(result.Results, reqResult)
			result.TotalReqs++
			if reqResult.Success {
				result.SuccessCount++
			} else {
				result.FailureCount++
			}
			mu.Unlock()
		}(i)
	}

	// Wait for all requests to complete
	wg.Wait()

	return calculateStageStats(result)
}

func calculateStageStats(result ProgressiveStageResult) ProgressiveStageResult {
	if result.TotalReqs == 0 {
		return result
	}

	// Calculate success rate
	result.SuccessRate = float64(result.SuccessCount) / float64(result.TotalReqs) * 100

	// Calculate duration statistics
	var totalDuration time.Duration
	for _, r := range result.Results {
		totalDuration += r.Duration
		if r.Duration < result.MinDuration {
			result.MinDuration = r.Duration
		}
		if r.Duration > result.MaxDuration {
			result.MaxDuration = r.Duration
		}
	}

	if len(result.Results) > 0 {
		result.AvgDuration = totalDuration / time.Duration(len(result.Results))
	}

	// Reset MinDuration if it wasn't updated
	if result.MinDuration == time.Hour {
		result.MinDuration = 0
	}

	return result
}

func printProgressiveResults(stageResults []ProgressiveStageResult) {
	separator := strings.Repeat("=", 100)
	fmt.Println("\n" + separator)
	fmt.Println("Progressive Stress Test Results")
	fmt.Println(separator)

	// Print header
	fmt.Printf("%-15s %-15s %-15s %-15s %-15s %-15s %-15s\n",
		"Concurrency", "Total Reqs", "Successful", "Failed", "Success Rate", "Avg Duration", "Max Duration")
	fmt.Println(strings.Repeat("-", 100))

	// Print each stage
	for _, stage := range stageResults {
		fmt.Printf("%-15d %-15d %-15d %-15d %-15s %-15v %-15v\n",
			stage.QPS,
			stage.TotalReqs,
			stage.SuccessCount,
			stage.FailureCount,
			fmt.Sprintf("%.2f%%", stage.SuccessRate),
			stage.AvgDuration.Round(time.Millisecond),
			stage.MaxDuration.Round(time.Millisecond))
	}

	fmt.Println(separator)

	// Print summary statistics
	fmt.Println("\nSummary:")
	totalRequests := 0
	totalSuccess := 0
	for _, stage := range stageResults {
		totalRequests += stage.TotalReqs
		totalSuccess += stage.SuccessCount
	}
	overallSuccessRate := float64(totalSuccess) / float64(totalRequests) * 100
	fmt.Printf("  Overall: %d/%d successful (%.2f%% success rate)\n",
		totalSuccess, totalRequests, overallSuccessRate)

	// Show failures for each stage
	fmt.Println("\nFailures by Stage:")
	for _, stage := range stageResults {
		if stage.FailureCount > 0 {
			fmt.Printf("  %d concurrent requests: %d failures\n", stage.QPS, stage.FailureCount)
			// Show first 3 failures for this stage
			failureCount := 0
			for _, result := range stage.Results {
				if !result.Success && failureCount < 3 {
					failureCount++
					fmt.Printf("    Request #%d: %s (duration: %v)\n",
						result.RequestID, result.ErrorMessage, result.Duration)
				}
				if failureCount >= 3 {
					break
				}
			}
		} else {
			fmt.Printf("  %d concurrent requests: No failures! 🎉\n", stage.QPS)
		}
	}
	fmt.Println()
}
