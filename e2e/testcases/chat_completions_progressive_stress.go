package testcases

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("chat-completions-progressive-stress", pkgtestcases.TestCase{
		Priority:    pkgtestcases.P2,
		Description: "10/20/50 concurrent requests, each answered by the Router inside the gateway's deadline, then a drained queue",
		Tags:        []string{"llm", "stress", "progressive", "concurrency"},
		Fn:          testProgressiveStress,
	})
}

const (
	// progressiveStressAnswerBound is the gateway's ext_proc message_timeout
	// (deploy/kubernetes/ai-gateway). The Router bounds a request's model
	// signals at 45 s so that it still answers inside it.
	progressiveStressAnswerBound = 60 * time.Second
	// progressiveStressRequestTimeout outlasts that deadline, so a request fails
	// only when the gateway or the Router fails it.
	progressiveStressRequestTimeout = 90 * time.Second
	// progressiveStressDrainBound is how long the Router may take, after the
	// last stage, to answer a fresh request in about its unloaded time again.
	// The model runtime keeps evaluating signals the Router stopped waiting
	// for, so requests that arrive after a burst queue behind that work.
	progressiveStressDrainBound = 5 * time.Minute
	// progressiveStressBaselineSamples fresh sequential answers set the
	// Router's unloaded latency before the stages.
	progressiveStressBaselineSamples = 3
	baselineRequestID                = 1000
	drainRequestID                   = 2000
	// responsePathCache and responsePathError are x-vsr-response-path values:
	// a cache hit evaluates no model, and the error path is no answer.
	responsePathCache = "cache"
	responsePathError = "error"
)

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
	// ResponsePaths counts the answers by x-vsr-response-path, and Undecided
	// the answers without a selected decision, such as default routes after a
	// signal error.
	ResponsePaths map[string]int
	Undecided     int
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

	baseline, err := measureBaselineLatency(ctx, localPort)
	if err != nil {
		return err
	}

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

	drained, drainErr := waitForDrain(ctx, localPort, baseline)
	if opts.Verbose && drainErr == nil {
		fmt.Printf("[Test] The Router answered in its unloaded time (%v) again %v after the last stage\n",
			baseline.Round(time.Millisecond), drained.Round(time.Second))
	}

	// Print comprehensive summary
	printProgressiveResults(stageResults)

	// Set details for reporting
	if opts.SetDetails != nil {
		details := make(map[string]interface{})
		totalRequests := 0
		totalSuccess := 0
		for _, stage := range stageResults {
			stageKey := fmt.Sprintf("qps_%d", stage.QPS)
			totalRequests += stage.TotalReqs
			totalSuccess += stage.SuccessCount
			details[stageKey] = map[string]interface{}{
				"total_requests":  stage.TotalReqs,
				"successful":      stage.SuccessCount,
				"failed":          stage.FailureCount,
				"success_rate":    fmt.Sprintf("%.2f%%", stage.SuccessRate),
				"avg_duration":    stage.AvgDuration.Milliseconds(),
				"min_duration":    stage.MinDuration.Milliseconds(),
				"max_duration":    stage.MaxDuration.Milliseconds(),
				"response_paths":  stage.ResponsePaths,
				"undecided":       stage.Undecided,
				"sample_failures": sampleStressFailures(stage.Results, 3),
			}
		}
		details["overall"] = map[string]interface{}{
			"total_requests": totalRequests,
			"successful":     totalSuccess,
			"failed":         totalRequests - totalSuccess,
			"success_rate":   fmt.Sprintf("%.2f%%", percentage(totalSuccess, totalRequests)),
		}
		details["drain"] = map[string]interface{}{
			"unloaded_latency_ms": baseline.Milliseconds(),
			"drained_after_ms":    drained.Milliseconds(),
			"bound_ms":            progressiveStressDrainBound.Milliseconds(),
		}
		opts.SetDetails(details)
	}

	return drainErr
}

// measureBaselineLatency returns the median latency of fresh sequential
// answers that evaluated the models, which a cache hit skips.
func measureBaselineLatency(ctx context.Context, localPort string) (time.Duration, error) {
	var samples []time.Duration
	for id := baselineRequestID; len(samples) < progressiveStressBaselineSamples; id++ {
		if id == baselineRequestID+3*progressiveStressBaselineSamples {
			return 0, fmt.Errorf("only %d of %d unloaded requests evaluated the models; the rest were cache hits",
				len(samples), id-baselineRequestID)
		}
		result := sendSingleRequest(ctx, id, localPort, progressiveStressRequestTimeout)
		judgeProgressiveAnswer(&result)
		if !result.Success {
			return 0, fmt.Errorf("unloaded request #%d failed before the stages: %s", id, result.ErrorMessage)
		}
		if result.ResponsePath != responsePathCache {
			samples = append(samples, result.Duration)
		}
	}
	return medianDuration(samples), nil
}

// waitForDrain sends fresh sequential requests until the Router answers one in
// about its unloaded time, and returns how long that took after the last stage.
// A much faster answer evaluated no model, as when the runtime refuses work it
// has no room for, so it does not show that the queue drained.
func waitForDrain(ctx context.Context, localPort string, baseline time.Duration) (time.Duration, error) {
	fastest, slowest := baseline/2, 2*baseline+time.Second
	start := time.Now()
	for id := drainRequestID; time.Since(start) < progressiveStressDrainBound; id++ {
		result := sendSingleRequest(ctx, id, localPort, progressiveStressRequestTimeout)
		judgeProgressiveAnswer(&result)
		if result.Success && result.ResponsePath != responsePathCache &&
			result.Duration >= fastest && result.Duration <= slowest {
			return time.Since(start), nil
		}
		if err := ctx.Err(); err != nil {
			return time.Since(start), err
		}
	}
	return time.Since(start), fmt.Errorf(
		"the Router did not answer in its unloaded time (%v) within %v after the last stage",
		baseline.Round(time.Millisecond), progressiveStressDrainBound)
}

func medianDuration(samples []time.Duration) time.Duration {
	sorted := append([]time.Duration(nil), samples...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	return sorted[len(sorted)/2]
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

			reqResult := sendSingleRequest(ctx, reqID, localPort, progressiveStressRequestTimeout)
			judgeProgressiveAnswer(&reqResult)

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

// judgeProgressiveAnswer accepts a 200 only when the Router produced it, as its
// x-vsr-response-path shows, inside the gateway's deadline. Under overload the
// Router answers slow or refused signals through their declared policies,
// such as a fast block or a default route; those are still its answers, and
// the stage reports their paths.
func judgeProgressiveAnswer(result *StressTestResult) {
	switch {
	case !result.Success:
	case result.ResponsePath == "":
		result.Success = false
		result.ErrorMessage = "status 200 without x-vsr-response-path"
	case result.ResponsePath == responsePathError:
		result.Success = false
		result.ErrorMessage = "status 200 on the Router's error path"
	case result.Duration > progressiveStressAnswerBound:
		result.Success = false
		result.ErrorMessage = fmt.Sprintf(
			"answered after %v, past the gateway's %v deadline",
			result.Duration.Round(time.Millisecond), progressiveStressAnswerBound)
	}
}

func calculateStageStats(result ProgressiveStageResult) ProgressiveStageResult {
	if result.TotalReqs == 0 {
		return result
	}

	// Calculate success rate
	result.SuccessRate = float64(result.SuccessCount) / float64(result.TotalReqs) * 100

	// Calculate duration statistics
	var totalDuration time.Duration
	result.ResponsePaths = map[string]int{}
	for _, r := range result.Results {
		if r.Success {
			result.ResponsePaths[r.ResponsePath]++
			if r.Decision == "" {
				result.Undecided++
			}
		}
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

	fmt.Println("\nAnswers by response path:")
	for _, stage := range stageResults {
		paths := make([]string, 0, len(stage.ResponsePaths))
		for path, count := range stage.ResponsePaths {
			paths = append(paths, fmt.Sprintf("%s=%d", path, count))
		}
		sort.Strings(paths)
		fmt.Printf("  %d concurrent requests: %s; %d without a selected decision\n",
			stage.QPS, strings.Join(paths, " "), stage.Undecided)
	}

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

func sampleStressFailures(results []StressTestResult, limit int) []map[string]interface{} {
	failures := make([]map[string]interface{}, 0, limit)
	for _, result := range results {
		if result.Success {
			continue
		}
		failures = append(failures, map[string]interface{}{
			"request_id":    result.RequestID,
			"status_code":   result.StatusCode,
			"duration_ms":   result.Duration.Milliseconds(),
			"error_message": result.ErrorMessage,
		})
		if len(failures) >= limit {
			break
		}
	}
	return failures
}

func percentage(passed, total int) float64 {
	if total == 0 {
		return 0
	}
	return float64(passed) / float64(total) * 100
}
