package testcases

import (
	"context"
	"fmt"
	"sort"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("performance-throughput", pkgtestcases.TestCase{
		Description: "Measure throughput and latency for benign backend-routed traffic under moderate load",
		Tags:        []string{"performance"},
		Fn:          testPerformanceThroughput,
	})
}

func testPerformanceThroughput(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, chatClient, err := openProductionStackChatSession(ctx, client, opts, 20*time.Second)
	if err != nil {
		return err
	}
	defer session.Close()

	// Fixed-run benchmark: send benign backend-routed requests sequentially and compute metrics.
	N := 100
	start := time.Now()
	var latencies []time.Duration
	requestErrors := 0
	fastResponses := 0
	backendSuccessCount := 0
	fastResponseDecisions := map[string]int{}
	backendDecisions := map[string]int{}
	for i := 0; i < N; i++ {
		t0 := time.Now()
		result, err := sendProductionStackChatRequest(ctx, chatClient, i+1)
		switch {
		case err != nil:
			requestErrors++
		case result.FastResponse:
			fastResponses++
			recordDecisionCount(fastResponseDecisions, result.SelectedDecision)
		default:
			backendSuccessCount++
			recordDecisionCount(backendDecisions, result.SelectedDecision)
			latencies = append(latencies, time.Since(t0))
		}
	}
	elapsed := time.Since(start)
	rps := float64(backendSuccessCount) / elapsed.Seconds()
	p50, p95, p99 := percentile(latencies, 50), percentile(latencies, 95), percentile(latencies, 99)

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total":                   N,
			"errors":                  requestErrors,
			"fast_responses":          fastResponses,
			"backend_success_count":   backendSuccessCount,
			"backend_decisions":       backendDecisions,
			"fast_response_decisions": fastResponseDecisions,
			"rps":                     rps,
			"p50_ms":                  p50.Milliseconds(),
			"p95_ms":                  p95.Milliseconds(),
			"p99_ms":                  p99.Milliseconds(),
		})
	}

	if fastResponses > 0 {
		return fmt.Errorf(
			"unexpected fast_response during throughput run: %d/%d requests were short-circuited before reaching upstream backends (%s)",
			fastResponses,
			N,
			formatDecisionCounts(fastResponseDecisions),
		)
	}

	if requestErrors > 0 {
		return fmt.Errorf("observed %d backend request errors during performance run", requestErrors)
	}
	return nil
}

func percentile(latencies []time.Duration, p int) time.Duration {
	if len(latencies) == 0 {
		return 0
	}
	// Copy and sort using efficient O(n log n) algorithm
	tmp := make([]time.Duration, len(latencies))
	copy(tmp, latencies)
	sort.Slice(tmp, func(i, j int) bool { return tmp[i] < tmp[j] })
	idx := (p * (len(tmp) - 1)) / 100
	return tmp[idx]
}
