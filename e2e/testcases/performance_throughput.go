package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("performance-throughput", pkgtestcases.TestCase{
		Description: "Measure throughput and latency under moderate load",
		Tags:        []string{"performance"},
		Fn:          testPerformanceThroughput,
	})
}

func testPerformanceThroughput(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPF, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPF()

	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	httpClient := &http.Client{Timeout: 20 * time.Second}

	// Fixed-run benchmark: send N requests sequentially and compute metrics
	N := 100
	start := time.Now()
	var latencies []time.Duration
	errors := 0
	for i := 0; i < N; i++ {
		t0 := time.Now()
		if err := sendSmallChat(ctx, httpClient, url, i); err != nil {
			errors++
		}
		latencies = append(latencies, time.Since(t0))
	}
	elapsed := time.Since(start)
	rps := float64(N) / elapsed.Seconds()
	p50, p95, p99 := percentile(latencies, 50), percentile(latencies, 95), percentile(latencies, 99)

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total":  N,
			"errors": errors,
			"rps":    rps,
			"p50_ms": p50.Milliseconds(),
			"p95_ms": p95.Milliseconds(),
			"p99_ms": p99.Milliseconds(),
		})
	}
	// Simple gates suitable for CI; can be tuned
	if errors > 0 {
		return fmt.Errorf("observed %d errors during performance run", errors)
	}
	return nil
}

func sendSmallChat(ctx context.Context, httpClient *http.Client, url string, id int) error {
	body := map[string]interface{}{
		"model": "MoM",
		"messages": []map[string]string{
			{"role": "user", "content": "hi"},
		},
	}
	b, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(b))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("status %d", resp.StatusCode)
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
