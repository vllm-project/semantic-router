package testcases

import (
	"context"
	"fmt"
	"sync"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("llmd-performance-baseline", pkgtestcases.TestCase{
		Description: "Measure success rate under moderate concurrency",
		Tags:        []string{"llmd", "perf"},
		Fn:          llmdPerf,
	})
}

type perfResult struct {
	concurrency int
	success     int
	total       int
	durations   []time.Duration
}

func llmdPerf(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	stages := []int{15, 30, 60}
	results := []perfResult{}

	for _, conc := range stages {
		res := runPerfStage(ctx, localPort, conc, 20*time.Second)
		if float64(res.success)/float64(res.total) < 0.95 {
			return fmt.Errorf("stage %d success %d/%d", conc, res.success, res.total)
		}
		results = append(results, res)
		time.Sleep(2 * time.Second)
	}

	if opts.SetDetails != nil {
		summary := map[string]interface{}{}
		for _, r := range results {
			p50, p95 := percentileDuration(r.durations, 0.5), percentileDuration(r.durations, 0.95)
			key := fmt.Sprintf("c%d", r.concurrency)
			summary[key] = map[string]interface{}{
				"success":      r.success,
				"total":        r.total,
				"success_rate": float64(r.success) / float64(r.total),
				"p50_ms":       p50.Milliseconds(),
				"p95_ms":       p95.Milliseconds(),
			}
		}
		opts.SetDetails(summary)
	}
	return nil
}

func runPerfStage(ctx context.Context, port string, conc int, duration time.Duration) perfResult {
	res := perfResult{concurrency: conc}
	stageCtx, cancel := context.WithTimeout(ctx, duration)
	defer cancel()

	var wg sync.WaitGroup
	var mu sync.Mutex

	for i := 0; i < conc; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stageCtx.Done():
					return
				default:
				}
				resItem, err := doLLMDChat(stageCtx, port, "auto", fmt.Sprintf("perf-%d-%d", conc, i), 60*time.Second)
				if err != nil && stageCtx.Err() != nil {
					continue
				}
				// On transient failure, retry once within stage window
				if err != nil {
					resItem2, err2 := doLLMDChat(stageCtx, port, "auto", fmt.Sprintf("perf-%d-%d", conc, i), 60*time.Second)
					if err2 == nil {
						err = nil
						resItem = resItem2
					}
				}
				mu.Lock()
				res.total++
				if err == nil {
					res.success++
					res.durations = append(res.durations, resItem.duration)
				}
				mu.Unlock()
			}
		}()
	}

	wg.Wait()
	return res
}
