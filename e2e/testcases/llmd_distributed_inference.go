package testcases

import (
	"context"
	"fmt"
	"sync"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("llmd-distributed-inference", pkgtestcases.TestCase{
		Description: "Verify multi-replica backends serve requests",
		Tags:        []string{"llmd", "distributed"},
		Fn:          llmdDistributed,
	})
}

func llmdDistributed(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	backendDeploys := []string{"vllm-llama3-8b-instruct", "phi4-mini"}
	for _, name := range backendDeploys {
		dep, err := client.AppsV1().Deployments("default").Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if dep.Status.ReadyReplicas < 2 {
			return fmt.Errorf("%s ready replicas %d < 2", name, dep.Status.ReadyReplicas)
		}
	}

	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	const total = 30
	var (
		success int
		mu      sync.Mutex
		podHits = map[string]int{}
	)
	var wg sync.WaitGroup

	for i := 0; i < total; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
            res, err := doLLMDChat(ctx, localPort, "llama3-8b", fmt.Sprintf("req-%d", i), 60*time.Second)
			if err != nil {
				return
			}
			pod := getInferencePod(res.headers)
			if pod == "" {
				return
			}
			mu.Lock()
			success++
			podHits[pod]++
			mu.Unlock()
		}()
	}

	wg.Wait()

	successRate := float64(success) / float64(total)
	if successRate < 0.98 {
		return fmt.Errorf("success rate %.2f below 0.98", successRate)
	}
	if len(podHits) < 2 {
		return fmt.Errorf("expected hits on >=2 pods, got %d", len(podHits))
	}
	var max, min int
	for _, c := range podHits {
		if c > max {
			max = c
		}
		if min == 0 || c < min {
			min = c
		}
	}
	if min == 0 || float64(max)/float64(min) > 2.0 {
		return fmt.Errorf("pod hit imbalance max/min=%d/%d", max, min)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"success_rate": successRate,
			"total":        total,
			"pod_hits":     podHits,
		})
	}
	return nil
}
