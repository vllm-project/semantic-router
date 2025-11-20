package testcases

import (
	"context"
	"fmt"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("llmd-failover-recovery", pkgtestcases.TestCase{
		Description: "Traffic survives backend pod loss",
		Tags:        []string{"llmd", "failover"},
		Fn:          llmdFailover,
	})
}

func llmdFailover(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	pods, err := client.CoreV1().Pods("default").List(ctx, metav1.ListOptions{LabelSelector: "app=phi4-mini"})
	if err != nil {
		return err
	}
	if len(pods.Items) < 2 {
		return fmt.Errorf("need >=2 phi4-mini pods for failover, got %d", len(pods.Items))
	}
	target := pods.Items[0].Name
	if err := client.CoreV1().Pods("default").Delete(ctx, target, metav1.DeleteOptions{}); err != nil {
		return err
	}
	deleteTime := time.Now()

	time.Sleep(5 * time.Second)

	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	deadline := time.Now().Add(60 * time.Second)
	total := 0
	success := 0
	podHits := map[string]int{}
	var recoveredAt time.Time

	for time.Now().Before(deadline) {
		total++
		res, err := doLLMDChat(ctx, localPort, "phi4-mini", fmt.Sprintf("failover-%d", total), 45*time.Second)
		if err == nil {
			success++
			pod := getInferencePod(res.headers)
			if pod == target {
				return fmt.Errorf("traffic routed to deleted pod %s", target)
			}
			if pod != "" {
				podHits[pod]++
			}
			if recoveredAt.IsZero() {
				recoveredAt = time.Now()
			}
		}
		time.Sleep(1 * time.Second)
	}
	rate := float64(success) / float64(total)
	if rate < 0.95 {
		return fmt.Errorf("success rate %.2f below 0.95", rate)
	}
    if len(podHits) == 0 {
        ep, err := client.CoreV1().Endpoints("default").Get(ctx, "phi4-mini", metav1.GetOptions{})
        if err != nil {
            return err
        }
        for _, s := range ep.Subsets {
            for _, a := range s.Addresses {
                if a.TargetRef != nil && a.TargetRef.Name == target {
                    return fmt.Errorf("deleted pod still present in endpoints %s", target)
                }
            }
        }
    }
	recoverySeconds := time.Since(deleteTime).Seconds()
	if !recoveredAt.IsZero() {
		recoverySeconds = recoveredAt.Sub(deleteTime).Seconds()
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"deleted_pod":      target,
			"success":          success,
			"total":            total,
			"success_rate":     rate,
			"pod_hits":         podHits,
			"recovery_seconds": recoverySeconds,
		})
	}
	return nil
}
