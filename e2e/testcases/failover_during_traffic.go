package testcases

import (
	"context"
	"fmt"
	"sync/atomic"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("failover-during-traffic", pkgtestcases.TestCase{
		Description: "While sending benign backend-routed requests, delete one vLLM pod and verify failover and recovery",
		Tags:        []string{"ha", "failover"},
		Fn:          testFailoverDuringTraffic,
	})
}

func testFailoverDuringTraffic(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, chatClient, err := openProductionStackChatSession(ctx, client, opts, 20*time.Second)
	if err != nil {
		return err
	}
	defer session.Close()

	// Start sending requests in background
	total := 100
	var requestErrors int32
	var fastResponses int32
	var backendSuccessCount int32
	done := make(chan struct{})
	fastResponseDecisions := map[string]int{}
	go func() {
		for i := 0; i < total; i++ {
			result, err := sendProductionStackChatRequest(ctx, chatClient, i+1)
			switch {
			case err != nil:
				atomic.AddInt32(&requestErrors, 1)
			case result.FastResponse:
				atomic.AddInt32(&fastResponses, 1)
				recordDecisionCount(fastResponseDecisions, result.SelectedDecision)
			default:
				atomic.AddInt32(&backendSuccessCount, 1)
			}
			time.Sleep(100 * time.Millisecond)
		}
		close(done)
	}()

	// Wait a short moment then delete one vLLM pod
	time.Sleep(2 * time.Second)
	pods, err := client.CoreV1().Pods("default").List(ctx, metav1.ListOptions{LabelSelector: "app=vllm-llama3-8b-instruct"})
	if err == nil && len(pods.Items) > 0 {
		if err := client.CoreV1().Pods("default").Delete(ctx, pods.Items[0].Name, metav1.DeleteOptions{}); err != nil {
			// Log but don't fail - pod might already be deleted
			if opts.Verbose {
				fmt.Printf("[Test] Warning: failed to delete pod: %v\n", err)
			}
		}
	}

	<-done

	errCount := int(atomic.LoadInt32(&requestErrors))
	fastResponseCount := int(atomic.LoadInt32(&fastResponses))
	successCount := int(atomic.LoadInt32(&backendSuccessCount))
	backendSuccessRate := float64(successCount) / float64(total)
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total":                   total,
			"errors":                  errCount,
			"fast_responses":          fastResponseCount,
			"success_count":           successCount,
			"backend_success_rate":    backendSuccessRate,
			"backend_success_count":   successCount,
			"fast_response_decisions": fastResponseDecisions,
		})
	}

	if fastResponseCount > 0 {
		return fmt.Errorf(
			"unexpected fast_response during failover traffic: %d/%d requests were short-circuited before reaching upstream backends (%s)",
			fastResponseCount,
			total,
			formatDecisionCounts(fastResponseDecisions),
		)
	}

	// Expect >= 0.95 backend success rate despite disruption.
	if backendSuccessRate < 0.95 {
		return fmt.Errorf("backend success rate too low during failover: %.2f", backendSuccessRate)
	}

	// Ensure deployment recovers to 2 ready replicas
	if err := waitDeploymentReadyReplicas(ctx, client, "default", "vllm-llama3-8b-instruct", 2, 5*time.Minute, opts.Verbose); err != nil {
		return fmt.Errorf("vllm demo did not recover: %w", err)
	}
	return nil
}
