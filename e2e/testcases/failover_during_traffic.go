package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync/atomic"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("failover-during-traffic", pkgtestcases.TestCase{
		Description: "While sending requests, delete one vLLM pod and verify high availability and recovery",
		Tags:        []string{"ha", "failover"},
		Fn:          testFailoverDuringTraffic,
	})
}

func testFailoverDuringTraffic(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPF, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPF()

	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	httpClient := &http.Client{Timeout: 15 * time.Second}

	// Start sending requests in background
	total := 100
	var errs int32 // Use atomic for thread-safe access
	done := make(chan struct{})
	go func() {
		for i := 0; i < total; i++ {
			if err := sendChat(ctx, httpClient, url, i); err != nil {
				atomic.AddInt32(&errs, 1)
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

	errCount := int(atomic.LoadInt32(&errs))
	successRate := float64(total-errCount) / float64(total)
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total":        total,
			"errors":       errCount,
			"success_rate": successRate,
		})
	}
	// Expect >= 0.95 success rate despite disruption
	if successRate < 0.95 {
		return fmt.Errorf("success rate too low during failover: %.2f", successRate)
	}

	// Ensure deployment recovers to 2 ready replicas
	if err := waitDeploymentReadyReplicas(ctx, client, "default", "vllm-llama3-8b-instruct", 2, 5*time.Minute, opts.Verbose); err != nil {
		return fmt.Errorf("vllm demo did not recover: %w", err)
	}
	return nil
}

func sendChat(ctx context.Context, httpClient *http.Client, url string, id int) error {
	reqBody := map[string]interface{}{
		"model": "MoM",
		"messages": []map[string]string{
			{"role": "user", "content": fmt.Sprintf("check %d", id)},
		},
	}
	b, _ := json.Marshal(reqBody)
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
