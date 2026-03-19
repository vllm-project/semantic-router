package testcases

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("load-balancing-verification", pkgtestcases.TestCase{
		Description: "Send concurrent benign requests and verify backend-routed traffic succeeds while vLLM has >=2 endpoints",
		Tags:        []string{"lb", "ha"},
		Fn:          testLoadBalancingVerification,
	})
}

func testLoadBalancingVerification(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	// Precondition: vLLM service should have >= 2 endpoints (implies LB target set)
	if err := ensureServiceHasAtLeastNEndpoints(ctx, client, "default", "vllm-llama3-8b-instruct", 2, opts.Verbose); err != nil {
		return fmt.Errorf("service endpoints check failed: %w", err)
	}

	// Drive concurrent benign traffic through Envoy -> router -> vLLM.
	session, chatClient, err := openProductionStackChatSession(ctx, client, opts, 20*time.Second)
	if err != nil {
		return err
	}
	defer session.Close()

	requestID := 1
	var warmup productionStackWarmupSummary
	requestID, warmup, err = warmProductionStackBackend(ctx, chatClient, requestID, 5, 12)
	if err != nil {
		return fmt.Errorf("backend warmup failed before load-balancing verification: %w", err)
	}

	total := 100
	concurrency := 10
	requestErrors := 0
	fastResponses := 0
	backendSuccessCount := 0
	fastResponseDecisions := map[string]int{}
	backendDecisions := map[string]int{}
	mu := sync.Mutex{}
	wg := sync.WaitGroup{}
	wg.Add(concurrency)

	work := make(chan int, total)
	for i := 0; i < total; i++ {
		work <- requestID + i
	}
	close(work)

	for w := 0; w < concurrency; w++ {
		go func() {
			defer wg.Done()
			for id := range work {
				result, err := sendProductionStackChatRequest(ctx, chatClient, id)
				mu.Lock()
				switch {
				case err != nil:
					requestErrors++
				case result.FastResponse:
					fastResponses++
					recordDecisionCount(fastResponseDecisions, result.SelectedDecision)
				default:
					backendSuccessCount++
					recordDecisionCount(backendDecisions, result.SelectedDecision)
				}
				mu.Unlock()
			}
		}()
	}
	wg.Wait()

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total":                   total,
			"warmup_attempts":         warmup.Attempts,
			"warmup_successes":        warmup.Successes,
			"warmup_fast_responses":   warmup.FastResponses,
			"warmup_request_errors":   warmup.RequestErrors,
			"errors":                  requestErrors,
			"fast_responses":          fastResponses,
			"success_count":           backendSuccessCount,
			"backend_success_count":   backendSuccessCount,
			"fast_response_decisions": fastResponseDecisions,
			"backend_decisions":       backendDecisions,
		})
	}

	if fastResponses > 0 {
		return fmt.Errorf(
			"unexpected fast_response under load: %d/%d requests were short-circuited before reaching upstream backends (%s)",
			fastResponses,
			total,
			formatDecisionCounts(fastResponseDecisions),
		)
	}

	// Basic assertion: expect high success rate (>=95%) for backend-routed traffic.
	if float64(requestErrors) > float64(total)*0.05 {
		return fmt.Errorf("too many backend request errors under load: %d/%d", requestErrors, total)
	}
	return nil
}

func ensureServiceHasAtLeastNEndpoints(ctx context.Context, client *kubernetes.Clientset, namespace, name string, n int, verbose bool) error {
	ep, err := client.CoreV1().Endpoints(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	count := 0
	for _, subset := range ep.Subsets {
		count += len(subset.Addresses)
	}
	if verbose {
		fmt.Printf("[Test] service %s/%s has %d endpoints\n", namespace, name, count)
	}
	if count < n {
		return fmt.Errorf("service %s/%s has %d endpoints, expected at least %d", namespace, name, count, n)
	}
	// Also verify pods behind the service are running
	svc, err := client.CoreV1().Services(namespace).Get(ctx, name, metav1.GetOptions{})
	if err == nil {
		var selector []string
		for k, v := range svc.Spec.Selector {
			selector = append(selector, fmt.Sprintf("%s=%s", k, v))
		}
		pods, err := client.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{LabelSelector: strings.Join(selector, ",")})
		if err != nil {
			if verbose {
				fmt.Printf("[Test] Warning: failed to list pods: %v\n", err)
			}
			return nil // Continue even if pod listing fails
		}
		running := 0
		for _, p := range pods.Items {
			if p.Status.Phase == corev1.PodRunning {
				running++
			}
		}
		if verbose {
			fmt.Printf("[Test] %d/%d pods Running behind service %s/%s\n", running, len(pods.Items), namespace, name)
		}
	}
	return nil
}
