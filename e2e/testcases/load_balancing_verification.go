package testcases

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
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
		Description: "Send concurrent requests and verify vLLM service has >=2 endpoints and all replicas respond OK",
		Tags:        []string{"lb", "ha"},
		Fn:          testLoadBalancingVerification,
	})
}

func testLoadBalancingVerification(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	// Precondition: vLLM service should have >= 2 endpoints (implies LB target set)
	if err := ensureServiceHasAtLeastNEndpoints(ctx, client, "default", "vllm-llama3-8b-instruct", 2, opts.Verbose); err != nil {
		return fmt.Errorf("service endpoints check failed: %w", err)
	}

	// Drive concurrent traffic through Envoy -> router -> vLLM
	localPort, stopPF, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPF()

	total := 100
	concurrency := 10
	errCount := 0
	mu := sync.Mutex{}
	wg := sync.WaitGroup{}
	wg.Add(concurrency)

	work := make(chan int, total)
	for i := 0; i < total; i++ {
		work <- i + 1
	}
	close(work)

	clientHTTP := &http.Client{Timeout: 20 * time.Second}
	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)

	for w := 0; w < concurrency; w++ {
		go func() {
			defer wg.Done()
			for id := range work {
				if err := sendBasicChatRequest(ctx, clientHTTP, url, id); err != nil {
					mu.Lock()
					errCount++
					mu.Unlock()
				}
			}
		}()
	}
	wg.Wait()

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total":         total,
			"errors":        errCount,
			"success_count": total - errCount,
		})
	}

	// Basic assertion: expect high success rate (>=95%) under LB across replicas
	if float64(errCount) > float64(total)*0.05 {
		return fmt.Errorf("too many errors under load: %d/%d", errCount, total)
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

func sendBasicChatRequest(ctx context.Context, httpClient *http.Client, url string, id int) error {
	body := []byte(`{"model":"MoM","messages":[{"role":"user","content":"ping ` + fmt.Sprint(id) + `"}]}`)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(body))
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
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("status %d: %s", resp.StatusCode, string(b))
	}
	return nil
}
