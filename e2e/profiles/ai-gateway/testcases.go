package aigateway

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	// Register test cases for AI Gateway profile
	testcases.Register("basic-health-check", testcases.TestCase{
		Description: "Verify that all components are deployed and healthy",
		Tags:        []string{"ai-gateway", "health"},
		Fn:          testBasicHealthCheck,
	})

	testcases.Register("chat-completions-request", testcases.TestCase{
		Description: "Send a chat completions request and verify 200 OK response",
		Tags:        []string{"ai-gateway", "functional"},
		Fn:          testChatCompletionsRequest,
	})
}

func testBasicHealthCheck(ctx context.Context, client *kubernetes.Clientset, opts testcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Running basic health check")
	}

	// Check semantic-router deployment
	if err := checkDeployment(ctx, client, "vllm-semantic-router-system", "semantic-router", opts.Verbose); err != nil {
		return fmt.Errorf("semantic-router deployment not healthy: %w", err)
	}

	// Check envoy-gateway deployment
	if err := checkDeployment(ctx, client, "envoy-gateway-system", "envoy-gateway", opts.Verbose); err != nil {
		return fmt.Errorf("envoy-gateway deployment not healthy: %w", err)
	}

	// Check ai-gateway-controller deployment
	if err := checkDeployment(ctx, client, "envoy-ai-gateway-system", "ai-gateway-controller", opts.Verbose); err != nil {
		return fmt.Errorf("ai-gateway-controller deployment not healthy: %w", err)
	}

	if opts.Verbose {
		fmt.Println("[Test] ✅ All deployments are healthy")
	}

	return nil
}

func testChatCompletionsRequest(ctx context.Context, client *kubernetes.Clientset, opts testcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing chat completions endpoint")
	}

	// Get the Envoy service name
	envoyService, err := getEnvoyServiceName(ctx, opts.Verbose)
	if err != nil {
		return fmt.Errorf("failed to get Envoy service name: %w", err)
	}

	if opts.Verbose {
		fmt.Printf("[Test] Envoy service: %s\n", envoyService)
	}

	// Set up port forwarding
	portForwardCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	if err := startPortForward(portForwardCtx, "envoy-gateway-system", envoyService, "8080:80", opts.Verbose); err != nil {
		return fmt.Errorf("failed to start port forward: %w", err)
	}

	// Wait for port forward to be ready
	time.Sleep(3 * time.Second)

	// Send test request
	if opts.Verbose {
		fmt.Println("[Test] Sending chat completions request")
	}

	payload := `{
		"model": "MoM",
		"messages": [
			{"role": "user", "content": "What is the derivative of f(x) = x^3?"}
		]
	}`

	req, err := http.NewRequestWithContext(ctx, "POST", "http://localhost:8080/v1/chat/completions", strings.NewReader(payload))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	if opts.Verbose {
		fmt.Printf("[Test] Response status: %d\n", resp.StatusCode)
		fmt.Printf("[Test] Response body: %s\n", string(body))
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected status 200, got %d: %s", resp.StatusCode, string(body))
	}

	if opts.Verbose {
		fmt.Println("[Test] ✅ Chat completions request successful")
	}

	return nil
}

func checkDeployment(ctx context.Context, client *kubernetes.Clientset, namespace, name string, verbose bool) error {
	deployment, err := client.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return err
	}

	if deployment.Status.ReadyReplicas == 0 {
		return fmt.Errorf("no ready replicas")
	}

	if verbose {
		fmt.Printf("[Test] Deployment %s/%s: %d/%d replicas ready\n",
			namespace, name, deployment.Status.ReadyReplicas, *deployment.Spec.Replicas)
	}

	return nil
}

func getEnvoyServiceName(ctx context.Context, verbose bool) (string, error) {
	cmd := exec.CommandContext(ctx, "kubectl", "get", "svc",
		"-n", "envoy-gateway-system",
		"--selector=gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router",
		"-o", "jsonpath={.items[0].metadata.name}")

	output, err := cmd.Output()
	if err != nil {
		return "", err
	}

	serviceName := strings.TrimSpace(string(output))
	if serviceName == "" {
		return "", fmt.Errorf("no Envoy service found")
	}

	return serviceName, nil
}

func startPortForward(ctx context.Context, namespace, service, ports string, verbose bool) error {
	cmd := exec.CommandContext(ctx, "kubectl", "port-forward",
		"-n", namespace,
		fmt.Sprintf("svc/%s", service),
		ports)

	if verbose {
		fmt.Printf("[Test] Starting port forward: kubectl port-forward -n %s svc/%s %s\n", namespace, service, ports)
	}

	// Start port forward in background
	if err := cmd.Start(); err != nil {
		return err
	}

	// Wait a bit for port forward to establish
	time.Sleep(2 * time.Second)

	return nil
}
