package testcases

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("agentgateway-traffic-routing", pkgtestcases.TestCase{
		Description: "Test request routing through agentgateway proxy to Semantic Router",
		Tags:        []string{"agentgateway", "gateway", "routing"},
		Fn:          testAgentGatewayTrafficRouting,
	})
}

func testAgentGatewayTrafficRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	const (
		namespace = "agentgateway-system"
		svcName   = "agentgateway-proxy"
		localPort = "8080"
	)

	if opts.Verbose {
		fmt.Printf("[Test] Setting up port-forward to %s/%s\n", namespace, svcName)
	}

	stopFunc, err := helpers.StartPortForward(ctx, client, opts.RestConfig, namespace, svcName, localPort+":80", opts.Verbose)
	if err != nil {
		return fmt.Errorf("failed to start port-forward to agentgateway proxy: %w", err)
	}
	defer stopFunc()

	time.Sleep(2 * time.Second)

	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	payload := `{"model":"auto","messages":[{"role":"user","content":"What is the derivative of f(x) = x^3?"}],"max_tokens":64,"temperature":0}`

	if opts.Verbose {
		fmt.Printf("[Test] Sending POST to %s\n", url)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(payload))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request through agentgateway: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response body: %w", err)
	}

	if opts.Verbose {
		fmt.Printf("[Test] Response status: %d, body length: %d\n", resp.StatusCode, len(body))
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected HTTP 200, got %d: %s", resp.StatusCode, string(body))
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status_code":     resp.StatusCode,
			"response_length": len(body),
		})
	}

	if opts.Verbose {
		fmt.Println("[Test] ✅ Traffic routing through agentgateway successful")
	}

	return nil
}
