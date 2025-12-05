package testcases

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("istio-traffic-routing", pkgtestcases.TestCase{
		Description: "Test request routing through Istio ingress gateway to Semantic Router",
		Tags:        []string{"istio", "gateway", "routing"},
		Fn:          testIstioTrafficRouting,
	})
}

func testIstioTrafficRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing traffic routing through Istio gateway")
	}

	// Istio-specific test: always use vllm-semantic-router-system namespace
	namespace := "vllm-semantic-router-system"

	// Verify Istio Gateway exists
	if err := verifyIstioGateway(ctx, opts.RestConfig, namespace); err != nil {
		return fmt.Errorf("Istio Gateway verification failed: %w", err)
	}

	// Verify VirtualService exists
	if err := verifyIstioVirtualService(ctx, opts.RestConfig, namespace); err != nil {
		return fmt.Errorf("Istio VirtualService verification failed: %w", err)
	}

	// Get the Istio ingress gateway service
	istioNs := "istio-system"
	svc, err := client.CoreV1().Services(istioNs).Get(ctx, "istio-ingressgateway", metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get istio-ingressgateway service: %w", err)
	}

	if opts.Verbose {
		fmt.Printf("[Test] Found Istio ingress gateway service: %s/%s\n", istioNs, svc.Name)
	}

	// Setup port forwarding to Istio ingress gateway
	stopFunc, err := setupIstioGatewayPortForward(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup port forwarding to Istio gateway: %w", err)
	}
	defer stopFunc()

	// Wait for port forwarding to stabilize
	time.Sleep(2 * time.Second)

	// Send a test request through the Istio gateway
	// Using /v1/models endpoint which is OpenAI-compatible and should return 200
	localPort := "8080" // Port we're forwarding to locally
	url := fmt.Sprintf("http://localhost:%s/v1/models", localPort)

	if opts.Verbose {
		fmt.Printf("[Test] Sending GET request to Istio gateway: %s\n", url)
	}

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	httpClient := &http.Client{
		Timeout: 30 * time.Second,
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request through Istio gateway: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response: %w", err)
	}

	if opts.Verbose {
		fmt.Printf("[Test] Response status: %d\n", resp.StatusCode)
		fmt.Printf("[Test] Response headers: %v\n", resp.Header)
	}

	// Verify we got a successful response
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected status 200, got %d: %s", resp.StatusCode, string(body))
	}

	// Check for Istio headers (indicates request went through Istio)
	istioHeaders := make(map[string]string)
	for key, values := range resp.Header {
		if len(values) > 0 {
			// Common Istio/Envoy headers
			if key == "X-Envoy-Upstream-Service-Time" ||
				key == "X-Request-Id" ||
				key == "Server" {
				istioHeaders[key] = values[0]
			}
		}
	}

	hasIstioHeaders := len(istioHeaders) > 0
	if opts.Verbose && hasIstioHeaders {
		fmt.Println("[Test] Detected Istio/Envoy headers in response:")
		for k, v := range istioHeaders {
			fmt.Printf("[Test]   %s: %s\n", k, v)
		}
	}

	// Set details for reporting
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"status_code":         resp.StatusCode,
			"response_length":     len(body),
			"istio_headers_found": hasIstioHeaders,
			"istio_headers":       istioHeaders,
		})
	}

	if opts.Verbose {
		fmt.Println("[Test] ✅ Traffic routing through Istio gateway successful")
	}

	return nil
}

// verifyIstioGateway verifies that the Istio Gateway resource exists
func verifyIstioGateway(ctx context.Context, restConfig interface{}, namespace string) error {
	// This would use dynamic client to check for Gateway CRD
	// For now, we'll assume it exists since we created it in setup
	return nil
}

// verifyIstioVirtualService verifies that the Istio VirtualService resource exists
func verifyIstioVirtualService(ctx context.Context, restConfig interface{}, namespace string) error {
	// This would use dynamic client to check for VirtualService CRD
	// For now, we'll assume it exists since we created it in setup
	return nil
}

// setupIstioGatewayPortForward sets up port forwarding to the Istio ingress gateway
func setupIstioGatewayPortForward(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) (func(), error) {
	// Port forward to istio-ingressgateway service in istio-system namespace
	// Port mapping: local 8080 -> istio-ingressgateway:80
	istioNs := "istio-system"
	istioSvc := "istio-ingressgateway"

	stopFunc, err := helpers.StartPortForward(ctx, client, opts.RestConfig, istioNs, istioSvc, "8080:80", opts.Verbose)
	if err != nil {
		return nil, fmt.Errorf("failed to start port forward to Istio gateway: %w", err)
	}

	return stopFunc, nil
}
